"""Step 3: cut frames out of the packed videos and save one image per sample.

Each sample is a short clip of `frames_per_clip` frames. The frames are tiled
into one grid image (left to right, then top to bottom, in time order).
One image instead of several keeps the number of vision tokens fixed and small,
which keeps training memory and speed predictable on a 16 GB card.

Videos are AV1. PyAV decodes them with libdav1d. We seek to the nearest
keyframe before each target time and decode forward to the exact frame.
"""
import math

import av
import pandas as pd
from PIL import Image

from common import config_from_argv, load_episodes

SPLITS = ["random_clip", "random_episode", "held_out_operator", "held_out_robot"]


def frame_times(e: pd.Series, s: pd.Series, cfg: dict, fps: float) -> list[float]:
    """Seconds from the start of the episode for each frame of this sample."""
    dur = e["length"] / fps
    n = cfg["frames_per_clip"]
    if s["task"] == "skill":
        c, total = s["clip"], cfg["clips_per_episode"]
        lo, hi = dur * c / total, dur * (c + 1) / total
        return [lo + (i + 0.5) * (hi - lo) / n for i in range(n)]
    lo = dur * (1 - cfg["success_window"])
    last = dur - 1.0 / fps  # the final frame
    return [lo + (i + 1) * (last - lo) / n for i in range(n)]


def grid(frames: list[Image.Image], size: tuple[int, int]) -> Image.Image:
    cols = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)
    w, h = size
    canvas = Image.new("RGB", (cols * w, rows * h))
    for i, f in enumerate(frames):
        canvas.paste(f.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return canvas


def decode_at(container, stream, times: list[float]) -> list[Image.Image]:
    out = []
    for t in times:
        container.seek(int(t / stream.time_base), stream=stream, backward=True, any_frame=False)
        frame = None
        for fr in container.decode(stream):
            frame = fr
            if fr.pts * stream.time_base >= t - 1e-3:
                break
        out.append(frame.to_image())
    return out


def main() -> None:
    import json
    cfg = config_from_argv()
    fps = json.loads((cfg["data_dir"] / "meta" / "info.json").read_text())["fps"]
    eps = load_episodes(cfg).set_index("episode_index")
    samples = pd.concat([pd.read_parquet(cfg["run_dir"] / "splits" / f"{s}.parquet") for s in SPLITS])
    samples = samples.drop_duplicates("sample_key")
    out = cfg["run_dir"] / "frames"
    out.mkdir(exist_ok=True)
    samples["vid_file"] = samples["episode_index"].map(eps["vid_file"])
    todo = samples[[not (out / f"{k}.jpg").exists() for k in samples["sample_key"]]]
    print(f"{len(todo)} of {len(samples)} sample images to extract")
    done = 0
    for vf, group in todo.groupby("vid_file"):
        with av.open(str(cfg["data_dir"] / vf)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for _, s in group.sort_values(["episode_index", "clip"]).iterrows():
                e = eps.loc[s["episode_index"]]
                times = [e["vid_from"] + t for t in frame_times(e, s, cfg, fps)]
                frames = decode_at(container, stream, times)
                if cfg.get("rotate_180"):  # LIBERO renders images upside down
                    frames = [f.rotate(180) for f in frames]
                grid(frames, tuple(cfg["frame_size"])).save(
                    out / f"{s['sample_key']}.jpg", quality=90)
                done += 1
        print(f"{vf}: {done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
