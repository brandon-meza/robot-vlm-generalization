"""Step 0: download episode metadata, then a subset of the packed video files.

LeRobot v3 datasets pack many episodes into each ~0.5 GB mp4 file, so single
episodes cannot be fetched by name. We read the episode metadata first, then
pick whole video files greedily so that the subset covers as many operators
and robots as possible, up to `max_video_files`.

Gated datasets (such as airoa-org/airoa-moma) need the dataset owner's terms
to be accepted on the Hugging Face website and a token in the environment
(HF_TOKEN) or from `hf auth login`. This script never creates or stores a token.
"""
import json

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from common import build_episode_table, config_from_argv, episodes_path


def main() -> None:
    cfg = config_from_argv()
    repo, dd = cfg["repo_id"], cfg["data_dir"]
    snapshot_download(repo, repo_type="dataset", local_dir=dd, allow_patterns=["meta/*", "meta/**", "README.md"])
    files = [s.rfilename for s in HfApi().dataset_info(repo).siblings]
    raw = pd.concat([pd.read_parquet(f) for f in sorted((dd / "meta" / "episodes").rglob("*.parquet"))])
    if "tasks" not in raw.columns:
        # Task text lives only in the frame table: fetch the (small) data parquet files.
        snapshot_download(repo, repo_type="dataset", local_dir=dd, allow_patterns=["data/**"])

    cam = cfg["camera_key"]
    raw["vid_file"] = [f"videos/{cam}/chunk-{int(c):03d}/file-{int(f):03d}.mp4"
                       for c, f in zip(raw[f"videos/{cam}/chunk_index"], raw[f"videos/{cam}/file_index"])]
    gf = cfg["group_fields"]
    cover_cols = [gf[g] for g in ("operator", "robot") if not cfg["synthetic_groups"] and gf[g] in raw.columns]
    chosen, seen = [], set()
    candidates = sorted(raw["vid_file"].unique())
    while len(chosen) < min(cfg["max_video_files"], len(candidates)):
        def gain(f):
            sub = raw[raw.vid_file == f]
            new = {(c, v) for c in cover_cols for v in sub[c].astype(str).unique()} - seen
            return (len(new), -candidates.index(f))
        best = max((f for f in candidates if f not in chosen), key=gain)
        chosen.append(best)
        sub = raw[raw.vid_file == best]
        seen |= {(c, v) for c in cover_cols for v in sub[c].astype(str).unique()}
    missing = [f for f in chosen if f not in files]
    assert not missing, f"video files not in repo: {missing[:3]}"
    for f in chosen:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=dd)
        print("downloaded", f)

    table = build_episode_table(cfg)
    table["video_downloaded"] = table["vid_file"].isin(chosen)
    table.to_parquet(episodes_path(cfg))
    summary = {"video_files": len(chosen), "episodes_total": int(len(table)),
               "episodes_with_video": int(table.video_downloaded.sum())}
    (cfg["run_dir"] / "download_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
