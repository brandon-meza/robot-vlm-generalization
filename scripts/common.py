"""Shared helpers: config, paths, and one tidy episode table for any LeRobot v3 dataset.

Every later step works from the same table, `episodes.parquet`, with columns:
  episode_index, length, instruction, skill, success,
  g_operator, g_robot, g_episode   (the grouping keys used for splits)
  vid_file, vid_from, vid_to       (where the episode sits inside the packed mp4)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg["data_dir"] = ROOT / "data" / cfg["name"]   # raw downloads (never committed)
    cfg["run_dir"] = ROOT / "runs" / cfg["name"]    # frames, splits, predictions, adapters (never committed)
    cfg["result_dir"] = ROOT / "results" / cfg["name"]  # aggregate metrics only
    for k in ("data_dir", "run_dir", "result_dir"):
        cfg[k].mkdir(parents=True, exist_ok=True)
    return cfg


def config_from_argv() -> dict:
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {sys.argv[0]} configs/<dataset>.json [options]")
    return load_config(sys.argv[1])


def stable_hash(text: str) -> int:
    """Deterministic across machines and Python runs (unlike hash())."""
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16)


def first_verb(instruction: str) -> str:
    """'Grab the towel in the basket.' -> 'Grab'. The basic skill is the leading verb."""
    word = instruction.strip().split()[0] if instruction.strip() else "Unknown"
    return word.strip(".,").capitalize()


# Near-synonyms noted on the AIRoA MoMa card (Grab/Pick/Take, Place/Put).
SYNONYMS = {"Pick": "Grab", "Take": "Grab", "Put": "Place"}


def read_raw_episodes(cfg: dict) -> pd.DataFrame:
    meta = cfg["data_dir"] / "meta"
    files = sorted((meta / "episodes").rglob("*.parquet"))
    if not files:
        sys.exit(f"no episode metadata under {meta}; run 00_download.py first")
    eps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    eps = eps[[c for c in eps.columns if not c.startswith("stats/")]]
    if "tasks" not in eps.columns:
        # Some v3 datasets keep the task only per frame: read task_index from the data parquet.
        tasks = pd.read_parquet(meta / "tasks.parquet")
        names = dict(zip(tasks["task_index"], tasks.index))
        frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"])
                            for f in sorted((cfg["data_dir"] / "data").rglob("*.parquet"))])
        first = frames.groupby("episode_index")["task_index"].first()
        eps["tasks"] = eps["episode_index"].map(lambda e: [names[first[e]]] if e in first else None)
    return eps


def build_episode_table(cfg: dict) -> pd.DataFrame:
    raw = read_raw_episodes(cfg)
    cam = cfg["camera_key"]
    t = pd.DataFrame({
        "episode_index": raw["episode_index"].astype(int),
        "length": raw["length"].astype(int),
        "instruction": raw["tasks"].map(lambda x: str(x[0]) if x is not None and len(x) else ""),
        "vid_file": [f"videos/{cam}/chunk-{int(c):03d}/file-{int(f):03d}.mp4"
                     for c, f in zip(raw[f"videos/{cam}/chunk_index"], raw[f"videos/{cam}/file_index"])],
        "vid_from": raw[f"videos/{cam}/from_timestamp"].astype(float),
        "vid_to": raw[f"videos/{cam}/to_timestamp"].astype(float),
    })
    t["skill"] = t["instruction"].map(first_verb)
    if cfg.get("merge_synonyms"):
        t["skill"] = t["skill"].replace(SYNONYMS)
    if cfg["synthetic_groups"]:
        # SMOKE TEST ONLY: the public test dataset has no operator/robot fields,
        # so deterministic fake groups are made to exercise the split code.
        t["g_operator"] = [f"op{stable_hash(f'op:{e}') % 5}" for e in t.episode_index]
        t["g_robot"] = [f"rb{stable_hash(f'rb:{e}') % 4}" for e in t.episode_index]
        t["g_episode"] = t["episode_index"].astype(str)
    else:
        gf = cfg["group_fields"]
        for g in ("operator", "robot", "episode"):
            t[f"g_{g}"] = raw[gf[g]].astype(str).to_numpy()
    if cfg["success_field"]:
        t["success"] = raw[cfg["success_field"]].astype(bool).to_numpy()
    else:
        # SMOKE TEST ONLY: random but deterministic labels. A model should score at chance.
        rate = cfg["synthetic_success_rate"]
        t["success"] = [stable_hash(f"succ:{e}") % 1000 >= rate * 1000 for e in t.episode_index]
    return t


def episodes_path(cfg: dict) -> Path:
    return cfg["run_dir"] / "episodes.parquet"


def load_episodes(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(episodes_path(cfg))


def rng(cfg: dict) -> np.random.Generator:
    return np.random.default_rng(cfg["seed"])


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    """Samples of one split with instruction text and image path attached."""
    s = pd.read_parquet(cfg["run_dir"] / "splits" / f"{split}.parquet")
    eps = load_episodes(cfg)[["episode_index", "instruction"]]
    s = s.merge(eps, on="episode_index", how="left", validate="many_to_one", suffixes=("", "_episode"))
    s["image_path"] = [str(cfg["run_dir"] / "frames" / f"{k}.jpg") for k in s["sample_key"]]
    s["label"] = [sk if t == "skill" else ("Yes" if ok else "No")
                  for t, sk, ok in zip(s["task"], s["skill"], s["success"])]
    return s
