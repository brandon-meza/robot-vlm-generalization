"""Step 5: score the calibration and test parts of a split with one model variant.

Usage:
  python scripts/05_evaluate.py configs/X.json --split random_episode --mode zero_shot
  modes: zero_shot  base model, with images
         text_only  base model, NO image (shows what the prompt alone gives away)
         finetuned  base model + the LoRA adapter trained on this split
         shuffled   base model + the adapter trained on shuffled labels
  --quant nf4 loads the base weights in 4-bit (NF4) for the accuracy/latency comparison.

Output: runs/<name>/preds/<split>__<mode>__<quant>.parquet, one row per sample,
with the probability of every class and the measured latency (batch size 1).
"""
import argparse

import numpy as np
import pandas as pd
import torch

from common import load_config, load_episodes, load_split
from vlm_lib import SUCCESS_CLASSES, first_token_ids, load_model, messages, save_skill_classes, score, skill_classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", required=True)
    ap.add_argument("--mode", required=True, choices=["zero_shot", "text_only", "finetuned", "shuffled"])
    ap.add_argument("--quant", default="bf16", choices=["bf16", "nf4"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not (cfg["run_dir"] / "classes.json").exists():
        save_skill_classes(cfg, load_episodes(cfg)["skill"].value_counts().index.tolist())
    classes = skill_classes(cfg)

    adapter = None
    if args.mode in ("finetuned", "shuffled"):
        adapter = cfg["run_dir"] / "adapters" / f"{args.split}__{args.mode}"
    model, processor = load_model(cfg["model"], args.quant, adapter)
    ids = {"skill": first_token_ids(processor, classes), "success": first_token_ids(processor, SUCCESS_CLASSES)}
    names = {"skill": classes, "success": SUCCESS_CLASSES}

    data = load_split(cfg, args.split)
    data = data[data.part.isin(["calib", "test"])].reset_index(drop=True)
    rows = []
    # Latency: every calib and test sample is timed (no separate timing subset), first call included;
    # with thousands of samples one cold-start call does not move the median or p90.
    for i, r in data.iterrows():
        r = r.to_dict()
        probs, ms = score(model, processor, messages(r, classes, cfg, args.mode != "text_only"), ids[r["task"]])
        rows.append({"sample_key": r["sample_key"], "episode_index": r["episode_index"], "part": r["part"],
                     "task": r["task"], "label": r["label"], "pred": names[r["task"]][int(np.argmax(probs))],
                     "probs": probs, "latency_ms": ms})
        if i % 200 == 0:
            print(f"{args.split}/{args.mode}/{args.quant}: {i}/{len(data)}", flush=True)
    out = cfg["run_dir"] / "preds"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.attrs["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    df["peak_gib"] = df.attrs["peak_gib"]
    df.to_parquet(out / f"{args.split}__{args.mode}__{args.quant}.parquet")
    test = df[df.part == "test"]
    for t in ("skill", "success"):
        sub = test[test.task == t]
        print(t, "accuracy", round(float((sub.label == sub.pred).mean()), 4), "n", len(sub))


if __name__ == "__main__":
    main()
