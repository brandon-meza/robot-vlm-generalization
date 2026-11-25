"""Step 1: audit the episode metadata before any modelling.

Reports only aggregate counts (no episode IDs), so the output can be shared
even when the data licence forbids redistributing derived data.

Checks:
  - class balance of the skill label (leading verb) and of success
  - how concentrated operators and robots are (a split must not put one
    giant group into the test set)
  - near-synonym verbs (Grab/Pick/Take, Place/Put): a label-noise risk
  - instruction text vs the primitive_action field (when the dataset has it)
  - episode length outliers and video-timestamp consistency (to - from vs length / fps)
  - how many episodes share one uuid (the unit we must group by)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import SYNONYMS, config_from_argv, load_episodes, read_raw_episodes, ROOT


def main() -> None:
    cfg = config_from_argv()
    t = load_episodes(cfg)
    raw = read_raw_episodes(cfg)
    fps = json.loads((cfg["data_dir"] / "meta" / "info.json").read_text())["fps"]
    dur = t["length"] / fps
    vid_dur = t["vid_to"] - t["vid_from"]
    audit = {
        "dataset": cfg["repo_id"],
        "synthetic_groups_and_success": bool(cfg["synthetic_groups"]),
        "n_episodes": int(len(t)),
        "n_episodes_with_downloaded_video": int(t["video_downloaded"].sum()),
        "fps": fps,
        "skill_counts": t["skill"].value_counts().to_dict(),
        "n_skills": int(t["skill"].nunique()),
        "majority_skill_share": float(t["skill"].value_counts(normalize=True).iloc[0]),
        "success_rate": float(t["success"].mean()),
        "n_failures": int((~t["success"]).sum()),
        "n_instructions": int(t["instruction"].nunique()),
        "missing_instruction": int((t["instruction"] == "").sum()),
        "synonym_verb_episodes": {v: int((t["skill"] == v).sum()) for v in SYNONYMS},
        "operators": {"n": int(t.g_operator.nunique()),
                      "largest_share": float(t.g_operator.value_counts(normalize=True).iloc[0])},
        "robots": {"n": int(t.g_robot.nunique()),
                   "largest_share": float(t.g_robot.value_counts(normalize=True).iloc[0])},
        "episodes_per_uuid": {"max": int(t.g_episode.value_counts().max()),
                              "n_uuids": int(t.g_episode.nunique())},
        "duration_s": {"median": float(dur.median()), "p1": float(dur.quantile(0.01)),
                       "p99": float(dur.quantile(0.99)), "under_1s": int((dur < 1).sum()),
                       "over_60s": int((dur > 60).sum())},
        "video_vs_length_mismatch_over_0.5s": int((np.abs(vid_dur - dur) > 0.5).sum()),
    }
    if "primitive_action" in raw.columns:
        def mismatch(row):
            pa = row["primitive_action"]
            pa = list(pa) if pa is not None else []
            return bool(len(row["tasks"])) and str(row["tasks"][0]) not in [str(p) for p in pa]
        audit["instruction_not_in_primitive_action"] = int(raw.apply(mismatch, axis=1).sum())
    if "location_name" in raw.columns:
        audit["n_locations"] = int(raw["location_name"].nunique())
    if not cfg["synthetic_groups"]:
        ct = t.groupby("g_operator")["skill"].nunique()
        audit["skills_per_operator"] = {"min": int(ct.min()), "max": int(ct.max())}
    (cfg["result_dir"] / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    counts = t["skill"].value_counts()
    ax.bar(counts.index, counts.values, color="#2c5d8a")
    ax.set_ylabel("episodes")
    ax.set_title(f"Skill label balance ({cfg['name']})")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / f"{cfg['name']}_skill_balance.png", dpi=150)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
