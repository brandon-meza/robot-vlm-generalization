"""Step 2: make four deterministic train / calibration / test splits (a "split ladder").

  random_clip        clips assigned at random. Clips of ONE episode land in both
                     train and test, so near-duplicate frames leak. This is the
                     naive split many papers use; it is the optimistic upper end.
  random_episode     whole episodes (grouped by uuid) go to one side only.
  held_out_operator  every episode of a test operator is unseen in training.
  held_out_robot     every episode of a test robot is unseen in training.

Reading the ladder top to bottom separates two effects: within-episode leakage
(random_clip -> random_episode) and a real shift to new people or new robots
(random_episode -> held_out_*).

Samples. Each episode gives `clips_per_episode` skill clips (one per equal
time window, `frames_per_clip` frames each) and one success clip (the last
`success_window` fraction of the episode).

Determinism. Assignment uses a SHA-1 hash of (split name, group id), so the
same metadata always gives the same split, on any machine. The split LISTS are
written to runs/ (not committed: for licensed data, ID lists are derived data).
Only this script and the aggregate summary are shared.

The calibration part (for conformal prediction) is carved out of the non-test
groups in the same way as the test part, so it sees the same kind of shift.
"""
import json

import pandas as pd

from common import config_from_argv, load_episodes, stable_hash

SPLITS = {"random_clip": None, "random_episode": "g_episode",
          "held_out_operator": "g_operator", "held_out_robot": "g_robot"}


def make_samples(t: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for _, e in t.iterrows():
        for c in range(cfg["clips_per_episode"]):
            rows.append({"sample_key": f"{e.episode_index}_s{c}", "episode_index": e.episode_index,
                         "task": "skill", "clip": c})
        rows.append({"sample_key": f"{e.episode_index}_f", "episode_index": e.episode_index,
                     "task": "success", "clip": -1})
    s = pd.DataFrame(rows)
    return s.merge(t[["episode_index", "g_episode", "g_operator", "g_robot", "skill", "success"]],
                   on="episode_index", how="left", validate="many_to_one", suffixes=("", "_episode"))


def pick_groups(sizes: pd.Series, salt: str, target: float, max_share: float, exclude: set) -> set:
    """Walk groups in hash order; take them until `target` share of samples is reached."""
    total = sizes.sum()
    order = sorted(sizes.index, key=lambda g: stable_hash(f"{salt}:{g}"))
    chosen, share = set(), 0.0
    for g in order:
        if share >= target:
            break
        # Skip groups that are too big alone, or would overshoot the target by > 50%.
        if g in exclude or sizes[g] / total > max_share or share + sizes[g] / total > 1.5 * target:
            continue
        chosen.add(g)
        share += sizes[g] / total
    if not chosen:  # few large groups: take the first allowed one even if it overshoots
        allowed = [g for g in order if g not in exclude and sizes[g] / total <= max_share]
        if not allowed:
            raise ValueError(f"{salt}: every group is larger than max_group_share_in_test={max_share} "
                             "(or already used); lower the share or use another grouping key")
        chosen = {allowed[0]}
    return chosen


def assign(samples: pd.DataFrame, split: str, key: str | None, cfg: dict) -> pd.Series:
    if key is None:  # random clip: each sample on its own
        u = samples["sample_key"].map(lambda k: (stable_hash(f"{split}:{k}") % 10_000) / 10_000)
        tf, cf = cfg["test_fraction"], cfg["calib_fraction"]
        return pd.Series(["test" if x < tf else "calib" if x < tf + cf else "train" for x in u],
                         index=samples.index)
    sizes = samples.groupby(key).size()
    test = pick_groups(sizes, f"{split}:test", cfg["test_fraction"], cfg["max_group_share_in_test"], set())
    calib = pick_groups(sizes, f"{split}:calib", cfg["calib_fraction"], cfg["max_group_share_in_test"], test)
    return samples[key].map(lambda g: "test" if g in test else "calib" if g in calib else "train")


def check_leakage(df: pd.DataFrame, split: str, key: str | None) -> dict:
    tr, te, ca = (df[df.part == p] for p in ("train", "test", "calib"))
    assert df["sample_key"].is_unique, "a sample appears twice"
    report = {"n_train": int(len(tr)), "n_calib": int(len(ca)), "n_test": int(len(te))}
    assert len(tr) and len(te), f"{split}: empty train or test"
    shared_ep = set(tr.g_episode) & set(te.g_episode)
    report["test_samples_whose_episode_is_in_train"] = float(te.g_episode.isin(shared_ep).mean())
    if key is not None:
        # The defining guarantee of a grouped split: no shared group, and never a shared uuid.
        assert not (set(tr[key]) & set(te[key])), f"{split}: group leak train/test"
        assert not (set(tr[key]) & set(ca[key])), f"{split}: group leak train/calib"
        assert not shared_ep, f"{split}: uuid leak"
        # The calibration part feeds the conformal sets, so it must be as separate as the test part.
        assert not (set(tr.g_episode) & set(ca.g_episode)), f"{split}: uuid leak train/calib"
        assert not (set(ca.g_episode) & set(te.g_episode)), f"{split}: uuid leak calib/test"
        report["test_groups"] = int(te[key].nunique())
    unseen = ~te["skill"].isin(set(tr["skill"]))
    report["test_skill_samples_with_class_unseen_in_train"] = int((unseen & (te.task == "skill")).sum())
    return report


def main() -> None:
    cfg = config_from_argv()
    t = load_episodes(cfg)
    t = t[t["video_downloaded"]].reset_index(drop=True)
    samples = make_samples(t, cfg)
    out = cfg["run_dir"] / "splits"
    out.mkdir(exist_ok=True)
    summary = {}
    for split, key in SPLITS.items():
        parts = assign(samples, split, key, cfg)
        assert parts.equals(assign(samples, split, key, cfg)), "split is not deterministic"
        df = samples.assign(part=parts)
        summary[split] = check_leakage(df, split, key)
        # Cap sizes (deterministically) so every run fits the time budget.
        caps = {"train": cfg["max_train_samples"], "calib": cfg["max_test_samples"], "test": cfg["max_test_samples"]}
        df["_h"] = df["sample_key"].map(lambda k: stable_hash(f"cap:{k}"))
        df = df.sort_values("_h")
        df = pd.concat([df[df.part == p].head(caps[p]) for p in ("train", "calib", "test")]).drop(columns="_h")
        summary[split]["after_cap"] = df["part"].value_counts().to_dict()
        df.to_parquet(out / f"{split}.parquet")
    summary["note"] = "group leakage asserts passed for every grouped split"
    (cfg["result_dir"] / "splits_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
