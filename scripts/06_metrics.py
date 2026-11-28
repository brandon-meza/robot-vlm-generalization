"""Step 6: compute every reported number from the saved predictions.

Inputs : runs/<name>/preds/*.parquet, runs/<name>/splits/*.parquet
Outputs: results/<name>/metrics.json, results/<name>/results_table.md,
         figures/<name>_*.png, runs/<name>/review/ (error sheet for a hand check)

Confidence intervals: bootstrap over RECORDING SESSIONS (g_episode: the uuid on
airoa-moma; one session holds up to dozens of episodes). All samples of a
session are drawn together, because they are not independent.
  - Gain of one model over another on the same split: paired bootstrap.
  - Gap between two splits (different test sets): each test set resampled
    independently, then the difference is taken.
Conformal prediction (split conformal, skill task): the non-conformity score
is 1 - p(true class) on the calibration part; the threshold is its k-th
smallest value, k = ceil((n+1)(1-alpha)) (no finite threshold if k > n, so
the set is every class); the prediction set on a test sample is every
class with 1 - p <= threshold. Coverage is guaranteed (>= 1 - alpha) only if
calibration and test samples are exchangeable.
"""
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from common import ROOT, load_config, load_split
from vlm_lib import skill_classes

SPLITS = ["random_clip", "random_episode", "held_out_operator", "held_out_robot"]
MODES = ["zero_shot", "text_only", "finetuned", "shuffled"]


def episode_bootstrap(groups, fn, n, seed):
    """fn(row_indices) -> value. Returns array of n bootstrap values."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows_of = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = fn(np.concatenate([rows_of[i] for i in pick]))
    return out


def conformal_threshold(scores, alpha):
    """k-th smallest calibration score, k = ceil((n + 1)(1 - alpha)); +inf (full set) when k > n."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if n == 0 or k > n else float(np.sort(scores)[k - 1])


def ci(vals, point):
    return {"mean": float(point), "lo": float(np.nanpercentile(vals, 2.5)), "hi": float(np.nanpercentile(vals, 97.5))}


def main() -> None:
    cfg = load_config(sys.argv[1])
    nb, seed = cfg["bootstrap_samples"], cfg["seed"]
    classes = skill_classes(cfg)
    pdir = cfg["run_dir"] / "preds"
    ep = pd.read_parquet(cfg["run_dir"] / "episodes.parquet")
    session_of = dict(zip(ep.episode_index, ep.g_episode))   # bootstrap and calibration unit
    preds = {}
    for f in sorted(pdir.glob("*.parquet")):
        split, mode, quant = f.stem.split("__")
        df = pd.read_parquet(f)
        df["session"] = df.episode_index.map(session_of)
        assert df.session.notna().all(), f"{f.name}: episode without a recording session"
        preds[(split, mode, quant)] = df
    m = {"dataset": cfg["repo_id"], "synthetic_labels_and_groups": bool(cfg["synthetic_groups"]),
         "skill": {}, "success": {}, "majority": {}, "gains": {}, "split_gap": {}, "conformal": {},
         "latency": {}}

    test_sets = {}
    for (split, mode, quant), df in preds.items():
        te = df[(df.part == "test")].reset_index(drop=True)
        sk = te[te.task == "skill"].reset_index(drop=True)
        su = te[te.task == "success"].reset_index(drop=True)
        key = f"{split}/{mode}/{quant}"
        correct = (sk.label == sk.pred).to_numpy().astype(float)
        acc_b = episode_bootstrap(sk.session, lambda ix: correct[ix].mean(), nb, seed)
        present = sorted(set(sk.label) | set(sk.pred))   # a skill absent from both is not scored as F1 = 0
        m["skill"][key] = {"n": int(len(sk)), "n_sessions": int(sk.session.nunique()),
                           "accuracy": ci(acc_b, correct.mean()),
                           "macro_f1": float(f1_score(sk.label, sk.pred, average="macro", labels=present, zero_division=0)),
                           "macro_f1_classes": len(present)}
        test_sets[key] = (sk, correct)
        if len(su) and su.label.nunique() == 2:
            fail = (su.label == "No").to_numpy().astype(int)
            p_fail = np.array([p[1] for p in su.probs])
            m["success"][key] = {"n": int(len(su)), "failure_rate": float(fail.mean()),
                                 "pr_auc_failure": float(average_precision_score(fail, p_fail)),
                                 "roc_auc": float(roc_auc_score(fail, p_fail)),
                                 "accuracy": float((su.label == su.pred).mean())}
        lat = df["latency_ms"].to_numpy()
        m["latency"][key] = {"median_ms": float(np.median(lat)), "p90_ms": float(np.percentile(lat, 90)),
                             "peak_gib": float(df["peak_gib"].iloc[0])}

    # Majority-class baseline (most common training skill) per split
    for split in SPLITS:
        if not (cfg["run_dir"] / "splits" / f"{split}.parquet").exists():
            continue
        s = load_split(cfg, split)
        tr, te = s[(s.part == "train") & (s.task == "skill")], s[(s.part == "test") & (s.task == "skill")]
        maj = tr.skill.value_counts().index[0]
        m["majority"][split] = {"class": maj, "accuracy": float((te.skill == maj).mean()), "n": int(len(te))}

    # Gains on the same split (paired by sample)
    for split in SPLITS:
        ft = f"{split}/finetuned/bf16"
        if ft not in test_sets:
            continue
        sk_ft, c_ft = test_sets[ft]
        for other in ["zero_shot", "text_only", "shuffled"]:
            ok = f"{split}/{other}/bf16"
            if ok not in test_sets:
                continue
            sk_o, c_o = test_sets[ok]
            assert (sk_o.sample_key.to_numpy() == sk_ft.sample_key.to_numpy()).all(), "unpaired samples"
            d = c_ft - c_o
            b = episode_bootstrap(sk_ft.session, lambda ix: d[ix].mean(), nb, seed)
            m["gains"][f"{split}: finetuned minus {other}"] = ci(b, d.mean())

    # Split gap: random_clip vs each harder split (independent bootstraps)
    base = "random_clip/finetuned/bf16"
    if base in test_sets:
        sk0, c0 = test_sets[base]
        b0 = episode_bootstrap(sk0.session, lambda ix: c0[ix].mean(), nb, seed)
        for split in SPLITS[1:]:
            k = f"{split}/finetuned/bf16"
            if k in test_sets:
                sk1, c1 = test_sets[k]
                b1 = episode_bootstrap(sk1.session, lambda ix: c1[ix].mean(), nb, seed + 1)
                m["split_gap"][f"random_clip minus {split}"] = ci(b0 - b1, c0.mean() - c1.mean())

    # Task-mix check: the held-out splits test a different mix of skills, so part of
    # an accuracy drop could come from the mix rather than from new operators/robots.
    # Re-weight each split's per-skill accuracy to the random_episode test mix.
    m["mix_adjusted"] = {}
    ref = "random_episode/finetuned/bf16"
    if ref in test_sets:
        w = test_sets[ref][0].label.value_counts(normalize=True)
        for split in SPLITS:
            k = f"{split}/finetuned/bf16"
            if k not in test_sets:
                continue
            sk, c = test_sets[k]
            per = pd.Series(c, index=sk.index).groupby(sk.label).mean()
            common = [x for x in w.index if x in per.index]
            m["mix_adjusted"][split] = {
                "accuracy": float((per[common] * w[common]).sum() / w[common].sum()),
                "skills_present": len(common), "reference_weight_covered": float(w[common].sum())}

    # Split conformal prediction sets for the skill task
    alpha = cfg["conformal_alpha"]
    m["conformal_new_group"] = {}
    for (split, mode, quant), df in preds.items():
        if mode != "finetuned" or quant != "bf16" or not split.startswith("held_out"):
            continue
        # Variant: calibrate on half of the TEST recording sessions (same new operators/robots),
        # evaluate on the other half. This is the realistic recipe for a new robot:
        # label a few of its sessions first, then calibrate. Whole sessions go to one half,
        # so near-duplicate episodes never calibrate their own siblings.
        te_all = df[(df.task == "skill") & (df.part == "test")]
        sess = np.array(sorted(te_all.session.unique()))
        if len(sess) < 2:
            m["conformal_new_group"][split] = {"skipped": f"{len(sess)} test session(s); need at least 2"}
            continue
        np.random.default_rng(seed).shuffle(sess)
        cal_sess = set(sess[: len(sess) // 2])
        cal = te_all[te_all.session.isin(cal_sess)]
        te = te_all[~te_all.session.isin(cal_sess)]
        s_cal = np.array([1 - p[classes.index(l)] for p, l in zip(cal.probs, cal.label)])
        n = len(s_cal)
        qhat = conformal_threshold(s_cal, alpha)
        sets = [[c for c, p in zip(classes, ps) if 1 - p <= qhat] for ps in te.probs]
        cover = np.array([l in st for l, st in zip(te.label, sets)])
        m["conformal_new_group"][split] = {"alpha": alpha, "n_calib": int(n), "qhat": float(qhat),
                                           "test_coverage": float(cover.mean()),
                                           "mean_set_size": float(np.mean([len(s) for s in sets])),
                                           "n_test": int(len(te))}
    for (split, mode, quant), df in preds.items():
        if mode != "finetuned" or quant != "bf16":
            continue
        sk = df[df.task == "skill"]
        cal, te = sk[sk.part == "calib"], sk[sk.part == "test"]
        if len(cal) == 0:
            continue
        s_cal = np.array([1 - p[classes.index(l)] for p, l in zip(cal.probs, cal.label)])
        n = len(s_cal)
        qhat = conformal_threshold(s_cal, alpha)
        sets = [[c for c, p in zip(classes, ps) if 1 - p <= qhat] for ps in te.probs]
        cover = np.array([l in st for l, st in zip(te.label, sets)])
        m["conformal"][split] = {"calib_skills": int(cal.label.nunique()),
                                 "alpha": alpha, "n_calib": int(n), "qhat": float(qhat),
                                 "test_coverage": float(cover.mean()),
                                 "mean_set_size": float(np.mean([len(s) for s in sets])),
                                 "n_test": int(len(te))}

    rd = cfg["result_dir"]
    (rd / "metrics.json").write_text(json.dumps(m, indent=2))
    write_table(m, cfg, rd)
    figures(cfg, preds, classes)
    review_sheet(cfg, preds)
    update_readme(cfg, (rd / "results_table.md").read_text())
    print((rd / "results_table.md").read_text())


def fmt(c):
    return f"{100 * c['mean']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"


def write_table(m, cfg, rd):
    L = []
    if m["synthetic_labels_and_groups"]:
        L += ["**Smoke test only.** Operator/robot groups and success labels are synthetic; "
              "the numbers only prove the pipeline runs.", ""]
    unit = "episodes" if m["synthetic_labels_and_groups"] else "recording sessions (uuid)"
    L += [f"Skill accuracy (%) on the test part, 95% bootstrap interval over {unit}.", "",
          "| Split | Majority | Text only | Zero-shot | Fine-tuned | Shuffled-label FT |", "|---|---|---|---|---|---|"]
    for split in SPLITS:
        if split not in m["majority"]:
            continue
        row = [split, f"{100 * m['majority'][split]['accuracy']:.1f}"]
        for mode in ["text_only", "zero_shot", "finetuned", "shuffled"]:
            k = f"{split}/{mode}/bf16"
            row.append(fmt(m["skill"][k]["accuracy"]) if k in m["skill"] else "not run")
        L.append("| " + " | ".join(row) + " |")
    if m["split_gap"]:
        L += ["", "| Split gap, fine-tuned (points) | Value |", "|---|---|"]
        L += [f"| {k} | {fmt(v)} |" for k, v in m["split_gap"].items()]
    if m["gains"]:
        L += ["", "| Gain (points, paired) | Value |", "|---|---|"]
        L += [f"| {k} | {fmt(v)} |" for k, v in m["gains"].items()]
    if m["success"]:
        L += ["", "| Success detection | n | failure rate | PR-AUC (failure) | ROC-AUC |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['failure_rate']:.3f} | {v['pr_auc_failure']:.3f} | {v['roc_auc']:.3f} |"
              for k, v in m["success"].items()]
    if m["conformal"]:
        L += ["", f"| Conformal sets (alpha = {cfg['conformal_alpha']}), calibration part of the split | coverage | mean set size | n calib | skills in calib |",
              "|---|---|---|---|---|"]
        L += [f"| {k} | {v['test_coverage']:.3f} | {v['mean_set_size']:.2f} | {v['n_calib']} | {v['calib_skills']} |"
              for k, v in m["conformal"].items()]
        L += ["", "Coverage holds only when calibration and test samples are exchangeable. A calibration "
              "part with few skills (e.g. one robot that only did one skill) breaks this."]
    if m.get("conformal_new_group"):
        L += ["", f"| Conformal sets, calibrated on half of the new groups' test sessions | coverage | mean set size | n calib | n test |",
              "|---|---|---|---|---|"]
        L += [f"| {k} | {v['test_coverage']:.3f} | {v['mean_set_size']:.2f} | {v['n_calib']} | {v['n_test']} |"
              if "skipped" not in v else f"| {k} | not computed: {v['skipped']} | | | |"
              for k, v in m["conformal_new_group"].items()]
    if m.get("mix_adjusted"):
        L += ["", "| Fine-tuned skill accuracy re-weighted to the random_episode skill mix | accuracy (%) | skills present | reference weight covered |",
              "|---|---|---|---|"]
        L += [f"| {k} | {100 * v['accuracy']:.1f} | {v['skills_present']} | {v['reference_weight_covered']:.2f} |"
              for k, v in m["mix_adjusted"].items()]
    L += ["", "| Model variant | median latency (ms) | p90 (ms) | peak GPU memory (GiB) | skill accuracy |",
          "|---|---|---|---|---|"]
    for k, v in m["latency"].items():
        acc = m["skill"][k]["accuracy"]["mean"]
        L.append(f"| {k} | {v['median_ms']:.1f} | {v['p90_ms']:.1f} | {v['peak_gib']:.2f} | {100 * acc:.1f} |")
    (rd / "results_table.md").write_text("\n".join(L) + "\n")


def figures(cfg, preds, classes):
    fts = [(s, preds[(s, "finetuned", "bf16")]) for s in SPLITS if (s, "finetuned", "bf16") in preds]
    if not fts:
        return
    fig, axes = plt.subplots(1, len(fts), figsize=(4.2 * len(fts), 4), squeeze=False)
    for ax, (split, df) in zip(axes[0], fts):
        sk = df[(df.part == "test") & (df.task == "skill")]
        cm = confusion_matrix(sk.label, sk.pred, labels=classes).astype(float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)  # rows with no test samples stay empty
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(split, fontsize=9)
        ax.set_xlabel("predicted")
    axes[0][0].set_ylabel("true")
    fig.suptitle(f"Skill confusion (row-normalised), fine-tuned, {cfg['name']}")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{cfg['name']}_confusion.png", dpi=150)


def review_sheet(cfg, preds):
    """Export a fixed sample of fine-tuned errors for a person to check by hand."""
    key = next(((s, "finetuned", "bf16") for s in SPLITS[1:] + SPLITS[:1] if (s, "finetuned", "bf16") in preds), None)
    if key is None:
        return
    df = preds[key]
    err = df[(df.part == "test") & (df.task == "skill") & (df.label != df.pred)]
    err = err.sample(min(24, len(err)), random_state=cfg["seed"]) if len(err) else err
    out = cfg["run_dir"] / "review"
    out.mkdir(exist_ok=True)
    rows = []
    for _, r in err.iterrows():
        src = cfg["run_dir"] / "frames" / f"{r.sample_key}.jpg"
        shutil.copy(src, out / src.name)
        rows.append({"image": src.name, "split": key[0], "label": r.label, "pred": r.pred,
                     "p_pred": float(max(r.probs)), "hand_verdict (label wrong / model wrong / ambiguous)": ""})
    pd.DataFrame(rows).to_csv(out / "review_sheet.csv", index=False)


def update_readme(cfg, table: str) -> None:
    """Paste the generated table into README.md between this dataset's markers, if present."""
    import re
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text()
    pat = re.compile(rf"(<!-- BEGIN RESULTS:{cfg['name']} -->\n).*?(<!-- END RESULTS:{cfg['name']} -->)", re.S)
    if pat.search(text):
        path.write_text(pat.sub(lambda m: m.group(1) + table.rstrip() + "\n" + m.group(2), text))


if __name__ == "__main__":
    main()
