# Which skill is the robot doing, and did it work? A leakage-safe VLM fine-tune on real robot video

This project fine-tunes a small vision-language model (Qwen3-VL-2B-Instruct, LoRA) on clips from
teleoperated household-robot episodes. It predicts:

- **(a) the basic skill** in a clip (the leading verb of the instruction: Grab, Place, Open, ...), from images only;
- **(b) episode success**, from the last part of the episode plus the instruction.

The main question is about evaluation, not model size: **how much does a naive split overstate accuracy?**
The same model is trained and tested on a ladder of four splits:

| Split | What is held out | What it measures |
|---|---|---|
| `random_clip` | random clips | Clips of one episode fall on both sides. Near-duplicate frames leak. |
| `random_episode` | whole episodes (grouped by `uuid`) | No frame leakage. |
| `held_out_operator` | every episode of some operators | Shift to new people driving the robot. |
| `held_out_robot` | every episode of some robots | Shift to new robot units. |

A drop from `random_clip` to `random_episode` would be leakage between clips of one episode. A drop from
`random_episode` to the held-out splits is the cost of meeting a new operator or a new robot. Reporting them
separately keeps the headline honest.

Controls, so a number can be trusted:

- **majority class**: always predict the most common training skill;
- **text only**: the same model and prompt with no image (what the prompt alone gives away);
- **zero-shot**: the base model with images, before fine-tuning;
- **shuffled-label fine-tune**: same training, labels permuted. It can learn label frequencies, nothing more.

Also reported: per-skill confusion matrices, split-conformal prediction sets (coverage and set size),
bf16 vs 4-bit (NF4) latency and accuracy, and a fixed sample of errors exported for a hand check.

## Status

- Target dataset: [airoa-org/airoa-moma](https://huggingface.co/datasets/airoa-org/airoa-moma)
  (Toyota HSR teleoperation episodes; CC BY-NC-SA 4.0; gated, automatic approval). **Run complete**; results below.
- Smoke test: the full pipeline ran end to end on a small slice of the public
  [lerobot/libero](https://huggingface.co/datasets/lerobot/libero) dataset. LIBERO has no operator, robot or
  failure labels, so for this test **groups and success labels are synthetic** (deterministic hashes).
  The smoke numbers below prove that the code runs; they say nothing about model quality. The test parts are tiny,
  so intervals are wide, and success scores on random labels can look good or bad by chance alone.

## Findings (airoa-moma)
All numbers are in the generated table below.
- **Fine-tuning works.** The fine-tuned model is far above zero-shot, the text-only control, the majority
  class and the shuffled-label control on every split. The shuffled-label model only reaches the majority
  rate, so the gain comes from the images, not from label frequencies.
- **Clip leakage is small here.** The `random_clip` and `random_episode` accuracies overlap.
- **New operators and new robots cost accuracy.** Both held-out splits are clearly lower than
  `random_episode`. The held-out splits also test a different mix of skills, so the table re-weights
  per-skill accuracy to the `random_episode` mix: the drop remains.
- **Conformal sets need calibration data from the new setting.** On `held_out_robot`, the calibration
  part happened to be one robot that did only one skill, and coverage collapsed far below the target.
  Calibrating on half of the new robots' own episodes restores coverage near the target, with larger sets.
  For a new robot, label a few of its episodes before trusting the sets.
- **4-bit (NF4)** cuts peak GPU memory by more than half for a small loss of accuracy, with slightly
  higher latency on this GPU.
- **Success detection is inconclusive.** Failures are rare (a few percent of test episodes, so only a
  handful of failures per test set). PR-AUC is above the base rate on some splits, but with so few
  failures the estimates are unstable and the text-only control is sometimes better. No claim is made about it.

<!-- BEGIN RESULTS:libero_smoke -->
**Smoke test only.** Operator/robot groups and success labels are synthetic; the numbers only prove the pipeline runs.

Skill accuracy (%) on the test part, 95% bootstrap interval over episodes.

| Split | Majority | Text only | Zero-shot | Fine-tuned | Shuffled-label FT |
|---|---|---|---|---|---|
| random_clip | 78.5 | 9.2 [1.6, 17.7] | 13.8 [7.4, 22.5] | 90.8 [82.3, 98.4] | not run |
| random_episode | 81.8 | 9.1 [0.0, 22.7] | 15.2 [4.5, 27.3] | 98.5 [95.5, 100.0] | 81.8 [63.6, 95.5] |
| held_out_operator | 78.9 | 15.8 [0.0, 36.8] | 21.1 [7.9, 39.5] | 84.2 [63.2, 100.0] | not run |
| held_out_robot | 70.8 | 16.7 [4.2, 33.3] | 18.1 [6.9, 31.9] | 83.3 [66.7, 95.8] | not run |

| Split gap, fine-tuned (points) | Value |
|---|---|
| random_clip minus random_episode | -7.7 [-16.1, 0.7] |
| random_clip minus held_out_operator | 6.6 [-8.5, 26.4] |
| random_clip minus held_out_robot | 7.4 [-9.5, 24.7] |

| Gain (points, paired) | Value |
|---|---|
| random_clip: finetuned minus zero_shot | 76.9 [62.1, 89.4] |
| random_clip: finetuned minus text_only | 81.5 [64.6, 96.8] |
| random_episode: finetuned minus zero_shot | 83.3 [71.2, 93.9] |
| random_episode: finetuned minus text_only | 89.4 [76.5, 100.0] |
| random_episode: finetuned minus shuffled | 16.7 [0.7, 36.4] |
| held_out_operator: finetuned minus zero_shot | 63.2 [26.3, 91.2] |
| held_out_operator: finetuned minus text_only | 68.4 [26.3, 100.0] |
| held_out_robot: finetuned minus zero_shot | 65.3 [36.8, 88.9] |
| held_out_robot: finetuned minus text_only | 66.7 [33.3, 91.7] |

| Success detection | n | failure rate | PR-AUC (failure) | ROC-AUC |
|---|---|---|---|---|
| held_out_operator/finetuned/bf16 | 19 | 0.105 | 1.000 | 1.000 |
| held_out_operator/text_only/bf16 | 19 | 0.105 | 0.182 | 0.529 |
| held_out_operator/zero_shot/bf16 | 19 | 0.105 | 0.146 | 0.515 |
| held_out_robot/finetuned/bf16 | 24 | 0.208 | 0.470 | 0.695 |
| held_out_robot/text_only/bf16 | 24 | 0.208 | 0.230 | 0.495 |
| held_out_robot/zero_shot/bf16 | 24 | 0.208 | 0.303 | 0.621 |
| random_clip/finetuned/bf16 | 22 | 0.136 | 0.147 | 0.509 |
| random_clip/text_only/bf16 | 22 | 0.136 | 0.188 | 0.553 |
| random_clip/zero_shot/bf16 | 22 | 0.136 | 0.207 | 0.614 |
| random_episode/finetuned/bf16 | 22 | 0.136 | 0.167 | 0.605 |
| random_episode/finetuned/nf4 | 22 | 0.136 | 0.120 | 0.360 |
| random_episode/shuffled/bf16 | 22 | 0.136 | 0.178 | 0.632 |
| random_episode/text_only/bf16 | 22 | 0.136 | 0.154 | 0.456 |
| random_episode/zero_shot/bf16 | 22 | 0.136 | 0.251 | 0.719 |

| Conformal sets (alpha = 0.1) | coverage | mean set size | n calib |
|---|---|---|---|
| held_out_operator | 0.895 | 1.07 | 81 |
| held_out_robot | 0.972 | 1.15 | 78 |
| random_clip | 0.969 | 1.28 | 33 |
| random_episode | 0.909 | 0.91 | 33 |

| Model variant | median latency (ms) | p90 (ms) | peak GPU memory (GiB) | skill accuracy |
|---|---|---|---|---|
| held_out_operator/finetuned/bf16 | 73.5 | 75.9 | 4.12 | 84.2 |
| held_out_operator/text_only/bf16 | 13.4 | 14.1 | 3.98 | 15.8 |
| held_out_operator/zero_shot/bf16 | 63.7 | 66.3 | 4.04 | 21.1 |
| held_out_robot/finetuned/bf16 | 74.5 | 75.8 | 4.12 | 83.3 |
| held_out_robot/text_only/bf16 | 12.9 | 13.4 | 3.98 | 16.7 |
| held_out_robot/zero_shot/bf16 | 63.4 | 66.5 | 4.04 | 18.1 |
| random_clip/finetuned/bf16 | 72.6 | 74.0 | 4.12 | 90.8 |
| random_clip/text_only/bf16 | 12.7 | 13.5 | 3.98 | 9.2 |
| random_clip/zero_shot/bf16 | 61.9 | 63.0 | 4.04 | 13.8 |
| random_episode/finetuned/bf16 | 73.9 | 77.0 | 4.12 | 98.5 |
| random_episode/finetuned/nf4 | 83.6 | 87.8 | 1.71 | 95.5 |
| random_episode/shuffled/bf16 | 72.6 | 75.5 | 4.12 | 81.8 |
| random_episode/text_only/bf16 | 13.3 | 13.9 | 3.98 | 9.1 |
| random_episode/zero_shot/bf16 | 62.1 | 64.6 | 4.04 | 15.2 |
<!-- END RESULTS:libero_smoke -->

<!-- BEGIN RESULTS:airoa_moma -->
Skill accuracy (%) on the test part, 95% bootstrap interval over episodes.

| Split | Majority | Text only | Zero-shot | Fine-tuned | Shuffled-label FT |
|---|---|---|---|---|---|
| random_clip | 20.9 | 1.9 [1.2, 2.7] | 17.4 [15.5, 19.5] | 78.3 [76.2, 80.5] | not run |
| random_episode | 18.6 | 1.1 [0.4, 2.0] | 15.3 [13.1, 17.6] | 76.6 [74.2, 79.2] | 18.6 [15.7, 21.5] |
| held_out_operator | 0.9 | 1.2 [0.5, 2.0] | 20.7 [18.4, 23.1] | 61.3 [58.5, 64.0] | not run |
| held_out_robot | 14.7 | 5.0 [3.6, 6.6] | 12.8 [10.6, 14.9] | 54.0 [50.8, 56.9] | not run |

| Split gap, fine-tuned (points) | Value |
|---|---|
| random_clip minus random_episode | 1.7 [-1.7, 5.0] |
| random_clip minus held_out_operator | 17.0 [13.3, 20.4] |
| random_clip minus held_out_robot | 24.4 [20.6, 28.3] |

| Gain (points, paired) | Value |
|---|---|
| random_clip: finetuned minus zero_shot | 60.9 [57.8, 63.8] |
| random_clip: finetuned minus text_only | 76.4 [74.0, 78.9] |
| random_episode: finetuned minus zero_shot | 61.3 [58.3, 64.6] |
| random_episode: finetuned minus text_only | 75.5 [72.9, 78.3] |
| random_episode: finetuned minus shuffled | 58.0 [54.5, 61.8] |
| held_out_operator: finetuned minus zero_shot | 40.7 [37.3, 43.9] |
| held_out_operator: finetuned minus text_only | 60.1 [57.2, 63.1] |
| held_out_robot: finetuned minus zero_shot | 41.2 [37.5, 44.8] |
| held_out_robot: finetuned minus text_only | 48.9 [45.2, 52.6] |

| Success detection | n | failure rate | PR-AUC (failure) | ROC-AUC |
|---|---|---|---|---|
| held_out_operator/finetuned/bf16 | 497 | 0.008 | 0.018 | 0.682 |
| held_out_operator/text_only/bf16 | 497 | 0.008 | 0.016 | 0.674 |
| held_out_operator/zero_shot/bf16 | 497 | 0.008 | 0.009 | 0.465 |
| held_out_robot/finetuned/bf16 | 495 | 0.028 | 0.050 | 0.647 |
| held_out_robot/text_only/bf16 | 495 | 0.028 | 0.056 | 0.567 |
| held_out_robot/zero_shot/bf16 | 495 | 0.028 | 0.042 | 0.498 |
| random_clip/finetuned/bf16 | 524 | 0.034 | 0.062 | 0.693 |
| random_clip/text_only/bf16 | 524 | 0.034 | 0.130 | 0.701 |
| random_clip/zero_shot/bf16 | 524 | 0.034 | 0.041 | 0.571 |
| random_episode/finetuned/bf16 | 491 | 0.033 | 0.152 | 0.766 |
| random_episode/finetuned/nf4 | 491 | 0.033 | 0.146 | 0.747 |
| random_episode/shuffled/bf16 | 491 | 0.033 | 0.048 | 0.530 |
| random_episode/text_only/bf16 | 491 | 0.033 | 0.113 | 0.661 |
| random_episode/zero_shot/bf16 | 491 | 0.033 | 0.033 | 0.533 |

| Conformal sets (alpha = 0.1), calibration part of the split | coverage | mean set size | n calib | skills in calib |
|---|---|---|---|---|
| held_out_operator | 0.892 | 2.27 | 1498 | 15 |
| held_out_robot | 0.162 | 0.18 | 1095 | 1 |
| random_clip | 0.901 | 1.35 | 1436 | 17 |
| random_episode | 0.886 | 1.36 | 1425 | 17 |

Coverage holds only when calibration and test samples are exchangeable. A calibration part with few skills (e.g. one robot that only did one skill) breaks this.

| Conformal sets, calibrated on half of the new groups' test episodes | coverage | mean set size | n calib | n test |
|---|---|---|---|---|
| held_out_operator | 0.893 | 2.34 | 744 | 759 |
| held_out_robot | 0.912 | 4.39 | 754 | 751 |

| Fine-tuned skill accuracy re-weighted to the random_episode skill mix | accuracy (%) | skills present | reference weight covered |
|---|---|---|---|
| random_clip | 78.1 | 17 | 1.00 |
| random_episode | 76.6 | 17 | 1.00 |
| held_out_operator | 64.8 | 15 | 0.97 |
| held_out_robot | 53.7 | 17 | 1.00 |

| Model variant | median latency (ms) | p90 (ms) | peak GPU memory (GiB) | skill accuracy |
|---|---|---|---|---|
| held_out_operator/finetuned/bf16 | 90.8 | 92.2 | 4.14 | 61.3 |
| held_out_operator/text_only/bf16 | 17.9 | 19.1 | 3.98 | 1.2 |
| held_out_operator/zero_shot/bf16 | 76.7 | 77.9 | 4.05 | 20.7 |
| held_out_robot/finetuned/bf16 | 90.9 | 92.2 | 4.14 | 54.0 |
| held_out_robot/text_only/bf16 | 17.9 | 19.0 | 3.98 | 5.0 |
| held_out_robot/zero_shot/bf16 | 76.6 | 77.8 | 4.05 | 12.8 |
| random_clip/finetuned/bf16 | 91.0 | 92.3 | 4.14 | 78.3 |
| random_clip/text_only/bf16 | 17.9 | 19.0 | 3.98 | 1.9 |
| random_clip/zero_shot/bf16 | 76.6 | 77.8 | 4.05 | 17.4 |
| random_episode/finetuned/bf16 | 91.1 | 92.4 | 4.14 | 76.6 |
| random_episode/finetuned/nf4 | 101.0 | 102.4 | 1.72 | 72.8 |
| random_episode/shuffled/bf16 | 90.8 | 92.1 | 4.14 | 18.6 |
| random_episode/text_only/bf16 | 17.9 | 19.0 | 3.98 | 1.1 |
| random_episode/zero_shot/bf16 | 76.7 | 77.9 | 4.05 | 15.3 |
<!-- END RESULTS:airoa_moma -->

## Reproduce

```bash
# Python 3.11, CUDA 12.8 build of PyTorch (sm_120 for RTX 50-series)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Smoke test on public data (one 16 GB GPU)
bash scripts/run_all.sh configs/libero_smoke.json

# Real run (after accepting the dataset terms on Hugging Face and logging in with your own token)
bash scripts/run_all.sh configs/airoa_moma.json
```

| Script | What it does |
|---|---|
| `00_download.py` | episode metadata, then whole video files chosen to cover many operators and robots |
| `01_audit_metadata.py` | class balance, group concentration, synonym verbs, label mismatches, length outliers |
| `02_make_splits.py` | the four deterministic splits, with leakage asserts |
| `03_extract_frames.py` | AV1 decoding, one 2x2 frame grid per clip |
| `04_finetune_lora.py` | LoRA fine-tune (optionally with shuffled labels) |
| `05_evaluate.py` | zero-shot / text-only / fine-tuned / shuffled; bf16 or 4-bit |
| `06_metrics.py` | all numbers, bootstrap intervals, conformal sets, figures, error sheet |

## Design choices

- **Label = leading verb of the instruction.** Each episode is one primitive action, and its instruction
  names it ("Grab the towel ..."). The instruction is therefore **never** in the skill prompt; otherwise the
  text-only baseline would read the answer. Near-synonym verbs (Grab/Pick/Take, Place/Put) are kept apart by
  default and counted in the audit as a label-noise risk; `merge_synonyms` in the config merges them.
- **Clips, not single frames.** Four frames from one time window are tiled into one 2x2 image. One image keeps
  the vision-token count fixed and small. Each episode gives three skill clips (one per third of the episode)
  and one success clip (the last 30%).
- **Answers are scored, not parsed.** The model's next-token logits for the first token of each class name
  give a probability per class. No free-text parsing, and the probabilities feed the conformal sets.
- **Group by `uuid`.** Episodes of one recording session share a `uuid`; grouping by it stops leakage
  between parts of the same session.
- **Deterministic splits.** A SHA-1 hash of (split name, group) decides the side. Very large groups
  (over `max_group_share_in_test` of the data, e.g. one operator with about half the episodes) are never
  put in the test set. The split lists are written to `runs/` and **not committed**: for licensed data,
  episode-ID lists are derived data.
- **Calibration part.** Conformal sets need data not used for training. It is drawn by the same grouping rule
  as the test part. With few groups this can go wrong (see Findings), so a second variant calibrates on half
  of the test groups' own episodes.
- **Bootstrap over episodes.** Samples from one episode are correlated, so intervals resample whole episodes.
- **LoRA on the language model only**, rank 16, vision tower frozen, bf16, gradient checkpointing.

## Limitations

- Classification of skill and success, not action prediction. It is a monitor for a policy, not a policy.
- airoa-moma has one location, so a held-out-location split is impossible.
- Failures are rare in airoa-moma (the audit reports the rate), so success detection is reported as PR-AUC next to the failure base rate.
- Only a subset of the video files is used (set by `max_video_files`) to fit the time budget.
- Conformal coverage is only guaranteed when calibration and test data are exchangeable; on the held-out splits it is an empirical check.

## Licences

Code: MIT (see `LICENSE`). Data: airoa-moma is CC BY-NC-SA 4.0. This repository contains code and aggregate
metrics only: no data, no frames, no episode lists and no fine-tuned weights. Model: Qwen3-VL-2B-Instruct (Apache-2.0).
