**Smoke test only.** Operator/robot groups and success labels are synthetic; the numbers only prove the pipeline runs.

Skill accuracy (%) on the test part, 95% bootstrap interval over episodes.

| Split | Majority | Text only | Zero-shot | Fine-tuned | Shuffled-label FT |
|---|---|---|---|---|---|
| random_clip | 78.5 | 9.2 [1.5, 17.7] | 13.8 [6.2, 22.4] | 90.8 [82.3, 98.5] | not run |
| random_episode | 81.8 | 9.1 [0.0, 22.7] | 15.2 [4.5, 27.3] | 98.5 [95.5, 100.0] | 81.8 [63.6, 95.5] |
| held_out_operator | 78.9 | 15.8 [0.0, 36.8] | 21.1 [8.8, 38.6] | 84.2 [63.2, 100.0] | not run |
| held_out_robot | 70.8 | 16.7 [4.2, 31.4] | 18.1 [5.6, 31.9] | 83.3 [68.6, 95.8] | not run |

| Split gap, fine-tuned (points) | Value |
|---|---|
| random_clip minus random_episode | -7.7 [-17.1, -0.0] |
| random_clip minus held_out_operator | 6.6 [-10.6, 25.7] |
| random_clip minus held_out_robot | 7.4 [-7.6, 26.1] |

| Gain (points, paired) | Value |
|---|---|
| random_clip: finetuned minus zero_shot | 76.9 [61.8, 89.9] |
| random_clip: finetuned minus text_only | 81.5 [64.5, 96.9] |
| random_episode: finetuned minus zero_shot | 83.3 [71.2, 93.9] |
| random_episode: finetuned minus text_only | 89.4 [76.5, 100.0] |
| random_episode: finetuned minus shuffled | 16.7 [0.7, 36.4] |
| held_out_operator: finetuned minus zero_shot | 63.2 [28.1, 89.5] |
| held_out_operator: finetuned minus text_only | 68.4 [26.3, 100.0] |
| held_out_robot: finetuned minus zero_shot | 65.3 [38.2, 91.7] |
| held_out_robot: finetuned minus text_only | 66.7 [37.3, 91.7] |

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

| Conformal sets (alpha = 0.1), calibration part of the split | coverage | mean set size | n calib | skills in calib |
|---|---|---|---|---|
| held_out_operator | 0.895 | 1.07 | 81 | 3 |
| held_out_robot | 0.958 | 1.14 | 78 | 3 |
| random_clip | 0.954 | 1.15 | 33 | 3 |
| random_episode | 0.909 | 0.91 | 33 | 3 |

Coverage holds only when calibration and test samples are exchangeable. A calibration part with few skills (e.g. one robot that only did one skill) breaks this.

| Conformal sets, calibrated on half of the new groups' test sessions | coverage | mean set size | n calib | n test |
|---|---|---|---|---|
| held_out_operator | 0.867 | 1.07 | 27 | 30 |
| held_out_robot | 0.944 | 1.03 | 36 | 36 |

| Fine-tuned skill accuracy re-weighted to the random_episode skill mix | accuracy (%) | skills present | reference weight covered |
|---|---|---|---|
| random_clip | 90.9 | 3 | 1.00 |
| random_episode | 98.5 | 3 | 1.00 |
| held_out_operator | 90.9 | 3 | 1.00 |
| held_out_robot | 90.9 | 3 | 1.00 |

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
