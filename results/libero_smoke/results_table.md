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
