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
