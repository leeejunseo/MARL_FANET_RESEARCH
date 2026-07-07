# MADDPG vs MATD3 Comparison Report

- Generated: 2026-07-06 22:13:55
- Summary source: logs/compare_maddpg_vs_matd3_final_local_summary.csv
- AUC source: logs/compare_maddpg_vs_matd3_final_local_summary_auc.txt

## Executive Summary

- Mean delta PDR (MATD3 - MADDPG): 0.026892
- Mean delta Delay ms (MATD3 - MADDPG): -18.361130
- Mean delta Detection F1 (MATD3 - MADDPG): -0.000363
- AUC MADDPG: 0.657352
- AUC MATD3: 0.698513
- Delta AUC: 0.041160

## Scenario Win Count

- PDR improved scenarios: 4/4
- Delay improved scenarios (lower is better): 4/4
- Detection F1 improved scenarios: 1/4

## Interpretation

- Candidate is stronger in network efficiency if PDR increases while Delay decreases.
- Candidate is safer only if Detection F1 also improves.
- Current result indicates a trade-off: throughput/latency gains vs detection performance drop.

## Scenario Table

| Scenario | Delta PDR | Delta Delay (ms) | Delta Detection F1 |
|---|---:|---:|---:|
| Blackhole | 0.029927 | -16.554280 | 0.000915 |
| Default | 0.024058 | -18.009852 | -0.001357 |
| Selective Forwarding | 0.026452 | -19.432608 | -0.000548 |
| Sybil | 0.027132 | -19.447780 | -0.000462 |

## Artifact Links

- Learning Curve: logs/compare_maddpg_vs_matd3_final_local_learning_curve.png
- Evaluation Bar: logs/compare_maddpg_vs_matd3_final_local_eval_bar.png
- ROC-AUC: logs/compare_maddpg_vs_matd3_final_local_roc_auc.png
