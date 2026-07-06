# MADDPG vs MATD3 Comparison Report

- Generated: 2026-07-06 14:24:05
- Summary source: logs/compare_maddpg_vs_matd3_summary.csv
- AUC source: logs/compare_maddpg_vs_matd3_summary_auc.txt

## Executive Summary

- Mean delta PDR (MATD3 - MADDPG): 0.022084
- Mean delta Delay ms (MATD3 - MADDPG): -31.983170
- Mean delta Detection F1 (MATD3 - MADDPG): -0.121836
- AUC MADDPG: 0.623673
- AUC MATD3: 0.632915
- Delta AUC: 0.009243

## Scenario Win Count

- PDR improved scenarios: 4/4
- Delay improved scenarios (lower is better): 4/4
- Detection F1 improved scenarios: 0/4

## Interpretation

- Candidate is stronger in network efficiency if PDR increases while Delay decreases.
- Candidate is safer only if Detection F1 also improves.
- Current result indicates a trade-off: throughput/latency gains vs detection performance drop.

## Scenario Table

| Scenario | Delta PDR | Delta Delay (ms) | Delta Detection F1 |
|---|---:|---:|---:|
| Blackhole | 0.030830 | -28.780375 | -0.081128 |
| Default | 0.012645 | -32.609207 | -0.156615 |
| Selective Forwarding | 0.012315 | -31.547870 | -0.078978 |
| Sybil | 0.032545 | -34.995230 | -0.170623 |

## Artifact Links

- Learning Curve: logs/compare_maddpg_vs_matd3_learning_curve.png
- Evaluation Bar: logs/compare_maddpg_vs_matd3_eval_bar.png
- ROC-AUC: logs/compare_maddpg_vs_matd3_roc_auc.png
