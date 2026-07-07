# MADDPG vs MATD3 Comparison Report

- Generated: 2026-07-06 16:20:47
- Summary source: logs/compare_maddpg_vs_matd3_fair10_avoid_summary.csv
- AUC source: logs/compare_maddpg_vs_matd3_fair10_avoid_summary_auc.txt

## Executive Summary

- Mean delta PDR (MATD3 - MADDPG): -0.014956
- Mean delta Delay ms (MATD3 - MADDPG): 8.261915
- Mean delta Detection F1 (MATD3 - MADDPG): -0.000560
- AUC MADDPG: 0.657352
- AUC MATD3: 0.631636
- Delta AUC: -0.025716

## Scenario Win Count

- PDR improved scenarios: 0/4
- Delay improved scenarios (lower is better): 0/4
- Detection F1 improved scenarios: 0/4

## Interpretation

- Candidate is stronger in network efficiency if PDR increases while Delay decreases.
- Candidate is safer only if Detection F1 also improves.
- Current result indicates a trade-off: throughput/latency gains vs detection performance drop.

## Scenario Table

| Scenario | Delta PDR | Delta Delay (ms) | Delta Detection F1 |
|---|---:|---:|---:|
| Blackhole | -0.018865 | 9.642170 | -0.000070 |
| Default | -0.012305 | 7.546960 | -0.000350 |
| Selective Forwarding | -0.014110 | 7.717895 | -0.001535 |
| Sybil | -0.014545 | 8.140635 | -0.000285 |

## Artifact Links

- Learning Curve: logs/compare_maddpg_vs_matd3_fair10_avoid_learning_curve.png
- Evaluation Bar: logs/compare_maddpg_vs_matd3_fair10_avoid_eval_bar.png
- ROC-AUC: logs/compare_maddpg_vs_matd3_fair10_avoid_roc_auc.png
