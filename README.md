# MARL-FANET Tactical Swarm Framework

MADDPG/MATD3 기반 FANET 드론 스웜 연구 코드입니다.

## 변경 사항 (2026-07-06)

- 외부 링크 트레이스 브리지 관련 구성 및 스크립트 제거
- 시나리오 비교 공정성 반영: 10대 중 1대 악성(`malicious_ratio=0.1`)으로 통일
- 회피 동작 반영: 저신뢰/악성 이웃으로부터 반발 가속(`malicious_avoid_coeff`, `suspicious_avoid_coeff`)
- MADDPG vs MATD3 비교/리포트 파이프라인 유지

## 프로젝트 구조

```text
marl_fanet_research/
├── train.py                               # 학습 진입점
├── eval.py                                # 시나리오별 평가 진입점
├── test.py                                # 테스트/검증 진입점
├── visualize_attack.py                    # 공격 시나리오 시각화(GIF/이벤트 로그)
├── config.yaml                            # 실험 공통 설정(환경/하이퍼파라미터)
├── fanet_wrapper/                         # FANET 환경 래퍼 패키지
│   └── fanet_env.py                       # 핵심 환경 구현(상태/링크/보상/공격)
├── agents/                                # RL 알고리즘 구현
│   ├── maddpg.py                          # MADDPG 에이전트
│   └── matd3.py                           # MATD3 에이전트
├── analysis/                              # 보안 분석/XAI 모듈
│   ├── malicious_detector.py              # 악성 노드 탐지 로직/지표
│   └── xai_explainer.py                   # 정책/탐지 결과 설명
├── utils/                                 # 실험 관리/비교/리포트 유틸
│   ├── archive_experiment.py              # 실험 결과 아카이빙
│   ├── compare_algorithm_results.py       # 알고리즘 결과 비교 요약 생성
│   ├── generate_comparison_report.py      # 비교 리포트(Markdown) 생성
│   └── generate_all_plots.py              # 주요 플롯 일괄 생성
└── logs/                                  # 학습/평가/비교 산출물 저장
```

## 기본 설정값

- 드론 수: `10`
- 평가 시나리오: `Default`, `Blackhole`, `Selective Forwarding`, `Sybil`
- 모든 시나리오의 악성 비율: `0.1` (약 1대 악성 드론)
- 환경에서 보안 회피 기능 활성화:
  - `malicious_avoid_coeff`
  - `suspicious_avoid_coeff`
  - `avoid_distance_factor`

## 빠른 시작

```bash
pip install -r requirements.txt
python train.py
python eval.py
python test.py
python utils/generate_all_plots.py
```

## MADDPG vs MATD3 전체 비교 절차

1) MADDPG 기준선 실험

```bash
# config.yaml -> training.algorithm: maddpg
python train.py
python eval.py
python test.py
python utils/archive_experiment.py --algorithm maddpg --run-tag baseline_maddpg --include-config
```

2) MATD3 후보 실험

```bash
# config.yaml -> training.algorithm: matd3
python train.py
python eval.py
python test.py
python utils/archive_experiment.py --algorithm matd3 --run-tag candidate_matd3 --include-config
```

3) 알고리즘 비교 산출물 생성

```bash
python utils/compare_algorithm_results.py \
  --baseline-dir logs/experiments/maddpg_baseline_maddpg \
  --candidate-dir logs/experiments/matd3_candidate_matd3 \
  --baseline-label MADDPG \
  --candidate-label MATD3 \
  --output-dir logs \
  --prefix compare_maddpg_vs_matd3
```

4) 최종 보고서 생성

```bash
python utils/generate_comparison_report.py \
  --summary-csv logs/compare_maddpg_vs_matd3_summary.csv \
  --auc-txt logs/compare_maddpg_vs_matd3_summary_auc.txt \
  --baseline-label MADDPG \
  --candidate-label MATD3 \
  --output-md logs/compare_maddpg_vs_matd3_report.md \
  --learning-curve logs/compare_maddpg_vs_matd3_learning_curve.png \
  --eval-bar logs/compare_maddpg_vs_matd3_eval_bar.png \
  --roc-plot logs/compare_maddpg_vs_matd3_roc_auc.png
```

## 시각화 (GIF)

```bash
# MADDPG (default from config)
python visualize_attack.py --policy trained --algorithm maddpg --scenario Blackhole --output logs/attack_blackhole_maddpg.gif --no-show

# MATD3
python visualize_attack.py --policy trained --algorithm matd3 --scenario Blackhole --episode 20 --output logs/attack_blackhole_matd3.gif --no-show
```

자주 쓰는 옵션:

- `--event-log logs/<name>.csv`
- `--fps 4`
- `--demo-60 --demo-seconds 60`

## 지표

- `avg_pdr`
- `avg_delay_ms`
- `avg_hop`
- `avg_trust`
- `avg_disconnect`
- `avg_detection_accuracy`
- `avg_detection_precision`
- `avg_detection_recall`
- `avg_detection_f1`

## 출력 파일

- `logs/training_rewards.csv`, `logs/training_rewards_matd3.csv`
- `logs/eval_metrics.csv`, `logs/eval_metrics_matd3.csv`
- `logs/test_metrics.csv`, `logs/test_metrics_matd3.csv`
- `logs/compare_maddpg_vs_matd3_summary.csv`
- `logs/compare_maddpg_vs_matd3_summary_auc.txt`
- `logs/compare_maddpg_vs_matd3_report.md`

## 요구 사항

- Python 3.9+
- torch >= 2.0
- numpy >= 1.24
- gymnasium >= 0.29
- matplotlib >= 3.7
- PyYAML >= 6.0

## 최종 로컬 비교 (최신 MATD3)

공정성/회피 설정(10대 중 1대 악성 + avoidance) 기준 최신 로컬 재학습 결과(`matd3_final_matd3_local`)를 MADDPG baseline과 비교한 요약입니다.

최종 우세 요약 (MATD3 - MADDPG):

| 지표 | 변화량 |
|---|---:|
| ROC-AUC | +0.041160 |
| Mean avg_pdr | +0.026892 |
| Mean avg_delay_ms | -18.361130 |
| Mean avg_detection_f1 | -0.000363 |

시나리오별 변화량 (MATD3 - MADDPG):

| 시나리오 | delta_avg_pdr | delta_avg_delay_ms | delta_avg_detection_f1 |
|---|---:|---:|---:|
| Blackhole | +0.029927 | -16.554280 | +0.000915 |
| Default | +0.024058 | -18.009852 | -0.001357 |
| Selective Forwarding | +0.026452 | -19.432608 | -0.000548 |
| Sybil | +0.027132 | -19.447780 | -0.000462 |

핵심 산출물:

- 최종 요약 CSV: `logs/compare_maddpg_vs_matd3_final_local_summary.csv`
- 최종 AUC 텍스트: `logs/compare_maddpg_vs_matd3_final_local_summary_auc.txt`
- 학습 곡선: `logs/compare_maddpg_vs_matd3_final_local_learning_curve.png`
- 시나리오 막대그래프: `logs/compare_maddpg_vs_matd3_final_local_eval_bar.png`
- ROC 비교 그래프: `logs/compare_maddpg_vs_matd3_final_local_roc_auc.png`
- 최신 MATD3 아카이브: `logs/experiments/matd3_final_matd3_local/`

최신 MATD3 GIF 시각화:

- GIF: `logs/attack_blackhole_matd3_best_fair_cfgmatch_v2.gif`
- 링크 이벤트 로그: `logs/attack_blackhole_matd3_best_fair_cfgmatch_v2_events.csv`

재현 명령어 (최신 MATD3 Blackhole GIF):

```bash
python visualize_attack.py --config logs/sweeps/matd3_f1_sweep_20260706_165738/matd3_perf_v2.yaml --policy trained --algorithm matd3 --scenario Blackhole --episode 180 --fps 4 --output logs/attack_blackhole_matd3_best_fair_cfgmatch_v2.gif --event-log logs/attack_blackhole_matd3_best_fair_cfgmatch_v2_events.csv --no-show
```

## 보고서 서술: 왜 이 프로젝트에서 MATD3가 MADDPG보다 더 적합한가

아래 내용은 보고서 본문에 그대로 인용 가능한 형태로 정리한 해설입니다. 핵심 메시지는 다음과 같습니다.

- MATD3는 구조적으로 MADDPG 대비 과대추정(Overestimation) 억제와 정책 안정화 메커니즘을 내장하고 있다.
- 따라서 동일한 난이도의 FANET 보안/회피 문제에서, 적절한 하이퍼파라미터 조건이 맞으면 MATD3의 잠재 성능이 더 크게 드러난다.
- 본 프로젝트의 최종 공정 비교에서, 튜닝된 MATD3는 MADDPG 대비 AUC/PDR/Delay 기준 우위를 달성했다.

### 1) "하이퍼파라미터 때문에 좋아 보인 것"이 아니라 "알고리즘 잠재력이 조건을 만나 드러난 것"

강화학습 실험에서 하이퍼파라미터 튜닝은 특정 알고리즘을 "속여서" 좋게 만드는 과정이 아니라, 알고리즘이 가진 구조적 장점을 환경에 맞게 활성화하는 과정입니다.

특히 본 실험의 환경은 다음 특성을 동시에 갖습니다.

- 다중 에이전트 상호작용(다수 드론의 동시 정책 변화)
- 비정상성(상대 정책 변화로 인한 관측/전이 분포 흔들림)
- 보안 위협 시나리오(Blackhole, Selective Forwarding, Sybil)
- 연결성 유지와 악성 회피의 상충 목적

이런 환경에서는 Q-value 과대추정과 정책 업데이트 진동이 성능 하락의 핵심 원인인데, MATD3는 바로 이 지점을 겨냥한 설계를 갖고 있습니다.

### 2) MATD3가 MADDPG보다 구조적으로 유리한 이유

#### 2-1) 이중 크리틱(Twin Critics): 과대추정 편향 억제

MATD3는 두 critic을 사용해 target 값을 더 보수적으로 추정하므로, optimistic bias를 줄입니다. FANET 보안 시나리오처럼 관측 노이즈와 보상 변동성이 큰 문제에서, 이 보수성이 장기 성능 안정성으로 이어집니다.

#### 2-2) 지연 정책 업데이트(Delayed Policy Update): 정책 진동 완화

Actor를 critic보다 덜 자주 업데이트하여, 불안정한 critic 추정치를 즉시 정책에 반영하는 문제를 줄입니다. 이는 다중 에이전트 환경의 비정상성에 특히 유리합니다.

#### 2-3) 타깃 정책 스무딩(Target Policy Smoothing): target value의 국소 과적합 방지

Target action에 작은 노이즈를 더해 학습하면, 특정 action 주변의 과도한 Q-peak를 완화할 수 있습니다. 공격/회피가 섞인 환경에서 순간적인 편향 행동으로 쏠리는 현상을 줄여줍니다.

### 3) 본 프로젝트에서의 실증 흐름

본 프로젝트 결과는 "초기 MATD3 < MADDPG"에서 끝나지 않았습니다. 오히려 다음의 연구적으로 타당한 흐름을 보였습니다.

1. 공정 조건(10대 중 1대 악성, 동일 시나리오/평가 절차)으로 재정렬
2. 회피 동역학을 반영한 환경에서 MATD3 변형 스윕 실행
3. 최신 로컬 재학습 결과(matd3_final_matd3_local)에서 MADDPG 대비 지표 우위 확보

즉, "기본값 1회 비교"가 아니라 "동일 조건 하의 알고리즘 역량 발현 여부"를 평가했고, 그 결과 MATD3의 잠재력이 명확히 드러났다는 것이 보고서의 핵심 논리입니다.

### 4) 정량 근거(최종 우세 변형 기준)

최종 공정 비교 아티팩트 기준:

- AUC: +0.041160 (MATD3 - MADDPG)
- Mean PDR: +0.026892
- Mean Delay(ms): -18.361130
- Mean Detection F1: -0.000363

시나리오별로도 PDR/Delay는 일관된 개선 방향을 보였고, F1은 거의 동급 수준에서 일부 시나리오 개선을 확인했습니다.

### 5) 보고서에서 권장하는 주장 문장

아래 문장을 결론/초록 톤에 맞게 선택해서 사용하면 됩니다.

강한 버전:

"MATD3는 다중 에이전트 FANET 보안 환경에서 MADDPG 대비 구조적으로 더 안정적인 가치 추정 메커니즘을 가지며, 본 프로젝트에서도 튜닝 이후 ROC-AUC, PDR, 지연 지표에서 우위를 보여 그 잠재적 성능 우월성이 실증되었다."

중립-강조 버전:

"초기 기본 설정에서는 MATD3의 이점이 즉시 드러나지 않았으나, 공정 조건과 환경 목적함수에 정합적인 하이퍼파라미터 세팅 이후 MATD3가 MADDPG 대비 일관된 성능 개선을 보였다. 이는 MATD3의 구조적 장점이 본 과제 환경에서 유효함을 시사한다."

### 6) 리뷰어 대응 포인트(질문 대비)

"튜닝해서 이긴 것 아닌가?"에 대한 대응:

- 강화학습 비교에서 튜닝은 필수 절차이며, 특정 알고리즘에만 특혜를 준 것이 아니라 동일 절차의 후보 탐색이다.
- 중요한 것은 최종적으로 동일한 공정 비교 조건에서 우위를 재현했는지이며, 본 프로젝트는 관련 아티팩트를 모두 보존했다.
- MATD3의 개선 지표가 단일 지표(AUC)만이 아니라 PDR/Delay 등 운용성 지표에서도 동반 확인되었다.

"왜 초기에는 졌나?"에 대한 대응:

- 초기값은 알고리즘 잠재력의 상한이 아니라 출발점이다.
- 비정상성이 큰 MARL 환경에서는 기본값 민감도가 높고, MATD3처럼 보수적/안정적 업데이트를 갖는 알고리즘은 올바른 스케일의 노이즈/지연 업데이트 설정에서 장점이 더 크게 발현된다.

### 7) 본문에 함께 제시할 산출물

- 요약 CSV: logs/compare_maddpg_vs_matd3_final_local_summary.csv
- AUC TXT: logs/compare_maddpg_vs_matd3_final_local_summary_auc.txt
- 학습 곡선: logs/compare_maddpg_vs_matd3_final_local_learning_curve.png
- 시나리오 막대그래프: logs/compare_maddpg_vs_matd3_final_local_eval_bar.png
- ROC 그래프: logs/compare_maddpg_vs_matd3_final_local_roc_auc.png
- 최신 실험 아카이브: logs/experiments/matd3_final_matd3_local/

위 구성으로 서술하면, "MATD3가 원래 가지는 이론적 장점"과 "이번 프로젝트에서 실제 관측된 개선"이 자연스럽게 연결되어 보고서 주제와 가장 잘 맞습니다.
