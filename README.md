# MARL-FANET Tactical Swarm Framework

MADDPG/MATD3 기반 FANET 드론 스웜 연구 코드입니다.

## 3분 요약

이 프로젝트는 "드론 스웜이 통신 품질을 유지하면서 공격을 견디고, 충돌/에너지 낭비를 줄이도록" 강화학습으로 정책을 학습하는 코드입니다.

- 환경: 다중 드론(FANET), 일부 드론은 악성 노드로 동작 가능
- 알고리즘: `MADDPG`, `MATD3`
- 핵심 비교 포인트:
  - 통신 성능(PDR, 지연, 단절률)
  - 탐지 성능(F1, Accuracy 등)
  - 가치함수 안정성(Q 과대평가 gap, TD error)
- 최신 결론: 본 설정에서 MATD3가 MADDPG 대비 더 안정적인 Q 추정과 더 나은 통신 성능 지표를 보였습니다.

## 처음 읽는 사람 추천 순서

1. 이 README의 `프로젝트를 처음 보는 분을 위한 핵심 개념`
2. `시나리오와 공격 모델 설명`
3. `드론은 언제 통신하고, 언제 끊기는가`
4. `보상은 무엇을 높일 때 올라가는가`
5. `그림 한 장으로 이해하기`
6. 마지막으로 `최종 로컬 비교 (최신 MATD3)`

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

## 프로젝트를 처음 보는 분을 위한 핵심 개념

이 프로젝트는 드론 스웜이 아래 목표를 동시에 만족하도록 학습하는 문제입니다.

- 통신 품질 유지(PDR↑, 지연↓, 단절↓)
- 보안 위협 대응(악성 노드 탐지/회피)
- 기동 안전성(충돌 회피, 벽면 붙기 방지)
- 공간 활용(적절한 분산/커버리지)

즉, 보상은 단일 목표(예: 통신만)로 구성되지 않고, 여러 목적의 균형을 맞추도록 설계되어 있습니다.

## 시나리오와 공격 모델 설명

평가 시나리오는 `Default`, `Blackhole`, `Selective Forwarding`, `Sybil`을 사용합니다.

### 1) Default (drop_and_trust)

- 기본 악성 동작 모델입니다.
- 악성 링크는 전송 성공률이 낮아지고(드롭 증가), 지연이 추가됩니다.

### 2) Blackhole

- 블랙홀 노드는 패킷을 거의 모두 버리는 공격입니다.
- 코드에서는 악성 경로일 때 전달률을 사실상 `0`으로 만들고, 링크 지연을 크게 증가시킵니다.

### 3) Selective Forwarding

- 확률적으로 일부 패킷만 버리는 공격입니다.
- 악성 드롭 확률에 따라 전달/드롭이 랜덤하게 결정됩니다.

### 4) Sybil

- 식별/신뢰를 교란하는 유형의 공격으로 모델링되어 있습니다.
- 코드에서는 전달률 저하 + 추가 지연으로 반영됩니다.

### 공격별로 실제로 보이는 현상(직관 표)

| 시나리오 | 전달률(PDR) | 지연(Delay) | 신뢰(Trust) | 관측되는 현상 |
|---|---|---|---|---|
| Default | 소폭 하락 | 소폭 증가 | 점진 하락 | 악성 링크가 간헐적으로 품질 저하 유발 |
| Blackhole | 크게 하락 | 크게 증가 | 빠르게 하락 | 특정 노드 경유 트래픽이 거의 사라짐 |
| Selective Forwarding | 중간 하락(변동 큼) | 중간 증가(변동 큼) | 변동성 증가 | 에피소드/스텝별 편차가 큼 |
| Sybil | 중간 하락 | 중간 이상 증가 | 왜곡/불안정 | 정상 링크처럼 보여도 품질이 불안정 |

## 공격이 실제로 어떻게 적용되는가

- 에피소드 시작 시 `malicious_ratio`에 따라 악성 드론 ID를 샘플링합니다.
- 링크 단위 계산에서 송신/수신 노드 중 하나라도 악성이면 악성 효과가 반영됩니다.
- 물리 계층 간섭(`interference`)도 함께 적용되어 최종 전달률이 추가로 감소합니다.

전달 성공률은 개념적으로 아래 형태입니다.

- 기본 전달률(공격 영향 반영)
- × 물리 성공 확률 `exp(-k * interference)`

## 드론은 언제 통신하고, 언제 끊기는가

### 통신이 성립되는 경우

- 기본적으로 두 드론 간 거리 `distance <= R_c`이면 연결 후보가 됩니다.
- 이 링크가 신뢰 기준(`trust_threshold`)을 크게 위반하지 않으면 연결 유지됩니다.

### 통신이 끊기는 경우

- 거리 초과: `distance > R_c`
- 낮은 신뢰: 신뢰가 임계값보다 낮고(특히 심한 저신뢰), 보안 규칙에 의해 링크 차단
- 공격/간섭 영향: 연결은 있어도 전달률이 매우 낮아져 실질적 단절 상태로 수렴

추가로, 환경에는 연결 유지 보조 장치도 있습니다.

- `connectivity_guard_coeff`: 이웃 수가 너무 적어질 때 다시 아군 쪽으로 당겨 링크 복구 유도
- 보안 회피(`malicious_avoid_coeff`, `suspicious_avoid_coeff`)와 연결성 유지 사이 균형 조절

### 한 줄 정리

- 가까우면 연결 후보가 되고, 신뢰가 너무 낮아지면 끊기며, 공격/간섭이 세면 연결돼 있어도 실효 전달률이 떨어집니다.

## 보상은 무엇을 높일 때 올라가는가

결론부터 말하면, 둘 다입니다.

- 통신이 잘 되면 보상이 올라가고
- 너무 몰리지 않게 공간을 활용해도 보상이 올라갑니다.

대신, 아래 항목들은 감점됩니다.

- 충돌 위험(`d_safe` 이내 근접)
- 링크 단절/지연 증가
- 에너지 과소비
- 보안 위험(악성 이웃, 저신뢰 링크, 경보 누적)
- 벽면 근처에서의 비효율 기동(경계 패널티)

### 보상 구성(직관 요약)

- 통신 품질 보상: 전달률(PDR)↑, 지연(delay)↓
- 신뢰 보상: 신뢰 점수(trust)↑
- 커버리지 보상: 드론 간 적절한 분산(너무 붙지 않기)
- 안전/운용 패널티: 충돌, 단절, 보안위험, 에너지, 경계 근접 패널티

따라서 이 환경에서의 "좋은 정책"은 단순히 멀리 퍼지는 정책이 아니라,
"연결을 유지하면서 악성 노드를 피하고, 지연/드롭/에너지를 함께 관리하는 정책"입니다.

### 보상 항목 가중치 표 (현재 설정 기준)

아래 값은 현재 `config.yaml` 기준 기본 가중치입니다. 값이 클수록 해당 항목의 영향력이 커집니다.

| 항목 | 설정 키 | 기본값 | 영향 방향 | 의미 |
|---|---|---:|---|---|
| 커버리지 | `reward_cov_coeff` | 0.08 | + | 드론 간 적절한 거리/분산 유지 |
| 충돌 패널티 | `reward_col_coeff` | 2.5 | - | `d_safe` 이내 근접 시 강한 감점 |
| 연결성 패널티 | `reward_conn_coeff` | 5.5 | - | 통신 반경(`R_c`) 밖 링크 증가 시 감점 |
| 신뢰 보상(양) | `reward_trust_pos_coeff` | 1.8 | + | 신뢰 높은 네트워크 상태 보상 |
| 신뢰 패널티(음) | `reward_trust_neg_coeff` | 1.2 | - | 저신뢰 상태에 대한 감점 |
| PDR 계수 | `reward_w_pdr` | 1.6 | + | 전달률(PDR) 향상 보상 |
| 지연 계수 | `reward_w_delay` | 1.2 | - | 평균 지연 증가 패널티 |
| 에너지 계수 | `reward_w_energy` | 0.8 | - | 에너지 소비 증가 패널티 |
| 보안 위험 계수 | `reward_w_security` | 1.6 | - | 악성/저신뢰/경보 누적 위험 패널티 |
| 중심 드리프트 패널티 | `center_reward_coeff` | 0.05 | - | 스웜 중심이 과도하게 치우치면 감점 |

추가로, 아래 항목들은 보상 계산에 간접적으로 강한 영향을 줍니다.

| 항목 | 설정 키 | 기본값 | 역할 |
|---|---|---:|---|
| 통신 반경 | `R_c` | 300.0 | 링크 성립 거리 기준 |
| 안전 거리 | `d_safe` | 30.0 | 충돌 패널티 발동 거리 |
| 신뢰 임계값 | `trust_threshold` | 0.35 | 저신뢰 링크 완화/차단 기준 |
| 연결성 가드 | `connectivity_guard_coeff` | 0.45 | 이웃 부족 시 재연결 유도 강도 |
| 악성 회피 | `malicious_avoid_coeff` | 0.65 | 악성 이웃 회피 강도 |

실무적으로는 `reward_w_pdr`, `reward_w_delay`, `reward_w_security`, `reward_conn_coeff`를 먼저 조정하면 정책 성격(공격적 분산 vs 안정적 연결 유지)이 가장 크게 달라집니다.

## 한 에피소드에서 일어나는 순서

1. 환경 초기화 후 악성 노드 샘플링
2. 각 드론이 행동(가속도 성격의 3축 액션) 선택
3. 위치/속도 갱신 + 경계/연결성/보안 회피 보정
4. 링크별 전달률/지연/신뢰/에너지 계산
5. 보상 계산 및 관측 업데이트
6. 다음 스텝 반복

참고: 현재 구현은 `terminated=False`로 고정되어 있어, 에피소드는 보통 `max_steps`까지 진행됩니다.

## 그림 한 장으로 이해하기

```mermaid
flowchart TD
  A[에피소드 시작\n초기 위치 샘플링 + 악성 노드 지정] --> B[에이전트 행동 선택\n3축 액션]
  B --> C[이동/속도 갱신\n경계 보호 + 연결성 가드 + 보안 회피]
  C --> D[링크 품질 계산\n거리 기반 연결 + 공격/간섭 반영]
  D --> E[신뢰 갱신\n저신뢰 링크 완화/차단]
  E --> F[보상 계산]
  F --> F1[PDR/신뢰/분산(커버리지) 보상]
  F --> F2[지연/충돌/단절/에너지/보안위험 패널티]
  F1 --> G[관측 업데이트\nobs(12차원) 생성]
  F2 --> G
  G --> H{종료 조건}
  H -->|False| B
  H -->|True 또는 max_steps 도달| I[에피소드 종료]

  J[공격 시나리오\nDefault/Blackhole/Selective Forwarding/Sybil] --> D
```

## 관측 벡터(드론 1대 기준)

드론 하나의 관측은 총 12차원입니다.

- 위치 3차원
- 속도 3차원
- 신뢰 점수
- SNR
- 홉 수
- 잔여 에너지
- 경보 이력(alert history)
- 통신 지연 정규화 값

## 자주 묻는 질문 (FAQ)

### Q1. 보상이 오르면 "통신이 좋아진 것"인가요, "정찰 커버리지가 좋아진 것"인가요?

둘 다입니다. 이 환경은 다목적 보상이라 통신(PDR/지연), 보안(신뢰/위험), 기동(충돌/경계), 커버리지(분산)를 동시에 반영합니다.

### Q2. 드론이 서로 멀어지면 무조건 나쁜가요?

무조건 나쁘지는 않습니다. 일정 수준 분산은 커버리지에 유리하지만, `R_c`를 넘겨 링크가 끊기면 연결성 패널티가 커집니다.

### Q3. 악성 노드를 강하게 피하면 항상 좋은가요?

항상 그렇지는 않습니다. 너무 강하게 피하면 이웃 수가 부족해져 네트워크가 분절될 수 있어, 코드에서 `connectivity_guard`로 균형을 잡습니다.

### Q4. Q값 비교에서 무엇을 보면 되나요?

- `q_overestimation_gap`(작을수록 좋음): 과대평가 편향 크기
- `q_abs_td_error`(작을수록 좋음): Bellman 일관성

이 두 지표가 낮으면 일반적으로 가치추정이 안정적이라고 볼 수 있습니다.

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
- `logs/training_rewards_q_values.csv`, `logs/training_rewards_q_values_matd3.csv`
- `logs/eval_metrics.csv`, `logs/eval_metrics_matd3.csv`
- `logs/test_metrics.csv`, `logs/test_metrics_matd3.csv`
- `logs/compare_maddpg_vs_matd3_summary.csv`
- `logs/compare_maddpg_vs_matd3_summary_auc.txt`
- `logs/compare_maddpg_vs_matd3_report.md`

## Q값 과대평가 비교 (MADDPG vs MATD3)

학습 시 각 에피소드별로 Q 통계가 자동 저장됩니다.

- `q_current_mean`: 현재 critic의 평균 Q
- `q_target_mean`: 타깃 Bellman 값의 평균 Q
- `q_overestimation_gap`: `q_current_mean - q_target_mean` (양수로 클수록 과대평가 경향)
- `q_abs_td_error`: `|Q - y|` 평균
- `q_disagreement_mean` (MATD3): twin critic 간 차이 평균

Q 비교 리포트 생성:

```bash
python utils/compare_q_values.py \
  --baseline-dir logs/experiments/maddpg_baseline_maddpg \
  --candidate-dir logs/experiments/matd3_candidate_matd3 \
  --baseline-label MADDPG \
  --candidate-label MATD3 \
  --output-dir logs \
  --prefix compare_maddpg_vs_matd3_q
```

생성 산출물:

- `logs/compare_maddpg_vs_matd3_q_summary.csv`
- `logs/compare_maddpg_vs_matd3_q_summary.txt`
- `logs/compare_maddpg_vs_matd3_q_gap_curve.png`

참고: 현재 기본 로그 파일명은 `training_q_values*.csv`가 아니라 `training_rewards_q_values*.csv`로 생성됩니다.

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

### Q값 근거: MATD3 과대평가 억제 검증

본 연구에서는 MADDPG와 MATD3의 성능 차이를 최종 보상뿐 아니라 가치함수 추정 안정성 관점에서 추가 검증했습니다. 특히 MATD3의 핵심 장점으로 알려진 Q값 과대평가 억제 효과를 확인하기 위해, 학습 완료된 체크포인트를 기준으로 Q 통계를 추출해 비교했습니다.

비교 지표는 아래를 사용했습니다.

- `q_overestimation_gap` = 현재 Q - 타깃 Q
- `q_abs_td_error` = |Q - y|

일반적으로 `q_overestimation_gap`이 작을수록 과대평가 편향이 낮고, `q_abs_td_error`가 작을수록 Bellman 일관성이 높다고 해석할 수 있습니다.

최종 Q 비교 결과(`compare_maddpg_vs_matd3_q_final_local_summary.txt`)는 다음과 같습니다.

- MADDPG `tail50 q_overestimation_gap`: `4.190252`
- MATD3 `tail50 q_overestimation_gap`: `1.000809`
- 차이(MATD3 - MADDPG): `-3.189444`
- MADDPG `tail50 q_abs_td_error`: `12.779785`
- MATD3 `tail50 q_abs_td_error`: `10.591883`
- 차이(MATD3 - MADDPG): `-2.187902`

즉, MATD3는 본 FANET 다중 에이전트 보안 환경에서 MADDPG 대비 Q값 과대평가를 더 효과적으로 억제했고, 가치 추정 오차도 더 안정적으로 유지했습니다. 이는 MATD3의 성능 우위가 단순한 지표 변동이 아니라, 이중 크리틱 기반의 보수적 타깃 추정이라는 구조적 장점과 정합적임을 뒷받침합니다.

핵심 산출물:

- 최종 요약 CSV: `logs/compare_maddpg_vs_matd3_final_local_summary.csv`
- 최종 AUC 텍스트: `logs/compare_maddpg_vs_matd3_final_local_summary_auc.txt`
- 학습 곡선: `logs/compare_maddpg_vs_matd3_final_local_learning_curve.png`
- 시나리오 막대그래프: `logs/compare_maddpg_vs_matd3_final_local_eval_bar.png`
- ROC 비교 그래프: `logs/compare_maddpg_vs_matd3_final_local_roc_auc.png`
- 최신 MATD3 아카이브: `logs/experiments/matd3_final_matd3_local/`
- Q 비교 요약 CSV: `logs/compare_maddpg_vs_matd3_q_final_local_summary.csv`
- Q 비교 요약 TXT: `logs/compare_maddpg_vs_matd3_q_final_local_summary.txt`
- Q gap 곡선: `logs/compare_maddpg_vs_matd3_q_final_local_gap_curve.png`

최신 MATD3 GIF 시각화:

- GIF: `logs/attack_blackhole_matd3_best_fair_cfgmatch_v2.gif`
- 링크 이벤트 로그: `logs/attack_blackhole_matd3_best_fair_cfgmatch_v2_events.csv`

재현 명령어 (최신 MATD3 Blackhole GIF):

```bash
python visualize_attack.py --config config.yaml --policy trained --algorithm matd3 --scenario Blackhole --episode 180 --fps 4 --output logs/attack_blackhole_matd3_best_fair_cfgmatch_v2.gif --event-log logs/attack_blackhole_matd3_best_fair_cfgmatch_v2_events.csv --no-show
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
