# MARL-FANET 전술 스웜 프레임워크

다중 에이전트 강화학습(MADDPG) 기반 FANET(Flying Ad-hoc Network) 드론 스웜 연구 프로젝트입니다.

이 저장소는 다음 목표를 중심으로 설계되었습니다.
- 스웜 기동 중 통신 연결 유지
- 악의적 노드(blackhole, selective forwarding, sybil) 대응
- 탐지 성능 및 XAI 기반 해석 가능성 분석
- 2D 통신/공격 애니메이션 시각화
- 군집 중심 드리프트 억제(속도 감쇠 + 중앙 유지 항)

현재 메인 실험 목표:
- `MADDPG`와 `MATD3`를 동일 환경/지표로 비교
- 학습 수렴 곡선, 시나리오별 성능, ROC-AUC를 동일 포맷으로 저장

## 현재 기본 설정

- 드론 수: 10
- 악성 비율: 0.1 (기본 시나리오)
- 기본 악성 드론 수: 약 1대
- 기본 시나리오: Default, Blackhole, Selective Forwarding, Sybil
- 기본 평가 모델 에피소드: 1000

주의:
- 기존 3드론 체크포인트는 10드론 설정과 바로 호환되지 않으므로 재학습이 필요합니다.

## 프로젝트 구조

```text
marl_fanet_research/
├── train.py
├── test.py
├── eval.py
├── visualize_attack.py
├── config.yaml
├── agents/
│   └── maddpg.py
├── ns3_wrapper/
│   └── fanet_env.py
├── analysis/
│   ├── malicious_detector.py
│   └── xai_explainer.py
├── utils/
│   ├── replay_buffer.py
│   ├── metrics_logger.py
│   ├── plot_learning_curve.py
│   ├── plot_roc_auc.py
│   ├── plot_bar_comparison.py
│   ├── plot_detection_metrics.py
│   ├── plot_xai_heatmap.py
│   └── generate_all_plots.py
└── logs/
```

## 빠른 실행

```bash
pip install -r requirements.txt

# 1) 학습
python train.py

# 2) 학습 모델 평가 (지표 CSV 저장)
python test.py

# 2-1) multi-seed 평균 + bridge OFF/ON 동시 비교
python test.py --compare-bridge --seeds 42,43,44,45,46

# 3) 시나리오별 정량 평가 + ROC 데이터 저장
python eval.py

# 4) 논문용 그래프 일괄 생성
python utils/generate_all_plots.py
```

## 메인 워크플로우: MADDPG vs MATD3 비교

아래 순서대로 실행하면 두 알고리즘 비교 산출물이 자동으로 정리됩니다.

1) MADDPG 기준선 생성 및 보관

```bash
# config.yaml: training.algorithm: maddpg
python train.py
python eval.py
python test.py
python utils/archive_experiment.py --algorithm maddpg --run-tag baseline_maddpg --include-config
```

2) MATD3 후보 생성 및 보관

```bash
# config.yaml: training.algorithm: matd3
python train.py
python eval.py
python test.py
python utils/archive_experiment.py --algorithm matd3 --run-tag candidate_matd3 --include-config
```

3) 자동 비교 그래프/요약 생성

```bash
python utils/compare_algorithm_results.py \
	--baseline-dir logs/experiments/maddpg_baseline_maddpg \
	--candidate-dir logs/experiments/matd3_candidate_matd3 \
	--baseline-label MADDPG \
	--candidate-label MATD3 \
	--output-dir logs \
	--prefix compare_maddpg_vs_matd3
```

생성 결과:
- `logs/compare_maddpg_vs_matd3_learning_curve.png`
- `logs/compare_maddpg_vs_matd3_eval_bar.png`
- `logs/compare_maddpg_vs_matd3_roc_auc.png`
- `logs/compare_maddpg_vs_matd3_summary.csv`
- `logs/compare_maddpg_vs_matd3_summary_auc.txt`

4) 비교 리포트 자동 생성 (해석 문장 포함)

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

생성 결과:
- `logs/compare_maddpg_vs_matd3_report.md`

### 현재 비교 결과 요약 (최근 실행)

- 시나리오 평균 기준: MATD3는 PDR 증가(+), 지연 감소(-), 탐지 F1 감소(-) 경향
- AUC: MADDPG 0.623673, MATD3 0.632915 (delta +0.009243)
- 시나리오별 delta (`MATD3 - MADDPG`)
	- Blackhole: PDR +0.030830, Delay -28.780375ms, Detection F1 -0.081128
	- Default: PDR +0.012645, Delay -32.609207ms, Detection F1 -0.156615
	- Selective Forwarding: PDR +0.012315, Delay -31.547870ms, Detection F1 -0.078978
	- Sybil: PDR +0.032545, Delay -34.995230ms, Detection F1 -0.170623

## MATD3 탐지 F1 보완 스윕

짧은 MATD3 하이퍼파라미터 스윕(탐지 보상/threshold/target noise)을 자동 실행합니다.

```bash
python utils/run_matd3_f1_sweep.py \
	--baseline-dir logs/experiments/maddpg_baseline_maddpg \
	--episodes 180 \
	--eval-episodes 12
```

산출물:
- `logs/sweeps/matd3_f1_sweep_YYYYMMDD_HHMMSS/sweep_results.csv`
- 각 variant의 archive: `logs/experiments/matd3_candidate_<variant>/`
- 각 variant의 비교 요약: `logs/compare_maddpg_vs_<variant>_summary.csv`

## 평가 CSV 스키마 안정화

- `evaluation.overwrite_csv: true` (기본 권장)
- 목적: `eval_metrics.csv`에 이전 포맷 행이 누적되어 컬럼 시프트가 발생하는 문제 방지
- 스크립트 측 보정: `utils/compare_algorithm_results.py`는 `policy` 컬럼 유무가 혼재된 CSV도 자동 정렬

## 2D 통신 공격 시각화

Matplotlib 애니메이션으로 다음을 시각화합니다.
- 링크 연결/단절
- 공격 경유 링크
- 고립 노드
- PDR/지연/단절 비율
- 공격 감지 confidence 시간축

```bash
# 학습 정책
python visualize_attack.py --policy trained --scenario Default

# 공격 시나리오 + GIF 저장
python visualize_attack.py --policy trained --scenario Blackhole --output logs/attack_blackhole.gif

# 무작위 정책 비교
python visualize_attack.py --policy random --scenario Default

# 링크 이벤트 CSV 저장
python visualize_attack.py --policy trained --scenario Blackhole --event-log logs/link_events_blackhole.csv

# 60초 자동 재생 데모 (시나리오 자동 순환)
python visualize_attack.py --policy trained --demo-60 --demo-seconds 60 --fps 3 --output logs/demo_60s.gif --event-log logs/demo_60s_link_events.csv --no-show
```

표현 규칙:
- 파란 노드: 정상
- 빨간 노드: 악의적 노드
- 노란 노드: 고립 노드
- 초록 링크: 정상 링크
- 주황 링크: 공격 경유 링크
- 빨간 점선: 직전 프레임 대비 단절 링크

링크 이벤트 CSV 컬럼:
- step
- scenario
- node_i
- node_j
- event_type (disconnect/reconnect)
- reason (distance_exceeded_rc, recovered_within_rc, malicious_path 포함)

## 핵심 지표

- avg_pdr
- avg_delay_ms
- avg_hop
- avg_trust
- avg_disconnect
- avg_detection_accuracy
- avg_detection_precision
- avg_detection_recall
- avg_detection_f1

평가 결과 파일:
- logs/test_metrics.csv
- logs/eval_metrics.csv
- logs/eval_node_features.npz
- logs/eval_node_features_{scenario}.npz

## 시나리오

config.yaml의 evaluation.scenarios에서 설정합니다.
- Default
- Blackhole
- Selective Forwarding
- Sybil

## 드리프트 억제 파라미터

config.yaml의 environment에서 설정합니다.
- velocity_damping: 속도 감쇠율
- center_pull_coeff: 중심 복귀 항의 강도
- center_reward_coeff: 중심 이탈 패널티 계수

## 논문 수식 반영 항목

현재 환경은 논문 정식화의 다음 요소를 반영합니다.

- Dec-POMDP 관측 확장: o_i에 에너지(E_i), 경보 이력(H_i), 통신 지연(C_i 일부)을 추가
- 동적 신뢰도 업데이트:
	- T_ij(t+1) = (1-lambda)T_ij(t) + lambda*Psi_ij
	- Psi_ij = w_fr*FR_ij + w_cr*CR_ij + w_dr*DR_ij
	- T_ij < trust_threshold 이면 해당 링크를 격리(보안 리스크 가산)
- 보상 함수 확장:
	- 기본형(alpha*PDR - beta*Delay - gamma*FPR + delta*Trust)
	- 세분형(w_pdr, w_trust, w_delay, w_energy, w_security) 동시 반영
- 물리 통신 성공 확률:
	- P_succ = exp(-k * Interference)
	- 간섭은 거리 기반 항 + 악성 경로 부스트로 계산

설정 위치: config.yaml의 environment 섹션
- trust_update_rate, trust_w_fr, trust_w_cr, trust_w_dr, trust_threshold
- interference_k, interference_base, interference_distance_coeff, interference_malicious_boost
- energy_init, energy_move_coeff, energy_tx_coeff
- reward_alpha, reward_beta, reward_gamma, reward_delta
- reward_w_pdr, reward_w_trust, reward_w_delay, reward_w_energy, reward_w_security
- alert_decay

## 옵션: ns-3 링크 트레이스 브리지 실험

기본값은 비활성화이며, 켜면 환경 내부 거리 모델 대신 CSV 링크 트레이스를 사용합니다.

1) 테스트용 트레이스 생성

```bash
python utils/generate_mock_ns3_trace.py --output logs/ns3_link_trace.csv --num-drones 10 --steps 60
```

2) config.yaml에서 브리지 활성화

```yaml
ns3_bridge:
	enabled: true
	provider: csv_trace
	csv_trace_path: logs/ns3_link_trace.csv
	strict: false
```

3) 기존 실행 그대로 사용

```bash
python test.py
```

CSV 컬럼 스키마:
- step: 1부터 시작하는 시뮬레이션 step
- i, j: 노드 인덱스
- connected: 0/1
- delay_ms: 링크 지연(ms)
- delivery: 링크 전달률(0~1)
- hop: 홉 수 (미연결은 큰 값 권장)

### ns-3 이벤트 로그를 브리지 CSV로 변환

ns-3에서 패킷 이벤트 CSV를 뽑았다면 아래 스크립트로 바로 변환할 수 있습니다.

입력 CSV 권장 컬럼(별칭 지원):
- time_s (또는 time/timestamp)
- src, dst
- event: tx/rx/drop
- delay_ms (선택)
- hop (선택)

변환 실행:

```bash
python utils/convert_ns3_events_to_bridge_csv.py --input logs/ns3_events.csv --output logs/ns3_link_trace.csv --num-drones 10 --step-seconds 1.0
```

변환 후 config.yaml:

```yaml
ns3_bridge:
	enabled: true
	provider: csv_trace
	csv_trace_path: logs/ns3_link_trace.csv
	strict: false
```

### 원클릭 실행 (권장)

이벤트 CSV가 이미 있으면 한 번에 변환 + 테스트 + GIF 생성까지 실행할 수 있습니다.

```bash
python run_ns3_bridge_pipeline.py --events logs/ns3_events.csv --trace logs/ns3_link_trace.csv --num-drones 10 --step-seconds 1.0 --seeds 42,43,44,45,46 --policy trained --scenario Default --gif logs/ns3_bridge_view.gif
```

Windows 배치 파일:

```bash
run_ns3_bridge_pipeline.bat logs\ns3_events.csv
```

### ns-3 다운로드가 필요한가?

- 현재 저장소에는 이미 ns-3 소스 트리(ns-3-dev)가 포함되어 있습니다.
- 브리지 모드만 사용할 경우, ns-3를 실시간으로 빌드/실행할 필요는 없고 이벤트 CSV만 있으면 됩니다.
- ns-3 예제를 직접 새로 돌리고 싶을 때만 ns-3 빌드 환경(CMake, 컴파일러 등) 준비가 필요합니다.

## 추천 다음 단계

- MATD3 장기 학습(예: 1000ep)으로 동일 파이프라인 재실행
- `test.py --compare-bridge` 결과를 비교표에 추가
- `visualize_attack.py`로 동일 seed의 정책 행동 차이 GIF 비교

## 요구사항

- Python 3.9+
- torch>=2.0
- numpy>=1.24
- gymnasium>=0.29
- matplotlib>=3.7
- PyYAML>=6.0

## 실행 파이프라인

```bash
python run_all.py
# 또는
run_all.bat
# 또는
make run
```

## 라이선스

공군사관학교 소프트웨어응용(26-1학기) 연구 목적으로 작성되었습니다.
