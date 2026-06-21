# MARL-FANET 전술 스웜 프레임워크

**다중 에이전트 강화학습(MADDPG)** 기반 **FANET(Flying Ad-hoc Network) 드론 스웜** 전술 기동 시뮬레이션 및 Tacview 3D 시각화 연동 연구 프로젝트입니다.

공군사관학교 소프트웨어응용(26-1학기) 연구 과제로, Dec-POMDP 환경에서 UAV 스웜이 **탐지 영역 확장**, **충돌 회피**, **통신망 유지** 세 가지 목표를 동시에 최적화하도록 학습합니다.

---

## 주요 특징

- **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** 알고리즘 구현
  - Centralized Training, Decentralized Execution (CTDE) 구조
  - Actor-Critic 네트워크 + Target Network Soft Update
- **FANET 전술 환경** (`AdvancedFANETEnv`)
  - 3기 드론 스웜, 3D 공간 물리 시뮬레이션
  - 다중 목적 보상 함수 (커버리지 / 충돌 회피 / 통신 유지)
- **Tacview ACMI 로그** 연동
  - 학습·평가 결과를 3D 전술 뷰어에서 재생 가능
  - 통신 단절 시 드론 색상 Yellow 경고 표시
- **학습 수렴 그래프** 자동 생성 (논문용 300 DPI)
- **논문용 시각화 모듈** — ROC/AUC, 프로토콜 비교, XAI 히트맵/SHAP

---

## 프로젝트 구조

```
marl_fanet_research/
├── train.py                    # MADDPG 심층 학습 메인 스크립트
├── test.py                     # 학습된 모델 로드 및 전술 평가
├── agents/
│   └── maddpg.py               # Actor/Critic 네트워크 및 MADDPG 에이전트
├── ns3_wrapper/
│   └── fanet_env.py            # FANET 전술 시뮬레이션 환경 (Gymnasium)
├── utils/
│   ├── replay_buffer.py        # 다중 에이전트 경험 재생 버퍼
│   ├── tacview_logger.py       # Tacview ACMI 파일 로거
│   ├── metrics_logger.py       # 학습 보상 CSV 로깅
│   ├── plot_style.py           # 논문용 matplotlib 공통 스타일
│   ├── plot_learning_curve.py  # 학습 수렴 곡선
│   ├── plot_roc_auc.py         # ROC Curve & AUC
│   ├── plot_bar_comparison.py  # 프로토콜 성능 바 차트
│   ├── plot_xai_heatmap.py     # XAI 히트맵 / SHAP Summary
│   └── generate_all_plots.py   # 전체 시각화 일괄 생성
├── analysis/
│   ├── malicious_detector.py   # 악의적 노드 탐지 (ROC/AUC 평가)
│   └── xai_explainer.py        # XAI 특성 기여도 분석
├── requirements.txt
├── models/weights/             # 학습된 신경망 가중치 (.pth)
└── logs/                       # Tacview ACMI 로그 및 학습 그래프
```

---

## 환경 설정

### 요구 사항

- Python 3.9+
- PyTorch
- NumPy
- Gymnasium
- Matplotlib

### 설치

```bash
git clone https://github.com/leeejunseo/MARL_FANET_RESEARCH.git
cd MARL_FANET_RESEARCH

pip install -r requirements.txt
```

---

## 사용 방법

### 1. 학습 실행

```bash
python train.py
```

| 하이퍼파라미터 | 값 | 설명 |
|---|---|---|
| `num_drones` | 3 | 스웜 드론 수 |
| `max_episodes` | 1000 | 총 학습 에피소드 |
| `max_steps` | 60 | 에피소드당 최대 스텝 |
| `batch_size` | 128 | 미니배치 크기 |
| `warmup_steps` | 500 | 초기 무작위 탐색 스텝 |
| `save_interval` | 200 | 모델 저장 주기 (에피소드) |

학습 완료 후:
- `models/weights/` — 200, 400, …, 1000 에피소드 Actor/Critic 가중치 저장
- `logs/swarm_final_ep1000.acmi` — 최종 에피소드 Tacview 로그

### 2. 학습된 모델 평가

```bash
python test.py
```

1000 에피소드 모델을 로드하여 탐색 노이즈 없이 전술 기동을 평가하고, `logs/swarm_test_eval.acmi` 파일을 생성합니다.

### 3. 논문용 시각화 일괄 생성

```bash
python utils/generate_all_plots.py
```

| 스크립트 | 출력 파일 | 설명 |
|---|---|---|
| `plot_learning_curve.py` | `logs/learning_curve_high_dpi.png` | 학습 수렴 곡선 (Convergence Plot) |
| `plot_roc_auc.py` | `logs/roc_curve_auc.png` | 악의적 노드 탐지 ROC & AUC |
| `plot_bar_comparison.py` | `logs/protocol_comparison_bar.png` | AODV / MARL / EMARL-XAI 성능 비교 |
| `plot_xai_heatmap.py` | `logs/xai_*.png` | XAI 히트맵, SHAP Summary, 특성 중요도 |

개별 실행도 가능합니다:

```bash
python utils/plot_roc_auc.py
python utils/plot_bar_comparison.py
python utils/plot_xai_heatmap.py
```

> `train.py` 실행 시 `logs/training_rewards.csv`에 에피소드별 보상이 기록되며, 수렴 곡선은 실측 데이터를 우선 사용합니다.

### 4. Tacview에서 결과 확인

[Tacview](https://www.tacview.net/) 프로그램에서 아래 ACMI 파일을 열어 3D 전술 기동을 재생할 수 있습니다.

| 파일 | 설명 |
|---|---|
| `logs/swarm_final_ep1000.acmi` | 학습 마지막 에피소드 기동 |
| `logs/swarm_test_eval.acmi` | 평가 모드(노이즈 없음) 기동 |

---

## FANET 환경 상세

### 물리 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `max_pos` | 1000 m | 시뮬레이션 공간 크기 |
| `max_vel` | 20 m/s | 최대 속도 |
| `R_c` | 300 m | FANET 통신 반경 |
| `d_safe` | 30 m | 드론 간 최소 안전 거리 |

### 관측 / 행동 공간

- **관측 (obs_dim=6)**: 정규화된 위치(3) + 속도(3) — 에이전트별 부분 관측
- **전역 상태 (state_dim=18)**: 전체 드론 관측의 연결 — Critic 입력
- **행동 (action_dim=3)**: 3축 가속도 명령 [-1, 1]

### 보상 함수

각 드론 $i$에 대해:

$$R_i = 1.0 \cdot r_{cov} + 2.5 \cdot r_{col} + 3.0 \cdot r_{conn}$$

| 항목 | 설명 |
|---|---|
| $r_{cov}$ | 다른 드론과의 거리 합 → 탐지 영역 확장 유도 |
| $r_{col}$ | 안전 거리($d_{safe}$) 미만 진입 시 충돌 페널티 |
| $r_{conn}$ | 통신 반경($R_c$) 이탈 시 연결 유지 페널티 |

---

## MADDPG 알고리즘 개요

```
┌─────────────────────────────────────────────────────┐
│                  Centralized Critic                  │
│  Q(s, a₁, a₂, a₃) ← 전역 상태 + 모든 에이전트 행동  │
└─────────────────────────────────────────────────────┘
         ↑                              ↑
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Actor₀      │  │  Actor₁      │  │  Actor₂      │
│  π(o₀)→a₀   │  │  π(o₁)→a₁   │  │  π(o₂)→a₂   │
└──────────────┘  └──────────────┘  └──────────────┘
   Decentralized Execution (각 드론 독립 행동)
```

- **Critic**: MSBE(Mean Squared Bellman Error) 손실로 Q-함수 학습
- **Actor**: Policy Gradient로 Critic Q값 최대화
- **Target Network**: Soft Update ($\tau = 0.01$)으로 학습 안정화
- **탐색**: 가우시안 노이즈 ($\sigma = 0.1$) 추가

---

## 논문용 시각화

### ROC Curve & AUC — 악의적 노드 탐지

SNR, Trust Score, Hop Count, Packet Drop Rate, Latency 기반으로 정상/악의적 노드를 분류합니다. TPR-FPR 곡선과 AUC 값으로 EMARL-XAI의 탐지 우위를 증명합니다.

![ROC Curve](logs/roc_curve_auc.png)

### Convergence Plot — 학습 수렴

에피소드 진행에 따른 누적 보상 변화와 이동 평균선으로 학습 안정성을 확인합니다.

![학습 수렴 그래프](logs/learning_curve_high_dpi.png)

### Bar Chart — 프로토콜 성능 비교

정상 환경 vs 악의적 공격 환경에서 AODV, Standard MARL, EMARL-XAI의 PDR·Delay·Detection Accuracy를 비교합니다.

![프로토콜 비교](logs/protocol_comparison_bar.png)

### XAI — 특성 기여도 분석

Integrated Gradients 기반 SHAP Summary Plot과 상관 히트맵으로 라우팅 의사결정에 영향을 미치는 변수(SNR, Trust Score, Hop Count 등)를 시각화합니다.

| 그래프 | 파일 |
|---|---|
| 특성 상관 히트맵 | `logs/xai_feature_heatmap.png` |
| SHAP Summary Plot | `logs/xai_shap_summary.png` |
| 특성 중요도 바 차트 | `logs/xai_feature_importance.png` |

---

## 학습 결과

1000 에피소드 학습 후 총 전술 보상은 약 **+100.26** 수준으로 수렴합니다.

---

## 라이선스

본 프로젝트는 공군사관학교 소프트웨어응용 연구 목적으로 작성되었습니다.
