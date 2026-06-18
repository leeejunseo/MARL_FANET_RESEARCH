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
│   └── plot_learning_curve.py  # 학습 수렴 그래프 생성
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

pip install torch numpy gymnasium matplotlib
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

### 3. 학습 수렴 그래프 생성

```bash
python utils/plot_learning_curve.py
```

`logs/learning_curve_high_dpi.png` (300 DPI) 파일이 생성됩니다.

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

## 학습 결과

1000 에피소드 학습 후 총 전술 보상은 약 **+100.26** 수준으로 수렴합니다.

![학습 수렴 그래프](logs/learning_curve_high_dpi.png)

---

## 라이선스

본 프로젝트는 공군사관학교 소프트웨어응용 연구 목적으로 작성되었습니다.
