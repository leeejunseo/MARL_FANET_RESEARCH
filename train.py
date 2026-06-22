import os
import numpy as np
import torch
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from analysis.malicious_detector import MaliciousNodeDetector
from utils.replay_buffer import ReplayBuffer
from utils.tacview_logger import TacviewLogger
from utils.metrics_logger import MetricsLogger
from utils.config import load_config


def apply_ablation_obs(obs, ablation_cfg):
    """XAI/Trust 특성 ablation 처리를 진행합니다."""
    if obs is None:
        return obs
    obs_out = obs.copy()
    if not ablation_cfg.get("use_xai", True):
        obs_out[:, 6:] = 0.0
    elif not ablation_cfg.get("use_trust", True):
        obs_out[:, 6] = 0.5
    return obs_out


def main():
    config = load_config()
    env_cfg = config["environment"]
    training_cfg = config["training"]

    print("========================================================")
    print("  MARL-FANET 심층 학습 및 Tacview 시각화 연동 시스템")
    print("========================================================")
    
    num_drones = env_cfg["num_drones"]
    env = AdvancedFANETEnv(
        num_drones=num_drones,
        R_c=env_cfg["R_c"],
        d_safe=env_cfg["d_safe"],
        max_pos=env_cfg["max_pos"],
        max_vel=env_cfg["max_vel"],
        malicious_ratio=env_cfg.get("malicious_ratio", 0.0),
        malicious_drop_rate=env_cfg.get("malicious_drop_rate", 0.4),
        trust_noise=env_cfg.get("trust_noise", 0.05),
    )
    lr = training_cfg["lr"]
    gamma = training_cfg["gamma"]
    tau = training_cfg["tau"]
    ablation_cfg = training_cfg.get("ablation", {})

    agents = [MADDPGAgent(env.obs_dim, env.state_dim, env.action_dim, num_drones, agent_id=i, lr=lr) for i in range(num_drones)]
    replay_buffer = ReplayBuffer(100000, env.obs_dim, env.state_dim, env.action_dim, num_drones)
    
    # 학습 하이퍼파라미터
    batch_size = training_cfg["batch_size"]
    max_episodes = training_cfg["max_episodes"]
    max_steps = training_cfg["max_steps"]
    warmup_steps = training_cfg["warmup_steps"]
    save_interval = training_cfg["save_interval"]
    
    global_step = 0
    model_dir = training_cfg["model_dir"]
    metrics_logger = MetricsLogger(training_cfg["log_path"])
    
    print("\n[알림] 대규모 데이터 수집 및 인공지능 최적화 시작...")
    
    detector = MaliciousNodeDetector()
    for episode in range(1, max_episodes + 1):
        obs, state = env.reset()
        episode_reward = 0
        pdr_values = []
        delay_values = []
        hop_values = []
        trust_values = []
        disconnect_values = []
        detection_rates = []

        # 마지막 에피소드에서만 Tacview 로그를 기록하여 최종 전술 확인
        record_tacview = (episode == max_episodes)
        if record_tacview:
            tacview = TacviewLogger(f"logs/swarm_final_ep{episode}.acmi")
            print("\n[알림] 최종 에피소드 전술 기동 Tacview 로그 기록 중...")
            
        for step in range(max_steps):
            global_step += 1
            
            # 탐색 및 행동 결정
            actions = []
            obs_for_action = apply_ablation_obs(obs, ablation_cfg)
            for i in range(num_drones):
                if global_step < warmup_steps:
                    action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
                else:
                    explore = not record_tacview
                    action = agents[i].act(obs_for_action[i], explore=explore)
                actions.append(action)
            actions = np.array(actions)
            
            next_obs, next_state, rewards, terminated, info = env.step(actions)

            if info.get("node_features") is not None and info.get("node_labels") is not None:
                proba = detector.predict_proba(info["node_features"])
                predicted = (proba >= training_cfg.get("detection_threshold", 0.5)).astype(int)
                detection_rates.append(np.mean(predicted == info["node_labels"]))

                detection_cfg = training_cfg.get("detection_reward", {})
                tp_reward = detection_cfg.get("tp", 0.5)
                fp_penalty = detection_cfg.get("fp", -0.2)
                fn_penalty = detection_cfg.get("fn", -0.5)
                tn_reward = detection_cfg.get("tn", 0.0)
                shaped_rewards = np.zeros(num_drones, dtype=np.float32)

                for i in range(num_drones):
                    if info["node_labels"][i] == 1:
                        shaped_rewards[i] += tp_reward if predicted[i] == 1 else fn_penalty
                    else:
                        shaped_rewards[i] += tn_reward if predicted[i] == 0 else fp_penalty

                rewards = rewards + shaped_rewards
                info["detection_shaping"] = shaped_rewards.tolist()

            replay_buffer.add(obs_for_action, state, actions, rewards, apply_ablation_obs(next_obs, ablation_cfg), next_state, terminated)

            if record_tacview:
                tacview.log_step(time_step=float(step), positions=env.positions)

            obs = next_obs
            state = next_state
            episode_reward += np.sum(rewards)

            pdr_values.append(info.get("pdr", 0.0))
            delay_values.append(info.get("avg_delay_ms", 0.0))
            hop_values.append(info.get("avg_hop", 0.0))
            trust_values.append(np.mean(info.get("trust_scores", [0.0])))
            disconnect_values.append(info.get("disconnect_ratio", 0.0))

            if info.get("node_features") is not None and info.get("node_labels") is not None:
                proba = detector.predict_proba(info["node_features"])
                predicted = (proba >= training_cfg.get("detection_threshold", 0.5)).astype(int)
                detection_rates.append(np.mean(predicted == info["node_labels"]))

                detection_cfg = training_cfg.get("detection_reward", {})
                tp_reward = detection_cfg.get("tp", 0.5)
                fp_penalty = detection_cfg.get("fp", -0.2)
                fn_penalty = detection_cfg.get("fn", -0.5)
                tn_reward = detection_cfg.get("tn", 0.0)
                shaped_rewards = np.zeros(num_drones, dtype=np.float32)

                for i in range(num_drones):
                    if info["node_labels"][i] == 1:
                        shaped_rewards[i] += tp_reward if predicted[i] == 1 else fn_penalty
                    else:
                        shaped_rewards[i] += tn_reward if predicted[i] == 0 else fp_penalty

                rewards = rewards + shaped_rewards
                info["detection_shaping"] = shaped_rewards.tolist()

            if global_step >= warmup_steps and replay_buffer.size >= batch_size:
                sample_data = replay_buffer.sample(batch_size)
                for agent in agents:
                    agent.update(sample_data, agents, gamma=gamma, tau=tau)
                    
            if terminated:
                break
                
        avg_pdr = float(np.mean(pdr_values)) if pdr_values else 0.0
        avg_delay_ms = float(np.mean(delay_values)) if delay_values else 0.0
        avg_hop = float(np.mean(hop_values)) if hop_values else 0.0
        avg_trust = float(np.mean(trust_values)) if trust_values else 0.0
        avg_disconnect = float(np.mean(disconnect_values)) if disconnect_values else 0.0
        avg_detection = float(np.mean(detection_rates)) if detection_rates else None

        metrics_logger.log_episode(
            episode,
            episode_reward,
            num_drones,
            avg_pdr=avg_pdr,
            avg_delay_ms=avg_delay_ms,
            avg_hop=avg_hop,
            avg_trust=avg_trust,
            avg_disconnect=avg_disconnect,
            avg_detection=avg_detection,
        )

        # 로그 출력 (10 에피소드마다)
        if episode % 10 == 0 or episode == 1:
            print(f"에피소드 {episode:04d} | 총 전술 보상 합계: {episode_reward:7.2f} | 버퍼 크기: {replay_buffer.size:6d}")
            
        # 지정된 주기마다 모델 저장
        if episode % save_interval == 0:
            for agent in agents:
                agent.save_models(model_dir, episode)
            print(f"  -> [저장 완료] 에피소드 {episode} 신경망 가중치 저장됨 ({model_dir}/)")

    print("\n========================================================")
    print("  [연구 완료] 모델 저장 및 Tacview ACMI 로그 추출 완료")
    print("========================================================")

if __name__ == "__main__":
    main()
