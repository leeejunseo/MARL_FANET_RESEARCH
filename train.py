import os
import numpy as np
import torch
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from analysis.malicious_detector import MaliciousNodeDetector
from utils.replay_buffer import ReplayBuffer
from utils.metrics_logger import MetricsLogger
from utils.config import load_config
from ns3_wrapper.provider_factory import build_link_provider


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
    print("  MARL-FANET 심층 학습 시스템")
    print("========================================================")
    
    num_drones = env_cfg["num_drones"]
    link_provider = build_link_provider(config, num_drones)
    env = AdvancedFANETEnv(
        num_drones=num_drones,
        R_c=env_cfg["R_c"],
        d_safe=env_cfg["d_safe"],
        max_pos=env_cfg["max_pos"],
        max_vel=env_cfg["max_vel"],
        malicious_ratio=env_cfg.get("malicious_ratio", 0.0),
        malicious_drop_rate=env_cfg.get("malicious_drop_rate", 0.4),
        trust_noise=env_cfg.get("trust_noise", 0.05),
        velocity_damping=env_cfg.get("velocity_damping", 0.05),
        center_pull_coeff=env_cfg.get("center_pull_coeff", 0.12),
        center_reward_coeff=env_cfg.get("center_reward_coeff", 0.6),
        reward_cov_coeff=env_cfg.get("reward_cov_coeff", 0.6),
        reward_col_coeff=env_cfg.get("reward_col_coeff", 2.5),
        reward_conn_coeff=env_cfg.get("reward_conn_coeff", 4.0),
        reward_trust_pos_coeff=env_cfg.get("reward_trust_pos_coeff", 1.2),
        reward_trust_neg_coeff=env_cfg.get("reward_trust_neg_coeff", 0.8),
        trust_update_rate=env_cfg.get("trust_update_rate", 0.2),
        trust_w_fr=env_cfg.get("trust_w_fr", 0.5),
        trust_w_cr=env_cfg.get("trust_w_cr", 0.3),
        trust_w_dr=env_cfg.get("trust_w_dr", 0.2),
        trust_threshold=env_cfg.get("trust_threshold", 0.35),
        interference_k=env_cfg.get("interference_k", 1.2),
        interference_base=env_cfg.get("interference_base", 0.1),
        interference_distance_coeff=env_cfg.get("interference_distance_coeff", 0.9),
        interference_malicious_boost=env_cfg.get("interference_malicious_boost", 0.6),
        energy_init=env_cfg.get("energy_init", 100.0),
        energy_move_coeff=env_cfg.get("energy_move_coeff", 0.08),
        energy_tx_coeff=env_cfg.get("energy_tx_coeff", 0.25),
        reward_alpha=env_cfg.get("reward_alpha", 1.2),
        reward_beta=env_cfg.get("reward_beta", 1.0),
        reward_gamma=env_cfg.get("reward_gamma", 0.8),
        reward_delta=env_cfg.get("reward_delta", 1.0),
        reward_w_pdr=env_cfg.get("reward_w_pdr", 1.0),
        reward_w_trust=env_cfg.get("reward_w_trust", 1.0),
        reward_w_delay=env_cfg.get("reward_w_delay", 1.0),
        reward_w_energy=env_cfg.get("reward_w_energy", 0.8),
        reward_w_security=env_cfg.get("reward_w_security", 1.2),
        alert_decay=env_cfg.get("alert_decay", 0.9),
        link_provider=link_provider,
    )
    lr = training_cfg["lr"]
    gamma = training_cfg["gamma"]
    tau = training_cfg["tau"]
    ablation_cfg = training_cfg.get("ablation", {})

    agents = [MADDPGAgent(
        env.obs_dim,
        env.state_dim,
        env.action_dim,
        num_drones,
        agent_id=i,
        lr=lr,
        use_marl=ablation_cfg.get("use_marl", True),
    ) for i in range(num_drones)]
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

        for step in range(max_steps):
            global_step += 1
            
            # 탐색 및 행동 결정
            actions = []
            obs_for_action = apply_ablation_obs(obs, ablation_cfg)
            for i in range(num_drones):
                if global_step < warmup_steps:
                    action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
                else:
                    explore = True
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

            obs = next_obs
            state = next_state
            episode_reward += np.sum(rewards)

            pdr_values.append(info.get("pdr", 0.0))
            delay_values.append(info.get("avg_delay_ms", 0.0))
            hop_values.append(info.get("avg_hop", 0.0))
            trust_values.append(np.mean(info.get("trust_scores", [0.0])))
            disconnect_values.append(info.get("disconnect_ratio", 0.0))

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
    print("  [연구 완료] 모델 저장 및 학습 로그 생성 완료")
    print("========================================================")

if __name__ == "__main__":
    main()
