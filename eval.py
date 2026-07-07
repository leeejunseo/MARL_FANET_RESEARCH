import csv
import os
import numpy as np
import torch

from fanet_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from agents.matd3 import MATD3Agent
from analysis.malicious_detector import MaliciousNodeDetector
from utils.config import load_config
from utils.metrics_logger import EvalMetricsLogger
from utils.model_metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.algorithm import normalize_algorithm, display_algorithm, resolve_actor_prefix, resolve_file_path


def apply_ablation_obs(obs, ablation_cfg):
    if obs is None:
        return obs
    obs_out = obs.copy()
    if not ablation_cfg.get("use_xai", True):
        obs_out[:, 6:] = 0.0
    elif not ablation_cfg.get("use_trust", True):
        obs_out[:, 6] = 0.5
    return obs_out


def infer_actor_obs_dim_from_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    for key, value in state_dict.items():
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.shape[1]
    raise ValueError(f"Cannot infer actor obs_dim from checkpoint: {path}")


def load_agents(env, config):
    num_drones = config["environment"]["num_drones"]
    action_dim = env.action_dim
    obs_dim = env.obs_dim
    episode = config["evaluation"]["model_episode"]
    algorithm = normalize_algorithm(config.get("training", {}).get("algorithm", "maddpg"))
    prefix = resolve_actor_prefix(config["evaluation"]["actor_model_prefix"], algorithm)
    use_marl = config.get("training", {}).get("ablation", {}).get("use_marl", True)
    matd3_cfg = config.get("training", {}).get("matd3", {})

    agents = []
    for i in range(num_drones):
        path = prefix.format(i=i, episode=episode)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Actor 모델을 찾을 수 없습니다: {path}")
        actor_obs_dim = infer_actor_obs_dim_from_checkpoint(path)
        if algorithm == "matd3":
            agent = MATD3Agent(
                obs_dim,
                env.state_dim,
                action_dim,
                num_drones,
                agent_id=i,
                actor_lr=matd3_cfg.get("actor_lr", config["training"].get("lr", 1e-3)),
                critic_lr=matd3_cfg.get("critic_lr", config["training"].get("lr", 1e-3)),
                use_marl=use_marl,
                actor_obs_dim=actor_obs_dim,
                policy_delay=matd3_cfg.get("policy_delay", 2),
                target_policy_noise=matd3_cfg.get("target_policy_noise", 0.2),
                target_noise_clip=matd3_cfg.get("target_noise_clip", 0.5),
                explore_noise_std=matd3_cfg.get("explore_noise_std", 0.1),
            )
        else:
            agent = MADDPGAgent(
                obs_dim,
                env.state_dim,
                action_dim,
                num_drones,
                agent_id=i,
                use_marl=use_marl,
                actor_obs_dim=actor_obs_dim,
            )
        agent.actor.load_state_dict(torch.load(path, map_location="cpu"))
        agents.append(agent)
    return agents


def evaluate_episode(env, agents, max_steps, detection_threshold, ablation_cfg=None):
    obs, state = env.reset()
    if ablation_cfg is None:
        ablation_cfg = {}
    total_reward = 0.0
    pdr_values = []
    delay_values = []
    hop_values = []
    trust_values = []
    disconnect_values = []
    detection_rates = []
    detector = MaliciousNodeDetector()
    step_features = []
    step_labels = []
    step_predictions = []

    obs_for_action = apply_ablation_obs(obs, ablation_cfg)
    for _ in range(max_steps):
        actions = [agents[i].act(obs_for_action[i], explore=False) for i in range(len(agents))]
        next_obs, next_state, rewards, terminated, info = env.step(np.array(actions))

        total_reward += np.sum(rewards)
        pdr_values.append(info["pdr"])
        delay_values.append(info["avg_delay_ms"])
        hop_values.append(info["avg_hop"])
        trust_values.append(np.mean(info["trust_scores"]))
        disconnect_values.append(info["disconnect_ratio"])

        if info.get("node_features") is not None and info.get("node_labels") is not None:
            features = info["node_features"]
            labels = info["node_labels"]
            proba = detector.predict_proba(features)
            predicted = (proba >= detection_threshold).astype(int)
            step_features.append(features)
            step_labels.append(labels)
            step_predictions.append(predicted)
            detection_rates.append(np.mean(predicted == labels))

        obs = next_obs
        state = next_state
        obs_for_action = apply_ablation_obs(obs, ablation_cfg)

        if terminated:
            break

    all_step_features = np.vstack(step_features) if step_features else None
    all_step_labels = np.concatenate(step_labels) if step_labels else None
    all_step_predictions = np.concatenate(step_predictions) if step_predictions else None

    detection_accuracy = (
        float(accuracy_score(all_step_labels, all_step_predictions))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )
    detection_precision = (
        float(precision_score(all_step_labels, all_step_predictions, average="binary", pos_label=1))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )
    detection_recall = (
        float(recall_score(all_step_labels, all_step_predictions, average="binary", pos_label=1))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )
    detection_f1 = (
        float(f1_score(all_step_labels, all_step_predictions, average="binary", pos_label=1))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )

    return {
        "total_reward": total_reward,
        "avg_pdr": float(np.mean(pdr_values)) if pdr_values else 0.0,
        "avg_delay_ms": float(np.mean(delay_values)) if delay_values else 0.0,
        "avg_hop": float(np.mean(hop_values)) if hop_values else 0.0,
        "avg_trust": float(np.mean(trust_values)) if trust_values else 0.0,
        "avg_disconnect": float(np.mean(disconnect_values)) if disconnect_values else 0.0,
        "avg_detection": float(np.mean(detection_rates)) if detection_rates else None,
        "avg_detection_accuracy": detection_accuracy,
        "avg_detection_precision": detection_precision,
        "avg_detection_recall": detection_recall,
        "avg_detection_f1": detection_f1,
        "step_features": all_step_features,
        "step_labels": all_step_labels,
    }


def main():
    config = load_config()
    env_cfg = config["environment"]
    eval_cfg = config["evaluation"]
    algorithm = normalize_algorithm(config.get("training", {}).get("algorithm", "maddpg"))
    policy_label = display_algorithm(algorithm)

    scenarios = eval_cfg.get("scenarios") or [
        {
            "name": "Default",
            "malicious_ratio": env_cfg.get("malicious_ratio", 0.0),
            "malicious_behavior": env_cfg.get("malicious_behavior", "drop_and_trust"),
            "malicious_drop_rate": env_cfg.get("malicious_drop_rate", 0.4),
            "trust_noise": env_cfg.get("trust_noise", 0.05),
        }
    ]
    output_csv = resolve_file_path(eval_cfg["output_csv"], algorithm)
    logger = EvalMetricsLogger(output_csv, overwrite=eval_cfg.get("overwrite_csv", True))

    print(f"=== {policy_label} 평가 실행 ===")
    print(f"에피소드 수: {eval_cfg['episodes']} | 최대 스텝: {eval_cfg['max_steps']}")

    all_features = []
    all_labels = []
    roc_path_template = resolve_file_path(eval_cfg.get("roc_data_path_template", eval_cfg["roc_data_path"]), algorithm)
    ablation_cfg = config.get("training", {}).get("ablation", {})

    for scenario in scenarios:
        print(f"\n--- 시나리오: {scenario['name']} ({scenario['malicious_behavior']}) ---")
        env = AdvancedFANETEnv(
            num_drones=env_cfg["num_drones"],
            R_c=env_cfg["R_c"],
            d_safe=env_cfg["d_safe"],
            max_pos=env_cfg["max_pos"],
            max_vel=env_cfg["max_vel"],
            malicious_ratio=scenario.get("malicious_ratio", 0.0),
            malicious_drop_rate=scenario.get("malicious_drop_rate", 0.4),
            malicious_behavior=scenario.get("malicious_behavior", env_cfg.get("malicious_behavior", "drop_and_trust")),
            trust_noise=scenario.get("trust_noise", env_cfg.get("trust_noise", 0.05)),
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
            connectivity_guard_coeff=env_cfg.get("connectivity_guard_coeff", 0.35),
            min_neighbor_target=env_cfg.get("min_neighbor_target", 2),
            malicious_avoid_coeff=env_cfg.get("malicious_avoid_coeff", 0.65),
            suspicious_avoid_coeff=env_cfg.get("suspicious_avoid_coeff", 0.35),
            avoid_distance_factor=env_cfg.get("avoid_distance_factor", 1.15),
            alert_decay=env_cfg.get("alert_decay", 0.9),
        )
        agents = load_agents(env, config)

        scenario_features = []
        scenario_labels = []

        for episode in range(1, eval_cfg["episodes"] + 1):
            stats = evaluate_episode(
                env,
                agents,
                eval_cfg["max_steps"],
                eval_cfg["detection_threshold"],
                ablation_cfg,
            )
            logger.log_episode(episode, stats, scenario=scenario["name"], policy=policy_label)

            if stats["step_features"] is not None and stats["step_labels"] is not None:
                scenario_features.append(stats["step_features"])
                scenario_labels.append(stats["step_labels"])
                all_features.append(stats["step_features"])
                all_labels.append(stats["step_labels"])

            print(
                f"Episode {episode:02d} | reward={stats['total_reward']:.2f} "
                f"| PDR={stats['avg_pdr']:.3f} | delay={stats['avg_delay_ms']:.1f}ms "
                f"| hop={stats['avg_hop']:.2f} | trust={stats['avg_trust']:.2f} "
                f"| detect={stats['avg_detection'] if stats['avg_detection'] is not None else 'N/A'}"
            )

        if scenario_features and scenario_labels:
            scenario_features = np.vstack(scenario_features)
            scenario_labels = np.concatenate(scenario_labels)
            scenario_tag = scenario["name"].replace(" ", "_").lower()
            scenario_path = roc_path_template.format(scenario=scenario_tag) if "{scenario}" in roc_path_template else eval_cfg["roc_data_path"]
            os.makedirs(os.path.dirname(scenario_path), exist_ok=True)
            np.savez(scenario_path, features=scenario_features, labels=scenario_labels)
            print(f"시나리오별 ROC 데이터 저장 완료: {scenario_path}")

    roc_data_path = resolve_file_path(eval_cfg["roc_data_path"], algorithm)
    if all_features and all_labels:
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        os.makedirs(os.path.dirname(roc_data_path), exist_ok=True)
        np.savez(roc_data_path, features=all_features, labels=all_labels)
        print(f"통합 ROC 데이터 저장 완료: {roc_data_path}")

    print(f"평가 결과 CSV 저장 완료: {output_csv}")


if __name__ == "__main__":
    main()
