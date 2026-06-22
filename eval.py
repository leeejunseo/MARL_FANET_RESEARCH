import csv
import os
import numpy as np
import torch

from ns3_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from analysis.malicious_detector import MaliciousNodeDetector
from utils.config import load_config
from utils.metrics_logger import EvalMetricsLogger


def apply_ablation_obs(obs, ablation_cfg):
    if obs is None:
        return obs
    obs_out = obs.copy()
    if not ablation_cfg.get("use_xai", True):
        obs_out[:, 6:] = 0.0
    elif not ablation_cfg.get("use_trust", True):
        obs_out[:, 6] = 0.5
    return obs_out


def load_agents(env, config):
    num_drones = config["environment"]["num_drones"]
    action_dim = env.action_dim
    obs_dim = env.obs_dim
    episode = config["evaluation"]["model_episode"]
    prefix = config["evaluation"]["actor_model_prefix"]

    agents = [MADDPGAgent(obs_dim, env.state_dim, action_dim, num_drones, agent_id=i) for i in range(num_drones)]
    for i, agent in enumerate(agents):
        path = prefix.format(i=i, episode=episode)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Actor 모델을 찾을 수 없습니다: {path}")
        agent.actor.load_state_dict(torch.load(path, map_location="cpu"))
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
            step_features.append(info["node_features"])
            step_labels.append(info["node_labels"])
            proba = detector.predict_proba(info["node_features"])
            predicted = (proba >= detection_threshold).astype(int)
            detection_rates.append(np.mean(predicted == info["node_labels"]))

        obs = next_obs
        state = next_state
        obs_for_action = apply_ablation_obs(obs, ablation_cfg)

        if terminated:
            break

    return {
        "total_reward": total_reward,
        "avg_pdr": float(np.mean(pdr_values)) if pdr_values else 0.0,
        "avg_delay_ms": float(np.mean(delay_values)) if delay_values else 0.0,
        "avg_hop": float(np.mean(hop_values)) if hop_values else 0.0,
        "avg_trust": float(np.mean(trust_values)) if trust_values else 0.0,
        "avg_disconnect": float(np.mean(disconnect_values)) if disconnect_values else 0.0,
        "avg_detection": float(np.mean(detection_rates)) if detection_rates else None,
        "step_features": np.vstack(step_features) if step_features else None,
        "step_labels": np.concatenate(step_labels) if step_labels else None,
    }


def main():
    config = load_config()
    env_cfg = config["environment"]
    eval_cfg = config["evaluation"]

    scenarios = eval_cfg.get("scenarios") or [
        {
            "name": "Default",
            "malicious_ratio": env_cfg.get("malicious_ratio", 0.0),
            "malicious_behavior": env_cfg.get("malicious_behavior", "drop_and_trust"),
            "malicious_drop_rate": env_cfg.get("malicious_drop_rate", 0.4),
            "trust_noise": env_cfg.get("trust_noise", 0.05),
        }
    ]
    logger = EvalMetricsLogger(eval_cfg["output_csv"])

    print("=== EMARL-XAI 평가 실행 ===")
    print(f"에피소드 수: {eval_cfg['episodes']} | 최대 스텝: {eval_cfg['max_steps']}")

    all_features = []
    all_labels = []
    roc_path_template = eval_cfg.get("roc_data_path_template", eval_cfg["roc_data_path"])
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
            logger.log_episode(episode, stats, scenario=scenario["name"])

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

    if all_features and all_labels:
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        os.makedirs(os.path.dirname(eval_cfg["roc_data_path"]), exist_ok=True)
        np.savez(eval_cfg["roc_data_path"], features=all_features, labels=all_labels)
        print(f"통합 ROC 데이터 저장 완료: {eval_cfg['roc_data_path']}")

    print(f"평가 결과 CSV 저장 완료: {eval_cfg['output_csv']}")


if __name__ == "__main__":
    main()
