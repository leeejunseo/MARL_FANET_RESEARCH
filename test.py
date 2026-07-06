import argparse
import copy
import os
import numpy as np
import torch
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from agents.matd3 import MATD3Agent
from analysis.malicious_detector import MaliciousNodeDetector
from utils.metrics_logger import EvalMetricsLogger
from utils.config import load_config
from utils.model_metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.algorithm import normalize_algorithm, display_algorithm, resolve_actor_prefix, resolve_file_path
from ns3_wrapper.provider_factory import build_link_provider


def infer_actor_obs_dim_from_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    for key, value in state_dict.items():
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.shape[1]
    raise ValueError(f"Cannot infer actor obs_dim from checkpoint: {path}")


def evaluate_policy_episode(env, action_policy, max_steps, detector=None, seed=42):
    obs, state = env.reset(seed=seed)
    num_drones = env.num_drones
    pdr_values = []
    delay_values = []
    hop_values = []
    trust_values = []
    disconnect_values = []
    detection_rates = []
    step_labels = []
    step_predictions = []
    episode_reward = 0.0

    for step in range(max_steps):
        if action_policy == "random":
            actions = np.random.uniform(-1.0, 1.0, size=(num_drones, env.action_dim))
        else:
            actions = [agent.act(obs[i], explore=False) for i, agent in enumerate(action_policy)]
            actions = np.array(actions)

        next_obs, next_state, rewards, terminated, info = env.step(actions)
        episode_reward += float(np.sum(rewards))

        pdr_values.append(info.get("pdr", 0.0))
        delay_values.append(info.get("avg_delay_ms", 0.0))
        hop_values.append(info.get("avg_hop", 0.0))
        trust_values.append(np.mean(info.get("trust_scores", [0.0])))
        disconnect_values.append(info.get("disconnect_ratio", 0.0))

        if detector is not None and info.get("node_features") is not None and info.get("node_labels") is not None:
            features = info["node_features"]
            labels = info["node_labels"]
            proba = detector.predict_proba(features)
            predicted = (proba >= 0.5).astype(int)
            detection_rates.append(np.mean(predicted == labels))
            step_labels.append(labels)
            step_predictions.append(predicted)

        obs = next_obs
        state = next_state
        if terminated:
            break

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
        "total_reward": episode_reward,
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
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Policy benchmark with multi-seed averaging.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. e.g., 42,43,44")
    parser.add_argument("--compare-bridge", action="store_true", help="Evaluate both bridge OFF and ON.")
    parser.add_argument("--bridge-only", choices=["on", "off"], default=None, help="Force one bridge mode.")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def parse_seed_list(seed_text, default_seeds):
    if seed_text is None:
        return list(default_seeds)
    values = []
    for token in seed_text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values or list(default_seeds)


def average_stats(stats_list):
    out = {}
    if not stats_list:
        return out

    keys = [
        "total_reward",
        "avg_pdr",
        "avg_delay_ms",
        "avg_hop",
        "avg_trust",
        "avg_disconnect",
        "avg_detection",
        "avg_detection_accuracy",
        "avg_detection_precision",
        "avg_detection_recall",
        "avg_detection_f1",
    ]
    for key in keys:
        values = [s[key] for s in stats_list if s.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def build_env_from_config(config):
    env_cfg = config["environment"]
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
        connectivity_guard_coeff=env_cfg.get("connectivity_guard_coeff", 0.35),
        min_neighbor_target=env_cfg.get("min_neighbor_target", 2),
        alert_decay=env_cfg.get("alert_decay", 0.9),
        link_provider=link_provider,
    )
    return env


def load_trained_agents(env, config):
    eval_cfg = config["evaluation"]
    algorithm = normalize_algorithm(config.get("training", {}).get("algorithm", "maddpg"))
    model_prefix = resolve_actor_prefix(eval_cfg["actor_model_prefix"], algorithm)
    episode_to_load = eval_cfg["model_episode"]
    matd3_cfg = config.get("training", {}).get("matd3", {})
    agents = []
    for i in range(env.num_drones):
        path = model_prefix.format(i=i, episode=episode_to_load)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Actor 모델을 찾을 수 없습니다: {path}")
        actor_obs_dim = infer_actor_obs_dim_from_checkpoint(path)
        if algorithm == "matd3":
            agent = MATD3Agent(
                env.obs_dim,
                env.state_dim,
                env.action_dim,
                env.num_drones,
                agent_id=i,
                actor_lr=matd3_cfg.get("actor_lr", config["training"].get("lr", 1e-3)),
                critic_lr=matd3_cfg.get("critic_lr", config["training"].get("lr", 1e-3)),
                actor_obs_dim=actor_obs_dim,
                policy_delay=matd3_cfg.get("policy_delay", 2),
                target_policy_noise=matd3_cfg.get("target_policy_noise", 0.2),
                target_noise_clip=matd3_cfg.get("target_noise_clip", 0.5),
                explore_noise_std=matd3_cfg.get("explore_noise_std", 0.1),
            )
        else:
            agent = MADDPGAgent(
                env.obs_dim,
                env.state_dim,
                env.action_dim,
                env.num_drones,
                agent_id=i,
                actor_obs_dim=actor_obs_dim,
            )
        agent.actor.load_state_dict(torch.load(path, map_location="cpu"))
        agents.append(agent)
    return agents


def test_inference():
    args = parse_args()
    config = load_config()
    env_cfg = config["environment"]
    eval_cfg = config["evaluation"]
    algorithm = normalize_algorithm(config.get("training", {}).get("algorithm", "maddpg"))
    policy_label = display_algorithm(algorithm)
    default_seeds = eval_cfg.get("test_seeds", [42, 43, 44, 45, 46])
    seeds = parse_seed_list(args.seeds, default_seeds)
    max_steps = args.max_steps if args.max_steps is not None else eval_cfg["max_steps"]

    compare_bridge = args.compare_bridge or eval_cfg.get("test_compare_bridge", True)
    if args.bridge_only is not None:
        compare_bridge = False

    bridge_modes = []
    if args.bridge_only == "on":
        bridge_modes = [True]
    elif args.bridge_only == "off":
        bridge_modes = [False]
    elif compare_bridge:
        bridge_modes = [False, True]
    else:
        bridge_modes = [config.get("ns3_bridge", {}).get("enabled", False)]

    logger_path = resolve_file_path(eval_cfg.get("test_output_csv", "logs/test_metrics.csv"), algorithm)
    logger = EvalMetricsLogger(logger_path, overwrite=eval_cfg.get("test_overwrite_csv", True))

    print(f"=== {policy_label} 기반 전술 기동 평가 시작 ===")
    print(f"seeds={seeds} | max_steps={max_steps} | bridge_modes={bridge_modes}")

    detector = MaliciousNodeDetector()
    row_id = 1
    for bridge_enabled in bridge_modes:
        cfg_local = copy.deepcopy(config)
        cfg_local.setdefault("ns3_bridge", {})
        cfg_local["ns3_bridge"]["enabled"] = bridge_enabled

        try:
            env = build_env_from_config(cfg_local)
        except Exception as exc:
            print(f"[건너뜀] bridge={bridge_enabled} 환경 구성 실패: {exc}")
            continue

        agents = load_trained_agents(env, cfg_local)
        print(
            f"\n[모드] bridge={bridge_enabled} | model_ep={eval_cfg['model_episode']} | link_source={cfg_local['ns3_bridge'].get('provider', 'distance_model') if bridge_enabled else 'distance_model'}"
        )

        trained_runs = []
        random_runs = []
        for seed in seeds:
            trained_stats = evaluate_policy_episode(
                env,
                agents,
                max_steps,
                detector=detector,
                seed=seed,
            )
            random_stats = evaluate_policy_episode(
                env,
                "random",
                max_steps,
                detector=detector,
                seed=seed,
            )

            trained_runs.append(trained_stats)
            random_runs.append(random_stats)
            logger.log_episode(row_id, trained_stats, scenario=f"Default|bridge={bridge_enabled}|seed={seed}", policy=policy_label)
            row_id += 1
            logger.log_episode(row_id, random_stats, scenario=f"Default|bridge={bridge_enabled}|seed={seed}", policy="Random")
            row_id += 1

        trained_avg = average_stats(trained_runs)
        random_avg = average_stats(random_runs)

        improvement = {
            "reward_gain": trained_avg.get("total_reward", 0.0) - random_avg.get("total_reward", 0.0),
            "pdr_gain": trained_avg.get("avg_pdr", 0.0) - random_avg.get("avg_pdr", 0.0),
            "delay_gain": random_avg.get("avg_delay_ms", 0.0) - trained_avg.get("avg_delay_ms", 0.0),
            "trust_gain": trained_avg.get("avg_trust", 0.0) - random_avg.get("avg_trust", 0.0),
        }

        print("=== 학습된 정책 vs 무작위 정책 평균 비교 ===")
        print(f"총 보상 차이: {improvement['reward_gain']:.2f}")
        print(f"PDR 차이: {improvement['pdr_gain']:.3f}")
        print(f"지연 감소: {improvement['delay_gain']:.2f} ms")
        print(f"평균 신뢰도 차이: {improvement['trust_gain']:.3f}")

    print("\n=== 평가 종료 ===")
    print(f"  테스트 메트릭 CSV: {logger_path}")


if __name__ == "__main__":
    test_inference()
