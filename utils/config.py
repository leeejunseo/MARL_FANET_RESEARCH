import os

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_CONFIG = {
    "environment": {
        "num_drones": 3,
        "R_c": 300.0,
        "d_safe": 30.0,
        "max_pos": 1000.0,
        "max_vel": 20.0,
        "malicious_ratio": 0.0,
        "malicious_drop_rate": 0.4,
        "malicious_behavior": "drop_and_trust",
        "trust_noise": 0.05,
    },
    "training": {
        "algorithm": "maddpg",
        "max_episodes": 1000,
        "max_steps": 60,
        "warmup_steps": 500,
        "save_interval": 200,
        "batch_size": 128,
        "lr": 1e-3,
        "matd3": {
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "explore_noise_std": 0.1,
        },
        "gamma": 0.95,
        "tau": 0.01,
        "detection_threshold": 0.5,
        "model_dir": "models/weights",
        "log_path": "logs/training_rewards.csv",
        "detection_reward": {
            "tp": 0.5,
            "fp": -0.2,
            "fn": -0.5,
            "tn": 0.0,
        },
        "ablation": {
            "use_xai": True,
            "use_trust": True,
            "use_marl": True,
        },
    },
    "evaluation": {
        "scenarios": [
            {
                "name": "Default",
                "malicious_ratio": 0.33,
                "malicious_behavior": "drop_and_trust",
                "malicious_drop_rate": 0.4,
                "trust_noise": 0.05,
            },
            {
                "name": "Blackhole",
                "malicious_ratio": 0.33,
                "malicious_behavior": "blackhole",
                "malicious_drop_rate": 0.5,
                "trust_noise": 0.05,
            },
            {
                "name": "Selective Forwarding",
                "malicious_ratio": 0.33,
                "malicious_behavior": "selective_forwarding",
                "malicious_drop_rate": 0.4,
                "trust_noise": 0.05,
            },
            {
                "name": "Sybil",
                "malicious_ratio": 0.33,
                "malicious_behavior": "sybil",
                "malicious_drop_rate": 0.4,
                "trust_noise": 0.05,
            }
        ],
        "episodes": 20,
        "max_steps": 60,
        "model_episode": 1000,
        "actor_model_prefix": "models/weights/actor_agent_{i}_ep_{episode}.pth",
        "output_csv": "logs/eval_metrics.csv",
        "detection_threshold": 0.5,
        "roc_data_path_template": "logs/eval_node_features_{scenario}.npz",
        "roc_data_path": "logs/eval_node_features.npz",
        "baseline_metrics": {
            "normal": {
                "aodv": {
                    "avg_pdr": 72.3,
                    "avg_delay_ms": 85.0,
                    "avg_detection": 61.5,
                },
                "standard_marl": {
                    "avg_pdr": 84.7,
                    "avg_delay_ms": 62.0,
                    "avg_detection": 78.2,
                },
            },
            "malicious_attack": {
                "aodv": {
                    "avg_pdr": 48.5,
                    "avg_delay_ms": 142.0,
                    "avg_detection": 52.3,
                },
                "standard_marl": {
                    "avg_pdr": 71.2,
                    "avg_delay_ms": 98.0,
                    "avg_detection": 74.8,
                },
            },
        },
    },
}


def load_config(path="config.yaml"):
    env_path = os.environ.get("MARL_CONFIG_PATH")
    if env_path:
        path = env_path

    if os.path.exists(path) and yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if os.path.exists(path) and yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install it with `pip install PyYAML`."
        )

    return DEFAULT_CONFIG
