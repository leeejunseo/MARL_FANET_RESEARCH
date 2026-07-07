"""Run quick MATD3 hyperparameter sweep focused on detection F1 trade-off."""

import argparse
import copy
import csv
import os
import subprocess
import sys
from datetime import datetime

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Quick MATD3 F1-focused sweep")
    parser.add_argument("--baseline-dir", default="logs/experiments/maddpg_baseline_maddpg")
    parser.add_argument("--output-root", default="logs/sweeps")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=60)
    return parser.parse_args()


def run_cmd(args, env=None):
    print("[RUN]", " ".join(args))
    proc = subprocess.run(args, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(args)}")


def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_detection_f1(summary_csv):
    vals = []
    with open(summary_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("delta_avg_detection_f1") or "").strip()
            if text:
                vals.append(float(text))
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def read_auc_delta(auc_txt):
    if not os.path.exists(auc_txt):
        return None
    with open(auc_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("delta_auc="):
                return float(line.split("=", 1)[1])
    return None


def main():
    args = parse_args()
    root = os.getcwd()

    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.output_root, f"matd3_f1_sweep_{ts}")
    os.makedirs(sweep_dir, exist_ok=True)

    variants = [
        {
            "name": "matd3_f1_v1",
            "matd3": {
                "policy_delay": 3,
                "target_policy_noise": 0.15,
                "target_noise_clip": 0.35,
                "explore_noise_std": 0.08,
            },
            "detection_reward": {"tp": 0.7, "fp": -0.15, "fn": -0.8, "tn": 0.0},
            "detection_threshold": 0.45,
        },
        {
            "name": "matd3_f1_v2",
            "matd3": {
                "policy_delay": 3,
                "target_policy_noise": 0.12,
                "target_noise_clip": 0.30,
                "explore_noise_std": 0.07,
            },
            "detection_reward": {"tp": 0.8, "fp": -0.1, "fn": -1.0, "tn": 0.0},
            "detection_threshold": 0.42,
        },
        {
            "name": "matd3_perf_v1",
            "matd3": {
                "policy_delay": 2,
                "target_policy_noise": 0.10,
                "target_noise_clip": 0.25,
                "explore_noise_std": 0.06,
                "actor_lr": 0.0007,
                "critic_lr": 0.0010,
            },
            "detection_reward": {"tp": 0.45, "fp": -0.15, "fn": -0.55, "tn": 0.0},
            "detection_threshold": 0.50,
            "env_override": {
                "reward_conn_coeff": 6.0,
                "reward_w_delay": 1.0,
                "reward_w_security": 1.2,
                "connectivity_guard_coeff": 0.50,
                "malicious_avoid_coeff": 0.45,
                "suspicious_avoid_coeff": 0.20,
            },
        },
        {
            "name": "matd3_perf_v2",
            "matd3": {
                "policy_delay": 2,
                "target_policy_noise": 0.08,
                "target_noise_clip": 0.20,
                "explore_noise_std": 0.05,
                "actor_lr": 0.0005,
                "critic_lr": 0.0008,
            },
            "detection_reward": {"tp": 0.40, "fp": -0.12, "fn": -0.50, "tn": 0.0},
            "detection_threshold": 0.52,
            "env_override": {
                "reward_conn_coeff": 6.4,
                "reward_w_delay": 0.9,
                "reward_w_security": 1.1,
                "connectivity_guard_coeff": 0.55,
                "malicious_avoid_coeff": 0.40,
                "suspicious_avoid_coeff": 0.18,
            },
        },
    ]

    result_rows = []

    for var in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["algorithm"] = "matd3"
        cfg["training"]["max_episodes"] = args.episodes
        cfg["training"]["max_steps"] = args.max_steps
        cfg["training"]["save_interval"] = max(20, args.episodes // 3)
        cfg["training"]["warmup_steps"] = min(cfg["training"].get("warmup_steps", 500), 300)
        cfg["training"]["detection_threshold"] = var["detection_threshold"]
        cfg["training"]["detection_reward"] = var["detection_reward"]
        cfg["training"]["matd3"].update(var["matd3"])
        if var.get("env_override"):
            cfg["environment"].update(var["env_override"])

        cfg["evaluation"]["model_episode"] = args.episodes
        cfg["evaluation"]["episodes"] = args.eval_episodes
        cfg["evaluation"]["max_steps"] = args.max_steps
        cfg["evaluation"]["overwrite_csv"] = True

        config_path = os.path.join(sweep_dir, f"{var['name']}.yaml")
        write_yaml(config_path, cfg)

        env = os.environ.copy()
        env["MARL_CONFIG_PATH"] = config_path

        run_cmd([sys.executable, "train.py"], env=env)
        run_cmd([sys.executable, "eval.py"], env=env)
        run_cmd([sys.executable, "test.py"], env=env)

        archive_tag = f"candidate_{var['name']}"
        run_cmd(
            [
                sys.executable,
                "utils/archive_experiment.py",
                "--algorithm",
                "matd3",
                "--run-tag",
                archive_tag,
                "--include-config",
            ],
            env=env,
        )

        candidate_dir = os.path.join("logs", "experiments", f"matd3_{archive_tag}")
        prefix = f"compare_maddpg_vs_{var['name']}"
        run_cmd(
            [
                sys.executable,
                "utils/compare_algorithm_results.py",
                "--baseline-dir",
                args.baseline_dir,
                "--candidate-dir",
                candidate_dir,
                "--baseline-label",
                "MADDPG",
                "--candidate-label",
                "MATD3",
                "--output-dir",
                "logs",
                "--prefix",
                prefix,
            ],
            env=env,
        )

        summary_csv = os.path.join("logs", f"{prefix}_summary.csv")
        auc_txt = os.path.join("logs", f"{prefix}_summary_auc.txt")
        mean_f1_delta = read_detection_f1(summary_csv)
        auc_delta = read_auc_delta(auc_txt)
        result_rows.append(
            {
                "variant": var["name"],
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "mean_delta_detection_f1": mean_f1_delta,
                "delta_auc": auc_delta,
                "summary_csv": summary_csv,
                "auc_txt": auc_txt,
                "candidate_dir": candidate_dir,
            }
        )

    result_csv = os.path.join(sweep_dir, "sweep_results.csv")
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "episodes",
                "eval_episodes",
                "mean_delta_detection_f1",
                "delta_auc",
                "summary_csv",
                "auc_txt",
                "candidate_dir",
            ],
        )
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    print("=== MATD3 F1 Sweep Completed ===")
    print(f"Sweep root: {sweep_dir}")
    print(f"Result table: {result_csv}")


if __name__ == "__main__":
    main()
