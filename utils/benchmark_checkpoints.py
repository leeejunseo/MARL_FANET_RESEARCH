import argparse
import csv
import glob
import os
from copy import deepcopy

import torch

from agents.maddpg import MADDPGAgent
from analysis.malicious_detector import MaliciousNodeDetector
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from ns3_wrapper.provider_factory import build_link_provider
from test import evaluate_policy_episode, infer_actor_obs_dim_from_checkpoint
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark checkpoints and select best model episode.")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--bridge", choices=["off", "on"], default="off")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=str, default="logs/checkpoint_benchmark.csv")
    return parser.parse_args()


def parse_seeds(seed_text):
    seeds = []
    for token in seed_text.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    return seeds


def discover_episodes(prefix):
    pattern = prefix.format(i=0, episode="*")
    episodes = []
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        # actor_agent_0_ep_1000.pth -> 1000
        try:
            ep = int(base.split("_ep_")[1].split(".pth")[0])
            episodes.append(ep)
        except Exception:
            continue
    return sorted(set(episodes))


def build_env(config):
    env_cfg = config["environment"]
    provider = build_link_provider(config, env_cfg["num_drones"])
    env = AdvancedFANETEnv(
        num_drones=env_cfg["num_drones"],
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
        link_provider=provider,
    )
    return env


def load_agents_for_episode(env, actor_prefix, episode):
    agents = []
    for i in range(env.num_drones):
        path = actor_prefix.format(i=i, episode=episode)
        actor_obs_dim = infer_actor_obs_dim_from_checkpoint(path)
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


def safe_mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def main():
    args = parse_args()
    config = load_config()
    eval_cfg = config["evaluation"]
    actor_prefix = eval_cfg["actor_model_prefix"]

    cfg_local = deepcopy(config)
    cfg_local.setdefault("ns3_bridge", {})
    cfg_local["ns3_bridge"]["enabled"] = args.bridge == "on"

    seeds = parse_seeds(args.seeds)
    max_steps = args.max_steps if args.max_steps is not None else eval_cfg["max_steps"]
    episodes = discover_episodes(actor_prefix)
    if not episodes:
        raise RuntimeError("No actor checkpoints found for benchmarking.")

    env = build_env(cfg_local)
    detector = MaliciousNodeDetector()

    random_runs = [
        evaluate_policy_episode(env, "random", max_steps, detector=detector, seed=seed)
        for seed in seeds
    ]
    random_reward = safe_mean([r["total_reward"] for r in random_runs])
    random_pdr = safe_mean([r["avg_pdr"] for r in random_runs])
    random_delay = safe_mean([r["avg_delay_ms"] for r in random_runs])
    random_trust = safe_mean([r["avg_trust"] for r in random_runs])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = []
    best_row = None

    for ep in episodes:
        agents = load_agents_for_episode(env, actor_prefix, ep)
        trained_runs = [
            evaluate_policy_episode(env, agents, max_steps, detector=detector, seed=seed)
            for seed in seeds
        ]
        trained_reward = safe_mean([r["total_reward"] for r in trained_runs])
        trained_pdr = safe_mean([r["avg_pdr"] for r in trained_runs])
        trained_delay = safe_mean([r["avg_delay_ms"] for r in trained_runs])
        trained_trust = safe_mean([r["avg_trust"] for r in trained_runs])

        row = {
            "episode": ep,
            "bridge": args.bridge,
            "seeds": args.seeds,
            "trained_reward": trained_reward,
            "random_reward": random_reward,
            "reward_gain": trained_reward - random_reward,
            "trained_pdr": trained_pdr,
            "random_pdr": random_pdr,
            "pdr_gain": trained_pdr - random_pdr,
            "trained_delay_ms": trained_delay,
            "random_delay_ms": random_delay,
            "delay_gain_ms": random_delay - trained_delay,
            "trained_trust": trained_trust,
            "random_trust": random_trust,
            "trust_gain": trained_trust - random_trust,
        }
        rows.append(row)

        if best_row is None:
            best_row = row
        else:
            cur = (row["reward_gain"], row["pdr_gain"], row["delay_gain_ms"], row["trust_gain"])
            best = (best_row["reward_gain"], best_row["pdr_gain"], best_row["delay_gain_ms"], best_row["trust_gain"])
            if cur > best:
                best_row = row

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "bridge",
                "seeds",
                "trained_reward",
                "random_reward",
                "reward_gain",
                "trained_pdr",
                "random_pdr",
                "pdr_gain",
                "trained_delay_ms",
                "random_delay_ms",
                "delay_gain_ms",
                "trained_trust",
                "random_trust",
                "trust_gain",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark: {args.output}")
    print(f"Best episode ({args.bridge}): {best_row['episode']}")
    print(
        "reward_gain={:.2f}, pdr_gain={:.4f}, delay_gain_ms={:.2f}, trust_gain={:.4f}".format(
            best_row["reward_gain"],
            best_row["pdr_gain"],
            best_row["delay_gain_ms"],
            best_row["trust_gain"],
        )
    )


if __name__ == "__main__":
    main()
