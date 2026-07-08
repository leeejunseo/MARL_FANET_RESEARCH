"""Extract per-episode Q diagnostics from trained checkpoints (no retraining)."""

import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.maddpg import MADDPGAgent
from agents.matd3 import MATD3Agent
from fanet_wrapper.fanet_env import AdvancedFANETEnv
from utils.algorithm import normalize_algorithm, display_algorithm
from utils.config import load_config
from utils.metrics_logger import QMetricsLogger


def infer_actor_obs_dim_from_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    for key, value in state_dict.items():
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.shape[1]
    raise ValueError(f"Cannot infer actor obs_dim from checkpoint: {path}")


def build_env(env_cfg, scenario):
    return AdvancedFANETEnv(
        num_drones=env_cfg["num_drones"],
        R_c=env_cfg["R_c"],
        d_safe=env_cfg["d_safe"],
        max_pos=env_cfg["max_pos"],
        max_vel=env_cfg["max_vel"],
        malicious_ratio=scenario.get("malicious_ratio", env_cfg.get("malicious_ratio", 0.0)),
        malicious_drop_rate=scenario.get("malicious_drop_rate", env_cfg.get("malicious_drop_rate", 0.4)),
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


def latest_episode_from_dir(model_dir):
    pattern = re.compile(r"actor_agent_0_ep_(\d+)\.pth$")
    best_ep = None
    best_mtime = -1.0
    for name in os.listdir(model_dir):
        m = pattern.match(name)
        if not m:
            continue
        ep = int(m.group(1))
        mtime = os.path.getmtime(os.path.join(model_dir, name))
        if mtime > best_mtime:
            best_mtime = mtime
            best_ep = ep
    if best_ep is None:
        raise FileNotFoundError(f"No actor checkpoint found in {model_dir}")
    return best_ep


def load_agents(env, cfg, algorithm, model_episode):
    training_cfg = cfg["training"]
    num_drones = env.num_drones
    use_marl = training_cfg.get("ablation", {}).get("use_marl", True)
    matd3_cfg = training_cfg.get("matd3", {})

    model_dir = training_cfg.get("model_dir", "models/weights")
    if algorithm == "matd3":
        model_dir = os.path.join(model_dir, "matd3")

    agents = []
    for i in range(num_drones):
        actor_path = os.path.join(model_dir, f"actor_agent_{i}_ep_{model_episode}.pth")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
        actor_obs_dim = infer_actor_obs_dim_from_checkpoint(actor_path)

        if algorithm == "matd3":
            agent = MATD3Agent(
                env.obs_dim,
                env.state_dim,
                env.action_dim,
                num_drones,
                agent_id=i,
                actor_lr=matd3_cfg.get("actor_lr", training_cfg.get("lr", 1e-3)),
                critic_lr=matd3_cfg.get("critic_lr", training_cfg.get("lr", 1e-3)),
                use_marl=use_marl,
                actor_obs_dim=actor_obs_dim,
                policy_delay=matd3_cfg.get("policy_delay", 2),
                target_policy_noise=matd3_cfg.get("target_policy_noise", 0.2),
                target_noise_clip=matd3_cfg.get("target_noise_clip", 0.5),
                explore_noise_std=matd3_cfg.get("explore_noise_std", 0.1),
            )
            critic1_path = os.path.join(model_dir, f"critic1_agent_{i}_ep_{model_episode}.pth")
            critic2_path = os.path.join(model_dir, f"critic2_agent_{i}_ep_{model_episode}.pth")
            if not os.path.exists(critic1_path) or not os.path.exists(critic2_path):
                raise FileNotFoundError(f"MATD3 critic checkpoint missing for agent {i} ep {model_episode}")
            agent.critic1.load_state_dict(torch.load(critic1_path, map_location="cpu"))
            agent.critic2.load_state_dict(torch.load(critic2_path, map_location="cpu"))
            agent.critic1_target.load_state_dict(agent.critic1.state_dict())
            agent.critic2_target.load_state_dict(agent.critic2.state_dict())
        else:
            agent = MADDPGAgent(
                env.obs_dim,
                env.state_dim,
                env.action_dim,
                num_drones,
                agent_id=i,
                use_marl=use_marl,
                actor_obs_dim=actor_obs_dim,
            )
            critic_path = os.path.join(model_dir, f"critic_agent_{i}_ep_{model_episode}.pth")
            if not os.path.exists(critic_path):
                raise FileNotFoundError(f"MADDPG critic checkpoint missing for agent {i} ep {model_episode}")
            agent.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
            agent.critic_target.load_state_dict(agent.critic.state_dict())

        agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        agent.actor_target.load_state_dict(agent.actor.state_dict())
        agent.actor.eval()
        agent.actor_target.eval()
        if algorithm == "matd3":
            agent.critic1.eval()
            agent.critic2.eval()
            agent.critic1_target.eval()
            agent.critic2_target.eval()
        else:
            agent.critic.eval()
            agent.critic_target.eval()
        agents.append(agent)
    return agents


def compute_step_stats(algorithm, agents, obs, state, actions, rewards, next_obs, next_state, done, gamma):
    num_drones = len(agents)

    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    states_t = torch.FloatTensor(state).unsqueeze(0)
    actions_t = torch.FloatTensor(actions).unsqueeze(0)
    rewards_t = torch.FloatTensor(rewards)
    next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
    next_states_t = torch.FloatTensor(next_state).unsqueeze(0)
    done_t = torch.tensor([[float(done)]], dtype=torch.float32)

    results = []
    with torch.no_grad():
        for i, agent in enumerate(agents):
            if agent.use_marl:
                if algorithm == "matd3":
                    next_actions = []
                    for j, other in enumerate(agents):
                        next_actions.append(other._target_action(other, next_obs_t[:, j]))
                    next_actions_t = torch.cat(next_actions, dim=-1)
                    target_q1 = agent.critic1_target(next_states_t, next_actions_t)
                    target_q2 = agent.critic2_target(next_states_t, next_actions_t)
                    target_q = torch.min(target_q1, target_q2)
                    y = rewards_t[i].view(1, 1) + (1.0 - done_t) * gamma * target_q

                    current_actions_flat = actions_t.view(1, -1)
                    current_q1 = agent.critic1(states_t, current_actions_flat)
                    current_q2 = agent.critic2(states_t, current_actions_flat)
                    current_q = torch.min(current_q1, current_q2)

                    results.append(
                        {
                            "q_current_mean": float(current_q.item()),
                            "q_target_mean": float(y.item()),
                            "q_overestimation_gap": float((current_q - y).item()),
                            "q_abs_td_error": float(torch.abs(current_q - y).item()),
                            "q1_current_mean": float(current_q1.item()),
                            "q2_current_mean": float(current_q2.item()),
                            "q_disagreement_mean": float(torch.abs(current_q1 - current_q2).item()),
                        }
                    )
                else:
                    next_actions = []
                    for j, other in enumerate(agents):
                        next_actions.append(other.actor_target(other._prepare_actor_obs(next_obs_t[:, j])))
                    next_actions_t = torch.cat(next_actions, dim=-1)
                    target_q = agent.critic_target(next_states_t, next_actions_t)
                    y = rewards_t[i].view(1, 1) + (1.0 - done_t) * gamma * target_q

                    current_actions_flat = actions_t.view(1, -1)
                    current_q = agent.critic(states_t, current_actions_flat)
                    results.append(
                        {
                            "q_current_mean": float(current_q.item()),
                            "q_target_mean": float(y.item()),
                            "q_overestimation_gap": float((current_q - y).item()),
                            "q_abs_td_error": float(torch.abs(current_q - y).item()),
                            "q1_current_mean": float(current_q.item()),
                            "q2_current_mean": None,
                            "q_disagreement_mean": None,
                        }
                    )
            else:
                raise NotImplementedError("extract_q_from_trained currently expects use_marl=True")
    return results


def aggregate_stats(stats):
    if not stats:
        return {}

    keys = [
        "q_current_mean",
        "q_target_mean",
        "q_overestimation_gap",
        "q_abs_td_error",
        "q1_current_mean",
        "q2_current_mean",
        "q_disagreement_mean",
    ]
    out = {}
    for k in keys:
        vals = [s[k] for s in stats if s.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Q statistics from trained checkpoints")
    parser.add_argument("--algorithm", required=True, choices=["maddpg", "matd3"])
    parser.add_argument("--episode", type=int, default=None, help="Checkpoint episode; default uses latest by modified time")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    env_cfg = cfg["environment"]
    eval_cfg = cfg["evaluation"]
    training_cfg = cfg["training"]
    algorithm = normalize_algorithm(args.algorithm)
    gamma = float(training_cfg.get("gamma", 0.95))

    model_dir = training_cfg.get("model_dir", "models/weights")
    if algorithm == "matd3":
        model_dir = os.path.join(model_dir, "matd3")

    model_episode = args.episode if args.episode is not None else latest_episode_from_dir(model_dir)

    q_log_path_base = training_cfg.get("q_log_path")
    if not q_log_path_base:
        root, ext = os.path.splitext(training_cfg.get("log_path", "logs/training_rewards.csv"))
        q_log_path_base = f"{root}_q_values{ext}"
    if algorithm == "maddpg":
        q_log_path = q_log_path_base
    else:
        root, ext = os.path.splitext(q_log_path_base)
        q_log_path = f"{root}_{algorithm}{ext}"

    logger = QMetricsLogger(q_log_path)

    scenarios = eval_cfg.get("scenarios") or [
        {
            "name": "Default",
            "malicious_ratio": env_cfg.get("malicious_ratio", 0.0),
            "malicious_behavior": env_cfg.get("malicious_behavior", "drop_and_trust"),
            "malicious_drop_rate": env_cfg.get("malicious_drop_rate", 0.4),
            "trust_noise": env_cfg.get("trust_noise", 0.05),
        }
    ]

    max_steps = args.max_steps if args.max_steps is not None else int(eval_cfg.get("max_steps", 60))

    all_episode_stats = []
    for ep in range(1, args.eval_episodes + 1):
        step_stats_all = []

        for scenario in scenarios:
            env = build_env(env_cfg, scenario)
            agents = load_agents(env, cfg, algorithm, model_episode)

            obs, state = env.reset(seed=args.seed + ep)
            for _ in range(max_steps):
                actions = np.array([agents[i].act(obs[i], explore=False) for i in range(env.num_drones)])
                next_obs, next_state, rewards, done, _ = env.step(actions)

                step_agent_stats = compute_step_stats(
                    algorithm,
                    agents,
                    obs,
                    state,
                    actions,
                    rewards,
                    next_obs,
                    next_state,
                    done,
                    gamma,
                )
                step_stats_all.extend(step_agent_stats)

                obs = next_obs
                state = next_state
                if done:
                    break

        ep_stats = aggregate_stats(step_stats_all)
        all_episode_stats.append(ep_stats)
        logger.log_episode(
            ep,
            display_algorithm(algorithm),
            q_current_mean=ep_stats.get("q_current_mean"),
            q_target_mean=ep_stats.get("q_target_mean"),
            q_overestimation_gap=ep_stats.get("q_overestimation_gap"),
            q_abs_td_error=ep_stats.get("q_abs_td_error"),
            q1_current_mean=ep_stats.get("q1_current_mean"),
            q2_current_mean=ep_stats.get("q2_current_mean"),
            q_disagreement_mean=ep_stats.get("q_disagreement_mean"),
            critic_loss1_mean=None,
            critic_loss2_mean=None,
            actor_loss_mean=None,
            actor_update_ratio=None,
        )

        gap = ep_stats.get("q_overestimation_gap")
        gap_text = f"{gap:+.6f}" if gap is not None else "N/A"
        print(f"[{display_algorithm(algorithm)}] episode={ep:02d} q_gap={gap_text}")

    total = aggregate_stats(all_episode_stats)
    print("=== Q extraction done ===")
    print(f"algorithm={display_algorithm(algorithm)} | model_episode={model_episode}")
    print(f"q_log_path={q_log_path}")
    print(f"mean_q_overestimation_gap={total.get('q_overestimation_gap')}")
    print(f"mean_q_abs_td_error={total.get('q_abs_td_error')}")


if __name__ == "__main__":
    main()
