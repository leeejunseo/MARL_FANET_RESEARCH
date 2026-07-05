import argparse
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from agents.maddpg import MADDPGAgent
from analysis.malicious_detector import MaliciousNodeDetector
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from ns3_wrapper.provider_factory import build_link_provider
from utils.config import load_config


def infer_actor_obs_dim_from_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    for key, value in state_dict.items():
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.shape[1]
    raise ValueError(f"Cannot infer actor obs_dim from checkpoint: {path}")


def load_agents(env, config, episode):
    num_drones = config["environment"]["num_drones"]
    prefix = config["evaluation"]["actor_model_prefix"]
    use_marl = config.get("training", {}).get("ablation", {}).get("use_marl", True)

    agents = []
    for i in range(num_drones):
        path = prefix.format(i=i, episode=episode)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Actor model not found: {path}")

        actor_obs_dim = infer_actor_obs_dim_from_checkpoint(path)
        agent = MADDPGAgent(
            env.obs_dim,
            env.state_dim,
            env.action_dim,
            num_drones,
            agent_id=i,
            use_marl=use_marl,
            actor_obs_dim=actor_obs_dim,
        )
        agent.actor.load_state_dict(torch.load(path, map_location="cpu"))
        agents.append(agent)

    return agents


def build_env(config, scenario_name):
    env_cfg = config["environment"]
    link_provider = build_link_provider(config, env_cfg["num_drones"])
    scenarios = config.get("evaluation", {}).get("scenarios", [])
    scenario = None

    for s in scenarios:
        if s.get("name", "").lower().replace(" ", "_") == scenario_name.lower().replace(" ", "_"):
            scenario = s
            break

    if scenario is None:
        scenario = {
            "name": "Default",
            "malicious_ratio": env_cfg.get("malicious_ratio", 0.0),
            "malicious_behavior": env_cfg.get("malicious_behavior", "drop_and_trust"),
            "malicious_drop_rate": env_cfg.get("malicious_drop_rate", 0.4),
            "trust_noise": env_cfg.get("trust_noise", 0.05),
        }

    env = AdvancedFANETEnv(
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
        alert_decay=env_cfg.get("alert_decay", 0.9),
        link_provider=link_provider,
    )
    return env, scenario


def rollout_trace(env, agents, max_steps, seed, policy, detector=None, scenario_label="Default", frame_offset=0):
    obs, _ = env.reset(seed=seed)
    positions = []
    connected = []
    attacked_edges = []
    pdrs = []
    delays = []
    disconnects = []
    trusts = []
    attack_confidence = []
    events = []
    prev_conn = None

    for local_step in range(max_steps):
        if policy == "random":
            actions = np.random.uniform(-1.0, 1.0, size=(env.num_drones, env.action_dim))
        else:
            actions = np.array([agent.act(obs[i], explore=False) for i, agent in enumerate(agents)])

        next_obs, _, _, terminated, info = env.step(actions)
        pos = env.positions.copy()
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        conn = d <= env.R_c

        malicious = set(info.get("malicious_ids", []))
        edge_attack = np.zeros_like(conn, dtype=bool)
        for i in range(env.num_drones):
            for j in range(env.num_drones):
                if i != j and conn[i, j] and (i in malicious or j in malicious):
                    edge_attack[i, j] = True

        if detector is not None and info.get("node_features") is not None:
            proba = detector.predict_proba(info["node_features"])
            confidence = float(np.max(proba)) if len(proba) > 0 else 0.0
        else:
            confidence = 0.0
        attack_confidence.append(confidence)

        if prev_conn is not None:
            for i in range(env.num_drones):
                for j in range(i + 1, env.num_drones):
                    if prev_conn[i, j] and not conn[i, j]:
                        reason = "distance_exceeded_rc"
                        if i in malicious or j in malicious:
                            reason += "+malicious_path"
                        events.append(
                            {
                                "step": frame_offset + local_step,
                                "scenario": scenario_label,
                                "node_i": i,
                                "node_j": j,
                                "event_type": "disconnect",
                                "reason": reason,
                            }
                        )
                    elif (not prev_conn[i, j]) and conn[i, j]:
                        reason = "recovered_within_rc"
                        if i in malicious or j in malicious:
                            reason += "+malicious_path"
                        events.append(
                            {
                                "step": frame_offset + local_step,
                                "scenario": scenario_label,
                                "node_i": i,
                                "node_j": j,
                                "event_type": "reconnect",
                                "reason": reason,
                            }
                        )
        prev_conn = conn.copy()

        positions.append(pos)
        connected.append(conn)
        attacked_edges.append(edge_attack)
        pdrs.append(info.get("pdr", 0.0))
        delays.append(info.get("avg_delay_ms", 0.0))
        disconnects.append(info.get("disconnect_ratio", 0.0))
        trusts.append(np.array(info.get("trust_scores", np.zeros(env.num_drones)), dtype=float))

        obs = next_obs
        if terminated:
            break

    return {
        "positions": np.array(positions),
        "connected": np.array(connected),
        "attacked_edges": np.array(attacked_edges),
        "pdr": np.array(pdrs),
        "delay": np.array(delays),
        "disconnect": np.array(disconnects),
        "trust": np.array(trusts),
        "attack_confidence": np.array(attack_confidence),
        "malicious_ids": env.malicious_ids,
        "events": events,
    }


def merge_traces(traces):
    if not traces:
        return None

    merged = {
        "positions": np.concatenate([t["positions"] for t in traces], axis=0),
        "connected": np.concatenate([t["connected"] for t in traces], axis=0),
        "attacked_edges": np.concatenate([t["attacked_edges"] for t in traces], axis=0),
        "pdr": np.concatenate([t["pdr"] for t in traces], axis=0),
        "delay": np.concatenate([t["delay"] for t in traces], axis=0),
        "disconnect": np.concatenate([t["disconnect"] for t in traces], axis=0),
        "trust": np.concatenate([t["trust"] for t in traces], axis=0),
        "attack_confidence": np.concatenate([t["attack_confidence"] for t in traces], axis=0),
        "malicious_ids": traces[-1]["malicious_ids"],
        "events": [event for t in traces for event in t.get("events", [])],
    }
    return merged


def save_link_events(events, out_path):
    if out_path is None:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "scenario", "node_i", "node_j", "event_type", "reason"],
        )
        writer.writeheader()
        for row in events:
            writer.writerow(row)
    print(f"Saved link events to: {out_path}")


def build_demo_trace(config, policy, seed, fps, total_seconds, episode=None):
    scenarios = config.get("evaluation", {}).get("scenarios", [])
    if not scenarios:
        scenarios = [{"name": "Default"}]

    frames_total = max(1, int(total_seconds * fps))
    frames_per_scenario = max(1, frames_total // len(scenarios))
    detector = MaliciousNodeDetector()
    traces = []
    scenario_labels = []
    frame_offset = 0
    demo_env = None

    for scenario in scenarios:
        env, _ = build_env(config, scenario["name"])
        if demo_env is None:
            demo_env = env
        if policy == "trained":
            episode_to_load = episode if episode is not None else config["evaluation"]["model_episode"]
            agents = load_agents(env, config, episode_to_load)
        else:
            agents = None

        trace = rollout_trace(
            env=env,
            agents=agents,
            max_steps=frames_per_scenario,
            seed=seed,
            policy=policy,
            detector=detector,
            scenario_label=scenario["name"],
            frame_offset=frame_offset,
        )
        traces.append(trace)
        scenario_labels.extend([scenario["name"]] * len(trace["positions"]))
        frame_offset += len(trace["positions"])

    merged = merge_traces(traces)
    return merged, demo_env, scenario_labels


def animate_trace(trace, env, scenario_name, out_path=None, show=True, fps=3, scenario_labels=None):
    pos = trace["positions"]
    conn = trace["connected"]
    attacked = trace["attacked_edges"]
    pdr = trace["pdr"]
    delay = trace["delay"]
    disconnect = trace["disconnect"]
    trust = trace["trust"]
    confidence = trace["attack_confidence"]
    malicious_ids = set(trace["malicious_ids"])

    if len(pos) == 0:
        raise RuntimeError("No frames to animate.")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    ax = axes[0]
    ax_conf = axes[1]

    ax.set_xlim(0, env.max_pos)
    ax.set_ylim(0, env.max_pos)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax_conf.set_xlim(0, max(1, len(pos) - 1))
    ax_conf.set_ylim(0.0, 1.0)
    ax_conf.grid(alpha=0.25)
    ax_conf.set_xlabel("Frame")
    ax_conf.set_ylabel("Attack confidence")
    ax_conf.axhline(0.5, color="#b03a2e", linewidth=1.0, linestyle="--", alpha=0.8)
    conf_line, = ax_conf.plot([], [], color="#2c3e50", linewidth=2.0)

    info_box = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    line_artists = []
    label_artists = []
    scatter_artist = None

    def clear_lines():
        for artist in line_artists:
            artist.remove()
        line_artists.clear()

    def clear_labels():
        for artist in label_artists:
            artist.remove()
        label_artists.clear()

    def update(frame_idx):
        nonlocal scatter_artist
        clear_lines()
        clear_labels()
        if scatter_artist is not None:
            scatter_artist.remove()
            scatter_artist = None

        xy = pos[frame_idx][:, :2]
        this_conn = conn[frame_idx]
        this_attacked = attacked[frame_idx]
        prev_conn = conn[frame_idx - 1] if frame_idx > 0 else np.zeros_like(this_conn)

        # Draw active links first.
        for i in range(env.num_drones):
            for j in range(i + 1, env.num_drones):
                if this_conn[i, j]:
                    x = [xy[i, 0], xy[j, 0]]
                    y = [xy[i, 1], xy[j, 1]]
                    if this_attacked[i, j] or this_attacked[j, i]:
                        line, = ax.plot(x, y, color="#ff7f0e", linewidth=2.5, alpha=0.9)
                    else:
                        line, = ax.plot(x, y, color="#2ca02c", linewidth=1.8, alpha=0.65)
                    line_artists.append(line)
                elif prev_conn[i, j] and not this_conn[i, j]:
                    # Flash dropped links for one frame.
                    x = [xy[i, 0], xy[j, 0]]
                    y = [xy[i, 1], xy[j, 1]]
                    line, = ax.plot(x, y, color="#d62728", linewidth=1.5, alpha=0.75, linestyle="--")
                    line_artists.append(line)

        colors = []
        edge_colors = []
        for i in range(env.num_drones):
            neighbors = np.sum(this_conn[i]) - 1
            isolated = neighbors <= 0
            if i in malicious_ids:
                colors.append("#d62728")
                edge_colors.append("black")
            elif isolated:
                colors.append("#f1c40f")
                edge_colors.append("#7f8c8d")
            else:
                colors.append("#1f77b4")
                edge_colors.append("#34495e")

        scatter_artist = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=240,
            c=colors,
            edgecolors=edge_colors,
            linewidths=1.8,
            zorder=3,
        )

        for i in range(env.num_drones):
            trust_text = trust[frame_idx][i] if trust.shape[0] > frame_idx else 0.0
            marker = " (M)" if i in malicious_ids else ""
            text_artist = ax.text(
                xy[i, 0] + 12,
                xy[i, 1] + 12,
                f"UAV {i}{marker}\\nT:{trust_text:.2f}",
                fontsize=9,
                color="#1b1b1b",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
                zorder=4,
            )
            label_artists.append(text_artist)

        info_box.set_text(
            "\\n".join(
                [
                    f"Scenario: {scenario_labels[frame_idx] if scenario_labels is not None else scenario_name}",
                    f"Step: {frame_idx + 1}/{len(pos)}",
                    f"PDR: {pdr[frame_idx]:.2f}",
                    f"Avg Delay: {delay[frame_idx]:.1f} ms",
                    f"Disconnect Ratio: {disconnect[frame_idx]:.2f}",
                    f"Attack confidence: {confidence[frame_idx]:.2f}",
                    "Orange links: malicious-path communication",
                    "Red dashed: recently dropped links",
                ]
            )
        )
        ax.set_title("FANET Link & Attack Visualization (Top-down)")

        x_conf = np.arange(frame_idx + 1)
        y_conf = confidence[: frame_idx + 1]
        conf_line.set_data(x_conf, y_conf)
        return []

    interval_ms = max(1, int(1000 / max(1, fps)))
    anim = FuncAnimation(fig, update, frames=len(pos), interval=interval_ms, blit=False, repeat=True)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ext = os.path.splitext(out_path)[1].lower()
        saved = False

        if ext == ".gif":
            try:
                from matplotlib.animation import PillowWriter

                anim.save(out_path, writer=PillowWriter(fps=fps))
                print(f"Saved animation to: {out_path}")
                saved = True
            except Exception as exc:
                print(f"GIF save failed ({exc}). Trying ffmpeg...")

        if not saved:
            try:
                anim.save(out_path, writer="ffmpeg", fps=fps)
                print(f"Saved animation to: {out_path}")
                saved = True
            except Exception as exc:
                print(f"Video save skipped ({exc}). Use --show to view interactively.")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="2D FANET communication/attack visualization.")
    parser.add_argument("--policy", choices=["trained", "random"], default="trained")
    parser.add_argument("--scenario", default="Default", help="Default, Blackhole, Selective_Forwarding, Sybil")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=None, help="Model episode to load")
    parser.add_argument("--fps", type=int, default=3, help="Animation FPS")
    parser.add_argument("--demo-60", action="store_true", help="Run 60-second autoplay demo across scenarios")
    parser.add_argument("--demo-seconds", type=int, default=60, help="Total seconds for autoplay demo")
    parser.add_argument("--output", type=str, default=None, help="Optional output path (.gif or .mp4)")
    parser.add_argument("--event-log", type=str, default=None, help="Optional link event CSV output path")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib window")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    detector = MaliciousNodeDetector()

    if args.demo_60:
        trace, env, scenario_labels = build_demo_trace(
            config=config,
            policy=args.policy,
            seed=args.seed,
            fps=args.fps,
            total_seconds=args.demo_seconds,
            episode=args.episode,
        )
        if args.output is None:
            args.output = "logs/demo_60s.gif"
        if args.event_log is None:
            args.event_log = "logs/demo_60s_link_events.csv"
        scenario_name = "Autoplay 60s"
    else:
        env, scenario = build_env(config, args.scenario)
        max_steps = args.max_steps if args.max_steps is not None else config["evaluation"]["max_steps"]

        if args.policy == "trained":
            episode = args.episode if args.episode is not None else config["evaluation"]["model_episode"]
            agents = load_agents(env, config, episode)
        else:
            agents = None

        trace = rollout_trace(
            env=env,
            agents=agents,
            max_steps=max_steps,
            seed=args.seed,
            policy=args.policy,
            detector=detector,
            scenario_label=scenario["name"],
            frame_offset=0,
        )
        scenario_labels = None
        scenario_name = f"{scenario['name']} ({scenario['malicious_behavior']})"

    if args.event_log is not None:
        save_link_events(trace.get("events", []), args.event_log)

    animate_trace(
        trace,
        env,
        scenario_name=scenario_name,
        out_path=args.output,
        show=not args.no_show,
        fps=args.fps,
        scenario_labels=scenario_labels,
    )


if __name__ == "__main__":
    main()
