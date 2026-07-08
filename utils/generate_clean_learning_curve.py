"""Generate a clean/fair learning-curve comparison figure.

This script avoids mixed-run artifacts by:
1) selecting only the last contiguous run from each training log,
2) showing both full last-run view and common-episode fair view.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"episode": int(r["episode"]), "reward": float(r["total_reward"])})
    return rows


def last_contiguous_run(rows):
    if not rows:
        return []
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["episode"] < rows[i - 1]["episode"]:
            start = i
    return rows[start:]


def dedup_mean_by_episode(rows):
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode"]].append(r["reward"])
    episodes = sorted(by_ep.keys())
    rewards = np.array([float(np.mean(by_ep[ep])) for ep in episodes], dtype=np.float64)
    return np.array(episodes, dtype=np.int32), rewards


def moving_average(values, window=20):
    if values is None or len(values) < window:
        return None
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_curve(ax, x, y, label, color):
    ax.plot(x, y, color=color, alpha=0.25, linewidth=1.0, label=f"{label} raw")
    ma = moving_average(y, window=20)
    if ma is not None:
        ax.plot(x[19:], ma, color=color, linewidth=2.2, label=f"{label} MA(20)")


def main():
    maddpg_csv = "logs/training_rewards.csv"
    matd3_csv = "logs/training_rewards_matd3.csv"
    out_path = "logs/compare_maddpg_vs_matd3_clean_learning_curve.png"

    m_rows = dedup_mean_by_episode(last_contiguous_run(load_rows(maddpg_csv)))
    t_rows = dedup_mean_by_episode(last_contiguous_run(load_rows(matd3_csv)))

    m_ep, m_rw = m_rows
    t_ep, t_rw = t_rows

    common_max = int(min(m_ep.max(), t_ep.max()))
    m_mask = m_ep <= common_max
    t_mask = t_ep <= common_max

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: last run full lengths
    plot_curve(axes[0], m_ep, m_rw, "MADDPG", "#1f77b4")
    plot_curve(axes[0], t_ep, t_rw, "MATD3", "#d62728")
    axes[0].set_title("Last Contiguous Run (Full Length)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best", fontsize=9)

    # Right: fair common range only
    plot_curve(axes[1], m_ep[m_mask], m_rw[m_mask], "MADDPG", "#1f77b4")
    plot_curve(axes[1], t_ep[t_mask], t_rw[t_mask], "MATD3", "#d62728")
    axes[1].set_title(f"Fair View (Common Episodes: 1-{common_max})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Reward")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="best", fontsize=9)

    note = (
        f"Data policy: last contiguous run only | "
        f"MADDPG max ep={int(m_ep.max())}, MATD3 max ep={int(t_ep.max())}, common={common_max}"
    )
    fig.suptitle("MADDPG vs MATD3 Training Reward (Clean Comparison)", fontsize=13, y=1.02)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("clean_curve_saved=" + out_path)


if __name__ == "__main__":
    main()
