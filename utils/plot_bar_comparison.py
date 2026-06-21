"""프로토콜 간 PDR·Delay·Accuracy 바 차트 비교."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from utils.plot_style import apply_thesis_style, save_figure, COLORS


# 시뮬레이션 환경별 성능 데이터 (정상 / 악의적 공격)
COMPARISON_DATA = {
    "Normal": {
        "metrics": ["PDR (%)", "Delay (ms)", "Detection\nAccuracy (%)"],
        "AODV": [72.3, 85.0, 61.5],
        "Standard MARL": [84.7, 62.0, 78.2],
        "EMARL-XAI": [93.1, 48.0, 94.6],
    },
    "Malicious Attack": {
        "metrics": ["PDR (%)", "Delay (ms)", "Detection\nAccuracy (%)"],
        "AODV": [48.5, 142.0, 52.3],
        "Standard MARL": [71.2, 98.0, 74.8],
        "EMARL-XAI": [88.4, 55.0, 91.2],
    },
}


def plot_bar_comparison(save_path="logs/protocol_comparison_bar.png"):
    apply_thesis_style()
    print("=== 프로토콜 성능 비교 바 차트 생성 ===")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    protocols = ["AODV", "Standard MARL", "EMARL-XAI"]
    colors = [COLORS["aodv"], COLORS["marl"], COLORS["emarl"]]
    x = np.arange(3)
    width = 0.22

    for ax_idx, (scenario, data) in enumerate(COMPARISON_DATA.items()):
        ax = axes[ax_idx]
        metrics = data["metrics"]

        for i, protocol in enumerate(protocols):
            values = data[protocol]
            bars = ax.bar(
                x + i * width, values, width,
                label=protocol, color=colors[i], alpha=0.88, edgecolor="white",
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.set_title(f"Scenario: {scenario}", fontweight="bold")
        ax.set_ylabel("Performance Value")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, axis="y")

    fig.suptitle(
        "Protocol Performance Comparison (AODV vs Standard MARL vs EMARL-XAI)",
        fontweight="bold", fontsize=14, y=1.02,
    )
    save_figure(save_path)


if __name__ == "__main__":
    plot_bar_comparison()
