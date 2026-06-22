"""프로토콜 간 PDR·Delay·Accuracy 바 차트 비교."""

import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from utils.config import load_config
from utils.plot_style import apply_thesis_style, save_figure, COLORS


def load_eval_metrics(filepath="logs/eval_metrics.csv"):
    if not os.path.exists(filepath):
        return None
    values = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append({
                "scenario": row.get("scenario", "Default"),
                "avg_pdr": float(row["avg_pdr"]),
                "avg_delay_ms": float(row["avg_delay_ms"]),
                "avg_detection": float(row["avg_detection"]) if row["avg_detection"] else None,
            })
    return values


DEFAULT_COMPARISON_DATA = {
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


def build_comparison_data():
    config = load_config()
    eval_cfg = config.get("evaluation", {})
    baseline = eval_cfg.get("baseline_metrics", {})
    data = {
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

    normal = baseline.get("normal", {})
    if "aodv" in normal:
        aodv = normal["aodv"]
        data["Normal"]["AODV"] = [aodv.get("avg_pdr", data["Normal"]["AODV"][0]), aodv.get("avg_delay_ms", data["Normal"]["AODV"][1]), aodv.get("avg_detection", data["Normal"]["AODV"][2])]
    if "standard_marl" in normal:
        marl = normal["standard_marl"]
        data["Normal"]["Standard MARL"] = [marl.get("avg_pdr", data["Normal"]["Standard MARL"][0]), marl.get("avg_delay_ms", data["Normal"]["Standard MARL"][1]), marl.get("avg_detection", data["Normal"]["Standard MARL"][2])]

    attack = baseline.get("malicious_attack", {})
    if "aodv" in attack:
        aodv = attack["aodv"]
        data["Malicious Attack"]["AODV"] = [aodv.get("avg_pdr", data["Malicious Attack"]["AODV"][0]), aodv.get("avg_delay_ms", data["Malicious Attack"]["AODV"][1]), aodv.get("avg_detection", data["Malicious Attack"]["AODV"][2])]
    if "standard_marl" in attack:
        marl = attack["standard_marl"]
        data["Malicious Attack"]["Standard MARL"] = [marl.get("avg_pdr", data["Malicious Attack"]["Standard MARL"][0]), marl.get("avg_delay_ms", data["Malicious Attack"]["Standard MARL"][1]), marl.get("avg_detection", data["Malicious Attack"]["Standard MARL"][2])]

    return data


def plot_bar_comparison(save_path="logs/protocol_comparison_bar.png"):
    apply_thesis_style()
    print("=== 프로토콜 성능 비교 바 차트 생성 ===")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    data = build_comparison_data()
    eval_data = load_eval_metrics("logs/eval_metrics.csv")
    if eval_data:
        grouped = {}
        for row in eval_data:
            grouped.setdefault(row["scenario"], []).append(row)

        if "Default" in grouped:
            rows = grouped["Default"]
            avg_pdr = np.mean([d["avg_pdr"] for d in rows]) * 100.0
            avg_delay = np.mean([d["avg_delay_ms"] for d in rows])
            avg_detection = None
            if any(d["avg_detection"] is not None for d in rows):
                avg_detection = np.mean([d["avg_detection"] for d in rows if d["avg_detection"] is not None]) * 100.0
            data["Normal"]["EMARL-XAI"] = [
                avg_pdr,
                avg_delay,
                avg_detection if avg_detection is not None else data["Normal"]["EMARL-XAI"][2],
            ]

        malicious_rows = []
        for scenario_name, rows in grouped.items():
            if scenario_name != "Default":
                malicious_rows.extend(rows)

        if malicious_rows:
            avg_pdr = np.mean([d["avg_pdr"] for d in malicious_rows]) * 100.0
            avg_delay = np.mean([d["avg_delay_ms"] for d in malicious_rows])
            avg_detection = None
            if any(d["avg_detection"] is not None for d in malicious_rows):
                avg_detection = np.mean([d["avg_detection"] for d in malicious_rows if d["avg_detection"] is not None]) * 100.0
            data["Malicious Attack"]["EMARL-XAI"] = [
                avg_pdr,
                avg_delay,
                avg_detection if avg_detection is not None else data["Malicious Attack"]["EMARL-XAI"][2],
            ]

    protocols = ["AODV", "Standard MARL", "EMARL-XAI"]
    colors = [COLORS["aodv"], COLORS["marl"], COLORS["emarl"]]
    x = np.arange(3)
    width = 0.22

    for ax_idx, (scenario, scenario_data) in enumerate(data.items()):
        ax = axes[ax_idx]
        metrics = scenario_data["metrics"]

        for i, protocol in enumerate(protocols):
            values = scenario_data[protocol]
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
