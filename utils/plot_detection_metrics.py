"""Detection performance metrics plot for evaluation results."""

import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from utils.plot_style import apply_thesis_style, save_figure, COLORS


def load_eval_metrics(filepath="logs/eval_metrics.csv"):
    if not os.path.exists(filepath):
        return None
    metrics = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append({
                "scenario": row.get("scenario", "Default"),
                "accuracy": float(row["avg_detection_accuracy"]) if row.get("avg_detection_accuracy") else None,
                "precision": float(row["avg_detection_precision"]) if row.get("avg_detection_precision") else None,
                "recall": float(row["avg_detection_recall"]) if row.get("avg_detection_recall") else None,
                "f1": float(row["avg_detection_f1"]) if row.get("avg_detection_f1") else None,
            })
    return metrics


def aggregate_scenarios(metrics):
    groups = {
        "Normal": [],
        "Malicious Attack": [],
    }
    for row in metrics:
        if row["scenario"] == "Default":
            groups["Normal"].append(row)
        else:
            groups["Malicious Attack"].append(row)

    summary = {}
    for label, rows in groups.items():
        summary[label] = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
        if not rows:
            continue
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [r[metric] for r in rows if r[metric] is not None]
            if values:
                summary[label][metric] = float(np.mean(values) * 100.0)
    return summary


def plot_detection_metrics(save_path="logs/detection_metrics.png"):
    apply_thesis_style()
    print("=== Detection Metrics Comparison 그래프 생성 ===")

    metrics = load_eval_metrics("logs/eval_metrics.csv")
    if metrics is None:
        print("  -> logs/eval_metrics.csv를 찾을 수 없어 그래프 생성이 불가능합니다.")
        return

    summary = aggregate_scenarios(metrics)
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.35

    normal = [summary["Normal"][m.lower()] if summary["Normal"][m.lower()] is not None else 0.0 for m in labels]
    attack = [summary["Malicious Attack"][m.lower()] if summary["Malicious Attack"][m.lower()] is not None else 0.0 for m in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, normal, width, label="Normal", color=COLORS["emarl"], alpha=0.9)
    rects2 = ax.bar(x + width / 2, attack, width, label="Malicious Attack", color=COLORS["marl"], alpha=0.9)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Detection Classification Metrics by Scenario", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.4)

    save_figure(save_path)
    return summary


if __name__ == "__main__":
    plot_detection_metrics()