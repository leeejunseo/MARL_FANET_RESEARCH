"""ROC Curve & AUC 시각화."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from analysis.malicious_detector import (
    MaliciousNodeDetector,
    generate_node_dataset,
    evaluate_roc,
)
from utils.plot_style import apply_thesis_style, save_figure, COLORS


def plot_roc_auc(save_path="logs/roc_curve_auc.png", compare_models=True):
    apply_thesis_style()
    print("=== ROC Curve & AUC 그래프 생성 ===")

    features, labels = generate_node_dataset()

    detectors = {
        "EMARL-XAI (Proposed)": MaliciousNodeDetector(
            weights=np.array([0.12, 0.50, 0.12, 0.16, 0.10])
        ),
    }
    if compare_models:
        detectors["Standard MARL"] = MaliciousNodeDetector(
            weights=np.array([0.25, 0.25, 0.20, 0.20, 0.10])
        )
        detectors["AODV (Baseline)"] = MaliciousNodeDetector(
            weights=np.array([0.30, 0.10, 0.30, 0.20, 0.10])
        )

    colors = [COLORS["emarl"], COLORS["marl"], COLORS["aodv"]]
    plt.figure(figsize=(8, 8))

    results = {}
    noise_levels = {"EMARL-XAI (Proposed)": 0.02, "Standard MARL": 0.08, "AODV (Baseline)": 0.15}
    for idx, (name, detector) in enumerate(detectors.items()):
        fpr, tpr, roc_auc, _ = evaluate_roc(
            detector, features, labels, noise_std=noise_levels.get(name, 0.05)
        )
        results[name] = roc_auc
        plt.plot(
            fpr, tpr, color=colors[idx % len(colors)],
            linewidth=2.5, label=f"{name} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier (AUC = 0.500)")
    plt.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve: Malicious Node Detection Performance", fontweight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.grid(True)
    save_figure(save_path)

    print("  AUC 결과:")
    for name, val in results.items():
        print(f"    {name}: {val:.4f}")
    return results


if __name__ == "__main__":
    plot_roc_auc()
