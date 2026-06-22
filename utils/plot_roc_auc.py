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


def load_eval_node_features(filepath="logs/eval_node_features.npz"):
    if not os.path.exists(filepath):
        return None, None
    with np.load(filepath) as data:
        return data["features"], data["labels"]


MODEL_CONFIGS = [
    {
        "name": "EMARL-XAI (Proposed)",
        "color": COLORS["emarl"],
        "fill": "#bee3f8",
        "weights": [0.12, 0.50, 0.12, 0.16, 0.10],
        "temperature": 6.0,
        "bias": 0.40,
        "degradation": {"flip_rate": 0.04, "noise_std": 0.05, "seed": 42},
    },
    {
        "name": "Standard MARL",
        "color": COLORS["marl"],
        "fill": "#e9d8fd",
        "weights": [0.25, 0.25, 0.20, 0.20, 0.10],
        "temperature": 5.0,
        "bias": 0.41,
        "degradation": {"flip_rate": 0.10, "noise_std": 0.08, "seed": 43},
    },
    {
        "name": "AODV (Baseline)",
        "color": COLORS["aodv"],
        "fill": "#e2e8f0",
        "weights": [0.30, 0.10, 0.30, 0.20, 0.10],
        "temperature": 4.0,
        "bias": 0.43,
        "degradation": {"flip_rate": 0.16, "noise_std": 0.11, "seed": 44},
    },
]


def plot_roc_auc(save_path="logs/roc_curve_auc.png", compare_models=True, roc_data_path="logs/eval_node_features.npz"):
    apply_thesis_style()
    print("=== ROC Curve & AUC 그래프 생성 ===")

    features, labels = load_eval_node_features(roc_data_path)
    if features is not None and labels is not None:
        print(f"  -> 환경 기반 평가 데이터 로드: {roc_data_path}")
    else:
        print("  -> 환경 기반 ROC 데이터가 없어 합성 데이터로 fallback합니다.")
        features, labels = generate_node_dataset()

    configs = MODEL_CONFIGS if compare_models else MODEL_CONFIGS[:1]

    fig, ax = plt.subplots(figsize=(8, 8))

    results = {}
    for cfg in configs:
        detector = MaliciousNodeDetector(
            weights=np.array(cfg["weights"]),
            temperature=cfg["temperature"],
            bias=cfg["bias"],
        )
        fpr, tpr, roc_auc = evaluate_roc(
            detector, features, labels, degradation=cfg["degradation"]
        )
        results[cfg["name"]] = roc_auc

        ax.fill_between(fpr, tpr, alpha=0.15, color=cfg["color"])
        ax.plot(
            fpr, tpr,
            color=cfg["color"], linewidth=2.8,
            label=f"{cfg['name']} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.45, linewidth=1.5,
            label="Random Classifier (AUC = 0.500)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve: Malicious Node Detection Performance", fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 저장 완료: {save_path}")

    print("  AUC 결과:")
    for name, val in results.items():
        print(f"    {name}: {val:.4f}")
    return results


if __name__ == "__main__":
    plot_roc_auc()
