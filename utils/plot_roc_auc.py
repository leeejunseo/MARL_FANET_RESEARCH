"""ROC Curve & AUC 시각화."""

import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
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


def discover_scenario_roc_files(template="logs/eval_node_features_{scenario}.npz"):
    search_pattern = template.replace("{scenario}", "*")
    return sorted(glob.glob(search_pattern))


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


def load_eval_metrics(filepath="logs/eval_metrics.csv"):
    if not os.path.exists(filepath):
        return None
    rows = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_roc_auc(
    save_path="logs/roc_curve_auc.png",
    compare_models=True,
    roc_data_path="logs/eval_node_features.npz",
    roc_data_template="logs/eval_node_features_{scenario}.npz",
):
    apply_thesis_style()
    print("=== ROC Curve & AUC 그래프 생성 ===")

    scenario_paths = discover_scenario_roc_files(roc_data_template)
    features, labels = None, None

    if scenario_paths:
        print(f"  -> 시나리오별 ROC 데이터 파일 발견: {len(scenario_paths)}개")
    else:
        features, labels = load_eval_node_features(roc_data_path)
        if features is not None and labels is not None:
            print(f"  -> 통합 ROC 데이터 로드: {roc_data_path}")
        else:
            print("  -> ROC 데이터가 없어 합성 데이터로 fallback합니다.")
            features, labels = generate_node_dataset()

    configs = MODEL_CONFIGS if compare_models else MODEL_CONFIGS[:1]
    fig, ax = plt.subplots(figsize=(10, 9))
    results = {}

    if scenario_paths:
        for scenario_path in scenario_paths:
            scenario_name = os.path.splitext(os.path.basename(scenario_path))[0].replace("eval_node_features_", "")
            scenario_features, scenario_labels = load_eval_node_features(scenario_path)
            if scenario_features is None or scenario_labels is None:
                continue

            for cfg in configs:
                detector = MaliciousNodeDetector(
                    weights=np.array(cfg["weights"]),
                    temperature=cfg["temperature"],
                    bias=cfg["bias"],
                )
                fpr, tpr, roc_auc = evaluate_roc(
                    detector,
                    scenario_features,
                    scenario_labels,
                    degradation=cfg["degradation"],
                )
                label = f"{cfg['name']} ({scenario_name})"
                results[label] = roc_auc
                ax.plot(
                    fpr,
                    tpr,
                    linewidth=2.2,
                    label=f"{label} (AUC={roc_auc:.3f})",
                    color=cfg["color"],
                    alpha=0.9,
                    linestyle="-" if scenario_name == "default" else "--",
                )
    else:
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
            ax.plot(
                fpr,
                tpr,
                color=cfg["color"],
                linewidth=2.8,
                label=f"{cfg['name']} (AUC = {roc_auc:.3f})",
            )

    eval_rows = load_eval_metrics("logs/eval_metrics.csv")
    detection_text = []
    if eval_rows:
        grouped = {}
        for row in eval_rows:
            grouped.setdefault(row.get("scenario", "Default"), []).append(row)

        for scenario_name, rows in grouped.items():
            accuracy_vals = [float(r["avg_detection_accuracy"]) for r in rows if r.get("avg_detection_accuracy")]
            precision_vals = [float(r["avg_detection_precision"]) for r in rows if r.get("avg_detection_precision")]
            recall_vals = [float(r["avg_detection_recall"]) for r in rows if r.get("avg_detection_recall")]
            f1_vals = [float(r["avg_detection_f1"]) for r in rows if r.get("avg_detection_f1")]
            if accuracy_vals and precision_vals and recall_vals and f1_vals:
                mean_acc = np.mean(accuracy_vals) * 100.0
                mean_prec = np.mean(precision_vals) * 100.0
                mean_rec = np.mean(recall_vals) * 100.0
                mean_f1 = np.mean(f1_vals) * 100.0
                detection_text.append(
                    f"{scenario_name}: Acc={mean_acc:.1f}%, Prec={mean_prec:.1f}%, Rec={mean_rec:.1f}%, F1={mean_f1:.1f}%"
                )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.45, linewidth=1.5,
            label="Random Classifier (AUC = 0.500)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve: Malicious Node Detection Performance", fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")

    if detection_text:
        text_y = 0.95
        for line in detection_text:
            ax.text(0.02, text_y, line, transform=ax.transAxes, fontsize=9, verticalalignment="top")
            text_y -= 0.05

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
