"""XAI 히트맵 및 SHAP Summary Plot 시각화."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from analysis.xai_explainer import (
    FEATURE_NAMES,
    RoutingSurrogate,
    build_extended_observations,
    train_surrogate,
    integrated_gradients,
    compute_feature_importance,
    compute_correlation_heatmap,
)
from utils.plot_style import apply_thesis_style, save_figure, COLORS


def plot_feature_heatmap(save_path="logs/xai_feature_heatmap.png"):
    apply_thesis_style()
    print("=== XAI 특성 상관 히트맵 생성 ===")

    observations = build_extended_observations(n_samples=500)
    model = RoutingSurrogate(input_dim=len(FEATURE_NAMES))
    labels = train_surrogate(model, observations)
    corr = compute_correlation_heatmap(observations, labels)

    feature_labels = FEATURE_NAMES + ["Route\nDecision"]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(feature_labels)))
    ax.set_yticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha="right")
    ax.set_yticklabels(feature_labels)
    ax.set_title("XAI Feature Correlation Heatmap", fontweight="bold", pad=15)

    for i in range(len(feature_labels)):
        for j in range(len(feature_labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")
    save_figure(save_path)


def plot_shap_summary(save_path="logs/xai_shap_summary.png"):
    apply_thesis_style()
    print("=== XAI SHAP Summary Plot 생성 ===")

    observations = build_extended_observations(n_samples=400)
    model = RoutingSurrogate(input_dim=len(FEATURE_NAMES))
    train_surrogate(model, observations)
    attributions = integrated_gradients(model, observations)

    # SHAP beeswarm 스타일: y=특성, x=기여도, 색=특성값
    fig, ax = plt.subplots(figsize=(10, 6))
    n_features = len(FEATURE_NAMES)

    for i in range(n_features):
        shap_vals = attributions[:, i]
        feat_vals = observations[:, i]
        y_jitter = np.full(len(shap_vals), n_features - 1 - i) + np.random.uniform(-0.15, 0.15, len(shap_vals))
        scatter = ax.scatter(
            shap_vals, y_jitter, c=feat_vals, cmap="coolwarm",
            alpha=0.6, s=12, edgecolors="none",
        )

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(list(reversed(FEATURE_NAMES)))
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Feature Contribution (Integrated Gradients)")
    ax.set_title("SHAP Summary Plot: Routing Decision Explainability", fontweight="bold", pad=15)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Feature Value (normalized)")
    ax.grid(True, axis="x", alpha=0.3)
    save_figure(save_path)


def plot_feature_importance_bar(save_path="logs/xai_feature_importance.png"):
    apply_thesis_style()
    print("=== XAI 특성 중요도 바 차트 생성 ===")

    observations = build_extended_observations(n_samples=500)
    model = RoutingSurrogate(input_dim=len(FEATURE_NAMES))
    train_surrogate(model, observations)
    attributions = integrated_gradients(model, observations)
    importance = compute_feature_importance(attributions)

    sorted_idx = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        importance[sorted_idx],
        color=COLORS["primary"], alpha=0.85,
    )
    for bar, val in zip(bars, importance[sorted_idx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |Integrated Gradient| Contribution")
    ax.set_title("XAI Feature Importance for Routing Decisions", fontweight="bold", pad=15)
    ax.grid(True, axis="x")
    save_figure(save_path)


def plot_all_xai():
    plot_feature_heatmap()
    plot_shap_summary()
    plot_feature_importance_bar()


if __name__ == "__main__":
    plot_all_xai()
