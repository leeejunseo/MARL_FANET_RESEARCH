"""논문용 전체 시각화 일괄 생성."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_learning_curve import plot_convergence
from utils.plot_roc_auc import plot_roc_auc
from utils.plot_bar_comparison import plot_bar_comparison
from utils.plot_xai_heatmap import plot_all_xai


def generate_all_plots():
    print("=" * 60)
    print("  MARL-FANET 논문용 시각화 일괄 생성")
    print("=" * 60)

    plot_convergence()
    plot_roc_auc()
    plot_bar_comparison()
    plot_all_xai()

    print("\n" + "=" * 60)
    print("  모든 그래프 생성 완료 — logs/ 디렉터리 확인")
    print("=" * 60)
    print("\n생성 파일:")
    outputs = [
        "logs/learning_curve_high_dpi.png",
        "logs/roc_curve_auc.png",
        "logs/protocol_comparison_bar.png",
        "logs/xai_feature_heatmap.png",
        "logs/xai_shap_summary.png",
        "logs/xai_feature_importance.png",
    ]
    for f in outputs:
        status = "OK" if os.path.exists(f) else "MISSING"
        print(f"  [{status}] {f}")


if __name__ == "__main__":
    generate_all_plots()
