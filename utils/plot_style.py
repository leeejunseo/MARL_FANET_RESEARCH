"""논문용 matplotlib 공통 스타일 설정."""

import matplotlib.pyplot as plt

DPI = 300
FIGSIZE = (10, 6)
COLORS = {
    "primary": "#2b6cb0",
    "secondary": "#38a169",
    "accent": "#dd6b20",
    "danger": "#e53e3e",
    "muted": "#a0aec0",
    "emarl": "#2b6cb0",
    "marl": "#805ad5",
    "aodv": "#718096",
}


def apply_thesis_style():
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    })


def save_figure(path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> 저장 완료: {path}")
