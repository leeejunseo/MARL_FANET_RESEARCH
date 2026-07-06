"""Compare two archived experiment runs with plots and summary tables."""

import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.malicious_detector import MaliciousNodeDetector, _auc, _roc_curve
from utils.plot_style import apply_thesis_style


def parse_args():
    parser = argparse.ArgumentParser(description="Compare archived MADDPG vs MATD3 results.")
    parser.add_argument("--baseline-dir", required=True, help="Archived baseline directory")
    parser.add_argument("--candidate-dir", required=True, help="Archived candidate directory")
    parser.add_argument("--baseline-label", default="MADDPG")
    parser.add_argument("--candidate-label", default="MATD3")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--prefix", default="compare_maddpg_vs_matd3")
    return parser.parse_args()


def pick_artifact(run_dir, base_name, preferred_suffix=None):
    if preferred_suffix:
        suffix_candidate = os.path.join(run_dir, f"{os.path.splitext(base_name)[0]}_{preferred_suffix}{os.path.splitext(base_name)[1]}")
        if os.path.exists(suffix_candidate):
            return suffix_candidate

    direct = os.path.join(run_dir, base_name)
    if os.path.exists(direct):
        return direct

    pattern = os.path.join(run_dir, f"{os.path.splitext(base_name)[0]}_*{os.path.splitext(base_name)[1]}")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else direct


def load_training_rewards(csv_path):
    if not os.path.exists(csv_path):
        return None, None

    episodes = []
    rewards = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
    if not episodes:
        return None, None
    return np.array(episodes), np.array(rewards)


def load_eval_rows(csv_path):
    if not os.path.exists(csv_path):
        return []

    out = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return out

        # Some archived files contain mixed row schemas where a "policy" value exists
        # but the file header does not include the "policy" column.
        has_policy_in_header = "policy" in header
        expected_len = len(header)

        for values in reader:
            if not values:
                continue

            row_values = values
            if (not has_policy_in_header) and len(values) == expected_len + 1:
                # Assume inserted position is after scenario:
                # episode, scenario, policy, total_reward, ...
                fixed_header = header[:2] + ["policy"] + header[2:]
                row = dict(zip(fixed_header, values))
            else:
                if len(values) < expected_len:
                    continue
                row = dict(zip(header, values[:expected_len]))

            out.append(row)
    return out


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def aggregate_eval(eval_rows):
    grouped = {}
    metrics = ["avg_pdr", "avg_delay_ms", "avg_detection", "avg_detection_f1", "avg_disconnect", "avg_trust"]

    for row in eval_rows:
        scenario = row.get("scenario", "Default")
        grouped.setdefault(scenario, {m: [] for m in metrics})
        for m in metrics:
            val = to_float(row.get(m))
            if val is not None:
                grouped[scenario][m].append(val)

    agg = {}
    for scenario, data in grouped.items():
        agg[scenario] = {}
        for m, values in data.items():
            agg[scenario][m] = float(np.mean(values)) if values else None
    return agg


def _smooth(values, window=30):
    if values is None or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curve(base_ep, base_rw, cand_ep, cand_rw, base_label, cand_label, out_path):
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if base_ep is not None and base_rw is not None:
        ax.plot(base_ep, base_rw, alpha=0.22, label=f"{base_label} raw")
        sm = _smooth(base_rw, window=30)
        if sm is not None:
            ax.plot(base_ep[len(base_ep) - len(sm):], sm, linewidth=2.4, label=f"{base_label} MA(30)")

    if cand_ep is not None and cand_rw is not None:
        ax.plot(cand_ep, cand_rw, alpha=0.22, label=f"{cand_label} raw")
        sm = _smooth(cand_rw, window=30)
        if sm is not None:
            ax.plot(cand_ep[len(cand_ep) - len(sm):], sm, linewidth=2.4, label=f"{cand_label} MA(30)")

    ax.set_title("Training Reward Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eval_bar(base_agg, cand_agg, base_label, cand_label, out_path):
    apply_thesis_style()

    scenarios = sorted(set(base_agg.keys()) | set(cand_agg.keys()))
    if not scenarios:
        return False

    metrics = ["avg_pdr", "avg_delay_ms", "avg_detection_f1"]
    metric_titles = ["PDR", "Delay (ms)", "Detection F1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(scenarios))
    width = 0.35

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        base_vals = []
        cand_vals = []

        for sc in scenarios:
            base_vals.append((base_agg.get(sc, {}) or {}).get(metric) or 0.0)
            cand_vals.append((cand_agg.get(sc, {}) or {}).get(metric) or 0.0)

        ax.bar(x - width / 2, base_vals, width=width, label=base_label)
        ax.bar(x + width / 2, cand_vals, width=width, label=cand_label)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0].legend(loc="best")
    fig.suptitle("Evaluation Metric Comparison by Scenario", y=1.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def load_roc_data(npz_path):
    if not os.path.exists(npz_path):
        return None, None
    with np.load(npz_path) as data:
        return data["features"], data["labels"]


def compute_auc_from_npz(npz_path):
    features, labels = load_roc_data(npz_path)
    if features is None or labels is None:
        return None, None, None

    detector = MaliciousNodeDetector()
    y_score = detector.predict_proba(features)
    fpr, tpr = _roc_curve(labels.astype(int), y_score)
    auc = _auc(fpr, tpr)
    return fpr, tpr, float(auc)


def plot_roc_comparison(base_npz, cand_npz, base_label, cand_label, out_path):
    base = compute_auc_from_npz(base_npz)
    cand = compute_auc_from_npz(cand_npz)

    if base[0] is None and cand[0] is None:
        return None, None

    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    base_auc = None
    cand_auc = None

    if base[0] is not None:
        fpr, tpr, base_auc = base
        ax.plot(fpr, tpr, linewidth=2.4, label=f"{base_label} (AUC={base_auc:.3f})")

    if cand[0] is not None:
        fpr, tpr, cand_auc = cand
        ax.plot(fpr, tpr, linewidth=2.4, label=f"{cand_label} (AUC={cand_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC Comparison")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return base_auc, cand_auc


def write_summary(summary_csv_path, base_label, cand_label, base_agg, cand_agg, base_auc, cand_auc):
    scenarios = sorted(set(base_agg.keys()) | set(cand_agg.keys()))
    fields = [
        "scenario",
        f"{base_label}_avg_pdr",
        f"{cand_label}_avg_pdr",
        "delta_avg_pdr",
        f"{base_label}_avg_delay_ms",
        f"{cand_label}_avg_delay_ms",
        "delta_avg_delay_ms",
        f"{base_label}_avg_detection_f1",
        f"{cand_label}_avg_detection_f1",
        "delta_avg_detection_f1",
    ]

    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sc in scenarios:
            b = base_agg.get(sc, {})
            c = cand_agg.get(sc, {})

            bpdr = b.get("avg_pdr")
            cpdr = c.get("avg_pdr")
            bdelay = b.get("avg_delay_ms")
            cdelay = c.get("avg_delay_ms")
            bf1 = b.get("avg_detection_f1")
            cf1 = c.get("avg_detection_f1")

            writer.writerow({
                "scenario": sc,
                f"{base_label}_avg_pdr": "" if bpdr is None else f"{bpdr:.6f}",
                f"{cand_label}_avg_pdr": "" if cpdr is None else f"{cpdr:.6f}",
                "delta_avg_pdr": "" if bpdr is None or cpdr is None else f"{(cpdr - bpdr):.6f}",
                f"{base_label}_avg_delay_ms": "" if bdelay is None else f"{bdelay:.6f}",
                f"{cand_label}_avg_delay_ms": "" if cdelay is None else f"{cdelay:.6f}",
                "delta_avg_delay_ms": "" if bdelay is None or cdelay is None else f"{(cdelay - bdelay):.6f}",
                f"{base_label}_avg_detection_f1": "" if bf1 is None else f"{bf1:.6f}",
                f"{cand_label}_avg_detection_f1": "" if cf1 is None else f"{cf1:.6f}",
                "delta_avg_detection_f1": "" if bf1 is None or cf1 is None else f"{(cf1 - bf1):.6f}",
            })

    auc_path = summary_csv_path.replace(".csv", "_auc.txt")
    with open(auc_path, "w", encoding="utf-8") as f:
        f.write(f"{base_label}_auc={'' if base_auc is None else f'{base_auc:.6f}'}\n")
        f.write(f"{cand_label}_auc={'' if cand_auc is None else f'{cand_auc:.6f}'}\n")
        if base_auc is not None and cand_auc is not None:
            f.write(f"delta_auc={cand_auc - base_auc:.6f}\n")


def main():
    args = parse_args()

    base_suffix = args.baseline_label.lower()
    cand_suffix = args.candidate_label.lower()
    base_train_csv = pick_artifact(args.baseline_dir, "training_rewards.csv", preferred_suffix=base_suffix)
    cand_train_csv = pick_artifact(args.candidate_dir, "training_rewards.csv", preferred_suffix=cand_suffix)
    base_eval_csv = pick_artifact(args.baseline_dir, "eval_metrics.csv", preferred_suffix=base_suffix)
    cand_eval_csv = pick_artifact(args.candidate_dir, "eval_metrics.csv", preferred_suffix=cand_suffix)
    base_roc_npz = pick_artifact(args.baseline_dir, "eval_node_features.npz", preferred_suffix=base_suffix)
    cand_roc_npz = pick_artifact(args.candidate_dir, "eval_node_features.npz", preferred_suffix=cand_suffix)

    base_ep, base_rw = load_training_rewards(base_train_csv)
    cand_ep, cand_rw = load_training_rewards(cand_train_csv)
    base_eval_rows = load_eval_rows(base_eval_csv)
    cand_eval_rows = load_eval_rows(cand_eval_csv)

    base_agg = aggregate_eval(base_eval_rows)
    cand_agg = aggregate_eval(cand_eval_rows)

    os.makedirs(args.output_dir, exist_ok=True)

    learning_path = os.path.join(args.output_dir, f"{args.prefix}_learning_curve.png")
    eval_bar_path = os.path.join(args.output_dir, f"{args.prefix}_eval_bar.png")
    roc_path = os.path.join(args.output_dir, f"{args.prefix}_roc_auc.png")
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")

    plot_learning_curve(
        base_ep,
        base_rw,
        cand_ep,
        cand_rw,
        args.baseline_label,
        args.candidate_label,
        learning_path,
    )
    bar_ok = plot_eval_bar(base_agg, cand_agg, args.baseline_label, args.candidate_label, eval_bar_path)
    base_auc, cand_auc = plot_roc_comparison(
        base_roc_npz,
        cand_roc_npz,
        args.baseline_label,
        args.candidate_label,
        roc_path,
    )
    write_summary(summary_path, args.baseline_label, args.candidate_label, base_agg, cand_agg, base_auc, cand_auc)

    print("=== Algorithm Comparison Completed ===")
    print(f"Learning curve: {learning_path}")
    print(f"Eval bar chart: {eval_bar_path if bar_ok else 'Skipped (no eval metrics)'}")
    print(f"ROC-AUC chart: {roc_path if (base_auc is not None or cand_auc is not None) else 'Skipped (no ROC npz)'}")
    print(f"Summary table: {summary_path}")


if __name__ == "__main__":
    main()
