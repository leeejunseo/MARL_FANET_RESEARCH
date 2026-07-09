"""Compare Q-value diagnostics between archived MADDPG and MATD3 runs."""

import argparse
import csv
import glob
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Q-value logs (overestimation diagnostics).")
    parser.add_argument("--baseline-dir", required=True, help="Archived baseline directory")
    parser.add_argument("--candidate-dir", required=True, help="Archived candidate directory")
    parser.add_argument("--baseline-label", default="MADDPG")
    parser.add_argument("--candidate-label", default="MATD3")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--prefix", default="compare_maddpg_vs_matd3_q")
    return parser.parse_args()


def pick_artifact(run_dir, base_name, preferred_suffix=None):
    if preferred_suffix:
        suffix_candidate = os.path.join(
            run_dir,
            f"{os.path.splitext(base_name)[0]}_{preferred_suffix}{os.path.splitext(base_name)[1]}",
        )
        if os.path.exists(suffix_candidate):
            return suffix_candidate

    direct = os.path.join(run_dir, base_name)
    if os.path.exists(direct):
        return direct

    pattern = os.path.join(run_dir, f"{os.path.splitext(base_name)[0]}_*{os.path.splitext(base_name)[1]}")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else direct


def load_q_rows(csv_path):
    if not os.path.exists(csv_path):
        return []

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("episode"):
                continue
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "q_current_mean": _to_float(row.get("q_current_mean")),
                    "q_target_mean": _to_float(row.get("q_target_mean")),
                    "q_overestimation_gap": _to_float(row.get("q_overestimation_gap")),
                    "q_abs_td_error": _to_float(row.get("q_abs_td_error")),
                    "q1_current_mean": _to_float(row.get("q1_current_mean")),
                    "q2_current_mean": _to_float(row.get("q2_current_mean")),
                    "q_disagreement_mean": _to_float(row.get("q_disagreement_mean")),
                    "critic_loss1_mean": _to_float(row.get("critic_loss1_mean")),
                    "critic_loss2_mean": _to_float(row.get("critic_loss2_mean")),
                    "actor_loss_mean": _to_float(row.get("actor_loss_mean")),
                    "actor_update_ratio": _to_float(row.get("actor_update_ratio")),
                }
            )
    return rows


def dedupe_and_sort_rows(rows):
    """Keep the last record per episode and return rows sorted by episode."""
    if not rows:
        return []

    by_episode = OrderedDict()
    for row in rows:
        ep = row.get("episode")
        if ep is None:
            continue
        by_episode[ep] = row

    return [by_episode[k] for k in sorted(by_episode.keys())]


def _to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    return float(s)


def series(rows, key):
    x = []
    y = []
    for row in rows:
        val = row.get(key)
        if val is None:
            continue
        x.append(row["episode"])
        y.append(val)
    if not x:
        return None, None
    return np.array(x), np.array(y)


def tail_mean(values, tail=50):
    if values is None or len(values) == 0:
        return None
    return float(np.mean(values[-min(tail, len(values)) :]))


def safe_mean(values):
    if values is None or len(values) == 0:
        return None
    return float(np.mean(values))


def summarize(rows):
    _, gap = series(rows, "q_overestimation_gap")
    _, td = series(rows, "q_abs_td_error")
    _, dis = series(rows, "q_disagreement_mean")
    _, upd = series(rows, "actor_update_ratio")
    abs_gap = np.abs(gap) if gap is not None else None

    return {
        "episodes": len(rows),
        "mean_q_overestimation_gap": safe_mean(gap),
        "tail50_q_overestimation_gap": tail_mean(gap, tail=50),
        "mean_abs_q_overestimation_gap": safe_mean(abs_gap),
        "median_abs_q_overestimation_gap": None if abs_gap is None or len(abs_gap) == 0 else float(np.median(abs_gap)),
        "p90_abs_q_overestimation_gap": None if abs_gap is None or len(abs_gap) == 0 else float(np.percentile(abs_gap, 90)),
        "positive_gap_ratio": None if gap is None or len(gap) == 0 else float(np.mean(gap > 0.0)),
        "mean_q_abs_td_error": safe_mean(td),
        "tail50_q_abs_td_error": tail_mean(td, tail=50),
        "mean_q_disagreement": safe_mean(dis),
        "tail50_q_disagreement": tail_mean(dis, tail=50),
        "mean_actor_update_ratio": safe_mean(upd),
    }


def fmt(v):
    if v is None:
        return ""
    return f"{v:.6f}"


def write_summary_csv(path, base_label, cand_label, base_summary, cand_summary):
    fields = [
        "metric",
        f"{base_label}",
        f"{cand_label}",
        "delta_candidate_minus_baseline",
    ]
    metrics = [
        "mean_q_overestimation_gap",
        "tail50_q_overestimation_gap",
        "mean_abs_q_overestimation_gap",
        "median_abs_q_overestimation_gap",
        "p90_abs_q_overestimation_gap",
        "positive_gap_ratio",
        "mean_q_abs_td_error",
        "tail50_q_abs_td_error",
        "mean_q_disagreement",
        "tail50_q_disagreement",
        "mean_actor_update_ratio",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in metrics:
            b = base_summary.get(m)
            c = cand_summary.get(m)
            writer.writerow(
                {
                    "metric": m,
                    f"{base_label}": fmt(b),
                    f"{cand_label}": fmt(c),
                    "delta_candidate_minus_baseline": "" if b is None or c is None else fmt(c - b),
                }
            )


def write_summary_text(path, base_label, cand_label, base_summary, cand_summary):
    b_gap = base_summary.get("tail50_q_overestimation_gap")
    c_gap = cand_summary.get("tail50_q_overestimation_gap")
    b_abs = base_summary.get("mean_abs_q_overestimation_gap")
    c_abs = cand_summary.get("mean_abs_q_overestimation_gap")
    b_p90 = base_summary.get("p90_abs_q_overestimation_gap")
    c_p90 = cand_summary.get("p90_abs_q_overestimation_gap")
    b_td = base_summary.get("tail50_q_abs_td_error")
    c_td = cand_summary.get("tail50_q_abs_td_error")

    lines = []
    lines.append("Q-value comparison summary")
    lines.append("")
    lines.append(f"Baseline={base_label}")
    lines.append(f"Candidate={cand_label}")
    lines.append("")
    lines.append(f"{base_label} tail50 q_overestimation_gap: {fmt(b_gap)}")
    lines.append(f"{cand_label} tail50 q_overestimation_gap: {fmt(c_gap)}")
    if b_gap is not None and c_gap is not None:
        lines.append(f"delta (candidate - baseline): {fmt(c_gap - b_gap)}")
        lines.append(
            "interpretation: lower q_overestimation_gap is better (closer to conservative value estimation)."
        )
    lines.append("")
    lines.append(f"{base_label} mean |q_overestimation_gap|: {fmt(b_abs)}")
    lines.append(f"{cand_label} mean |q_overestimation_gap|: {fmt(c_abs)}")
    if b_abs is not None and c_abs is not None:
        lines.append(f"delta (candidate - baseline): {fmt(c_abs - b_abs)}")
        lines.append("interpretation: lower |q_overestimation_gap| means more stable and less biased value estimates.")
    lines.append("")
    lines.append(f"{base_label} p90 |q_overestimation_gap|: {fmt(b_p90)}")
    lines.append(f"{cand_label} p90 |q_overestimation_gap|: {fmt(c_p90)}")
    if b_p90 is not None and c_p90 is not None:
        lines.append(f"delta (candidate - baseline): {fmt(c_p90 - b_p90)}")
        lines.append("interpretation: lower p90 indicates fewer severe overestimation spikes.")
    lines.append("")
    lines.append(f"{base_label} tail50 q_abs_td_error: {fmt(b_td)}")
    lines.append(f"{cand_label} tail50 q_abs_td_error: {fmt(c_td)}")
    if b_td is not None and c_td is not None:
        lines.append(f"delta (candidate - baseline): {fmt(c_td - b_td)}")
        lines.append("interpretation: lower q_abs_td_error is better (better Bellman consistency).")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_abs_gap_boxplot(base_rows, cand_rows, base_label, cand_label, out_path):
    _, b_gap = series(base_rows, "q_overestimation_gap")
    _, c_gap = series(cand_rows, "q_overestimation_gap")
    if b_gap is None and c_gap is None:
        return False

    data = []
    labels = []
    if b_gap is not None and len(b_gap) > 0:
        data.append(np.abs(b_gap))
        labels.append(base_label)
    if c_gap is not None and len(c_gap) > 0:
        data.append(np.abs(c_gap))
        labels.append(cand_label)

    if not data:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5.2))
    bp = plt.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    palette = ["#8fbad9", "#9fd3a8"]
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_alpha(0.8)

    plt.title("Absolute Q Overestimation Gap Distribution")
    plt.ylabel("|Q gap|")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_gap_curve(base_rows, cand_rows, base_label, cand_label, out_path):
    b_ep, b_gap = series(base_rows, "q_overestimation_gap")
    c_ep, c_gap = series(cand_rows, "q_overestimation_gap")
    if b_ep is None and c_ep is None:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 5.5))
    if b_ep is not None:
        plt.plot(b_ep, b_gap, alpha=0.35, linewidth=1.1, label=f"{base_label} gap")
        if len(b_gap) >= 20:
            plt.plot(b_ep[19:], moving_average(b_gap, 20), linewidth=2.0, label=f"{base_label} MA(20)")

    if c_ep is not None:
        plt.plot(c_ep, c_gap, alpha=0.35, linewidth=1.1, label=f"{cand_label} gap")
        if len(c_gap) >= 20:
            plt.plot(c_ep[19:], moving_average(c_gap, 20), linewidth=2.0, label=f"{cand_label} MA(20)")

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.title("Q Overestimation Gap (current - target)")
    plt.xlabel("Episode")
    plt.ylabel("Q gap")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def moving_average(x, window):
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="valid")


def main():
    args = parse_args()

    base_suffix = args.baseline_label.lower()
    cand_suffix = args.candidate_label.lower()

    base_q_csv = pick_artifact(args.baseline_dir, "training_q_values.csv", preferred_suffix=base_suffix)
    cand_q_csv = pick_artifact(args.candidate_dir, "training_q_values.csv", preferred_suffix=cand_suffix)

    # Backward-compatible fallback to the naming generated from log_path roots
    # (e.g., training_rewards_q_values.csv).
    if not os.path.exists(base_q_csv):
        base_q_csv = pick_artifact(args.baseline_dir, "training_rewards_q_values.csv", preferred_suffix=base_suffix)
    if not os.path.exists(cand_q_csv):
        cand_q_csv = pick_artifact(args.candidate_dir, "training_rewards_q_values.csv", preferred_suffix=cand_suffix)

    base_rows = load_q_rows(base_q_csv)
    cand_rows = load_q_rows(cand_q_csv)

    if not base_rows and not cand_rows:
        print("Q 로그를 찾지 못했습니다. train.py를 최신 버전으로 재학습하여 training_q_values*.csv를 생성하세요.")
        return

    base_raw_count = len(base_rows)
    cand_raw_count = len(cand_rows)
    base_rows = dedupe_and_sort_rows(base_rows)
    cand_rows = dedupe_and_sort_rows(cand_rows)

    base_summary = summarize(base_rows)
    cand_summary = summarize(cand_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")
    summary_txt = os.path.join(args.output_dir, f"{args.prefix}_summary.txt")
    gap_plot = os.path.join(args.output_dir, f"{args.prefix}_gap_curve.png")
    abs_gap_boxplot = os.path.join(args.output_dir, f"{args.prefix}_abs_gap_boxplot.png")

    write_summary_csv(summary_csv, args.baseline_label, args.candidate_label, base_summary, cand_summary)
    write_summary_text(summary_txt, args.baseline_label, args.candidate_label, base_summary, cand_summary)
    plotted = plot_gap_curve(base_rows, cand_rows, args.baseline_label, args.candidate_label, gap_plot)
    plotted_box = plot_abs_gap_boxplot(base_rows, cand_rows, args.baseline_label, args.candidate_label, abs_gap_boxplot)

    print("Q 비교 완료")
    print(f"- baseline q csv: {base_q_csv}")
    print(f"- candidate q csv: {cand_q_csv}")
    print(f"- baseline rows(raw->dedup): {base_raw_count} -> {len(base_rows)}")
    print(f"- candidate rows(raw->dedup): {cand_raw_count} -> {len(cand_rows)}")
    print(f"- summary csv: {summary_csv}")
    print(f"- summary txt: {summary_txt}")
    print(f"- gap plot: {gap_plot if plotted else '생성 안됨(데이터 부족)'}")
    print(f"- abs gap boxplot: {abs_gap_boxplot if plotted_box else '생성 안됨(데이터 부족)'}")


if __name__ == "__main__":
    main()
