"""Generate a Markdown report from algorithm comparison artifacts."""

import argparse
import csv
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison markdown report.")
    parser.add_argument("--summary-csv", required=True, help="Path to *_summary.csv")
    parser.add_argument("--auc-txt", required=True, help="Path to *_summary_auc.txt")
    parser.add_argument("--baseline-label", default="MADDPG")
    parser.add_argument("--candidate-label", default="MATD3")
    parser.add_argument("--output-md", default="logs/compare_report.md")
    parser.add_argument("--learning-curve", default="")
    parser.add_argument("--eval-bar", default="")
    parser.add_argument("--roc-plot", default="")
    return parser.parse_args()


def to_float(text):
    text = str(text or "").strip()
    if text == "":
        return None
    return float(text)


def read_summary(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_auc(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = to_float(v.strip())
    return out


def mean(values):
    if not values:
        return None
    return sum(values) / float(len(values))


def fmt(v, nd=4):
    if v is None:
        return "N/A"
    return f"{v:.{nd}f}"


def build_report(args, rows, auc_data):
    pdr_deltas = [to_float(r.get("delta_avg_pdr")) for r in rows if to_float(r.get("delta_avg_pdr")) is not None]
    delay_deltas = [to_float(r.get("delta_avg_delay_ms")) for r in rows if to_float(r.get("delta_avg_delay_ms")) is not None]
    f1_deltas = [to_float(r.get("delta_avg_detection_f1")) for r in rows if to_float(r.get("delta_avg_detection_f1")) is not None]

    pdr_mean = mean(pdr_deltas)
    delay_mean = mean(delay_deltas)
    f1_mean = mean(f1_deltas)

    auc_base = auc_data.get(f"{args.baseline_label}_auc")
    auc_cand = auc_data.get(f"{args.candidate_label}_auc")
    auc_delta = auc_data.get("delta_auc")

    better_pdr = sum(1 for v in pdr_deltas if v > 0)
    better_delay = sum(1 for v in delay_deltas if v < 0)
    better_f1 = sum(1 for v in f1_deltas if v > 0)

    lines = []
    lines.append(f"# {args.baseline_label} vs {args.candidate_label} Comparison Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Summary source: {args.summary_csv}")
    lines.append(f"- AUC source: {args.auc_txt}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Mean delta PDR ({args.candidate_label} - {args.baseline_label}): {fmt(pdr_mean, 6)}")
    lines.append(f"- Mean delta Delay ms ({args.candidate_label} - {args.baseline_label}): {fmt(delay_mean, 6)}")
    lines.append(f"- Mean delta Detection F1 ({args.candidate_label} - {args.baseline_label}): {fmt(f1_mean, 6)}")
    lines.append(f"- AUC {args.baseline_label}: {fmt(auc_base, 6)}")
    lines.append(f"- AUC {args.candidate_label}: {fmt(auc_cand, 6)}")
    lines.append(f"- Delta AUC: {fmt(auc_delta, 6)}")
    lines.append("")

    lines.append("## Scenario Win Count")
    lines.append("")
    lines.append(f"- PDR improved scenarios: {better_pdr}/{len(rows)}")
    lines.append(f"- Delay improved scenarios (lower is better): {better_delay}/{len(rows)}")
    lines.append(f"- Detection F1 improved scenarios: {better_f1}/{len(rows)}")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if pdr_mean is not None and delay_mean is not None and f1_mean is not None:
        lines.append("- Candidate is stronger in network efficiency if PDR increases while Delay decreases.")
        lines.append("- Candidate is safer only if Detection F1 also improves.")
        lines.append("- Current result indicates a trade-off: throughput/latency gains vs detection performance drop.")
    else:
        lines.append("- Not enough metrics to produce full interpretation.")
    lines.append("")

    lines.append("## Scenario Table")
    lines.append("")
    lines.append("| Scenario | Delta PDR | Delta Delay (ms) | Delta Detection F1 |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.get('scenario','')} | {row.get('delta_avg_pdr','')} | {row.get('delta_avg_delay_ms','')} | {row.get('delta_avg_detection_f1','')} |"
        )
    lines.append("")

    plot_items = [
        ("Learning Curve", args.learning_curve),
        ("Evaluation Bar", args.eval_bar),
        ("ROC-AUC", args.roc_plot),
    ]
    if any(path for _, path in plot_items):
        lines.append("## Artifact Links")
        lines.append("")
        for name, path in plot_items:
            if path:
                lines.append(f"- {name}: {path}")
        lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    rows = read_summary(args.summary_csv)
    auc_data = read_auc(args.auc_txt)

    report_text = build_report(args, rows, auc_data)
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("=== Comparison Report Generated ===")
    print(f"Report: {args.output_md}")


if __name__ == "__main__":
    main()
