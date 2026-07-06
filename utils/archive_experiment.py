"""Archive experiment artifacts to a stable, algorithm-tagged directory."""

import argparse
import glob
import os
import shutil
from datetime import datetime


DEFAULT_PATTERNS = [
    "training_rewards.csv",
    "training_rewards_*.csv",
    "eval_metrics.csv",
    "eval_metrics_*.csv",
    "test_metrics.csv",
    "test_metrics_*.csv",
    "eval_node_features.npz",
    "eval_node_features_*.npz",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Archive logs for later algorithm comparison.")
    parser.add_argument("--algorithm", default="maddpg", help="Algorithm tag (e.g., maddpg, matd3)")
    parser.add_argument("--source-dir", default="logs", help="Directory containing experiment artifacts")
    parser.add_argument("--dest-root", default="logs/experiments", help="Root directory for archives")
    parser.add_argument("--run-tag", default=None, help="Optional run tag. Default: YYYYMMDD_HHMMSS")
    parser.add_argument("--include-config", action="store_true", help="Copy config.yaml into archive")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src, dst):
    if not os.path.exists(src):
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True


def main():
    args = parse_args()
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(args.dest_root, f"{args.algorithm}_{run_tag}")
    ensure_dir(archive_dir)

    copied = []
    for pattern in DEFAULT_PATTERNS:
        src_pattern = os.path.join(args.source_dir, pattern)
        matches = sorted(glob.glob(src_pattern))
        for src in matches:
            name = os.path.basename(src)
            dst = os.path.join(archive_dir, name)
            if copy_if_exists(src, dst):
                copied.append(name)

    if args.include_config:
        if copy_if_exists("config.yaml", os.path.join(archive_dir, "config.yaml")):
            copied.append("config.yaml")

    print("=== Experiment Archive Completed ===")
    print(f"Archive directory: {archive_dir}")
    if copied:
        print("Copied files:")
        for name in copied:
            print(f"  - {name}")
    else:
        print("No matching files found. Check source-dir and generated logs.")


if __name__ == "__main__":
    main()
