"""Utilities for algorithm naming and artifact path separation."""

import os


def normalize_algorithm(name):
    return str(name or "maddpg").strip().lower()


def display_algorithm(name):
    algo = normalize_algorithm(name)
    if algo == "matd3":
        return "MATD3"
    return "MADDPG"


def resolve_model_dir(base_dir, algorithm):
    algo = normalize_algorithm(algorithm)
    if algo == "maddpg":
        return base_dir
    return os.path.join(base_dir, algo)


def resolve_file_path(base_path, algorithm):
    algo = normalize_algorithm(algorithm)
    if algo == "maddpg":
        return base_path

    root, ext = os.path.splitext(base_path)
    if root.endswith(f"_{algo}"):
        return base_path
    return f"{root}_{algo}{ext}"


def resolve_actor_prefix(base_prefix, algorithm):
    algo = normalize_algorithm(algorithm)
    if algo == "maddpg":
        return base_prefix

    parent = os.path.dirname(base_prefix)
    name = os.path.basename(base_prefix)
    return os.path.join(parent, algo, name)
