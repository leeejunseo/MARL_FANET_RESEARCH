"""학습·평가 지표 CSV 로깅."""

import csv
import os


class MetricsLogger:
    def __init__(self, filepath="logs/training_rewards.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "total_reward",
                    "avg_reward_per_drone",
                    "avg_pdr",
                    "avg_delay_ms",
                    "avg_hop",
                    "avg_trust",
                    "avg_disconnect",
                    "avg_detection",
                ])

    def log_episode(
        self,
        episode,
        total_reward,
        num_drones,
        avg_pdr=None,
        avg_delay_ms=None,
        avg_hop=None,
        avg_trust=None,
        avg_disconnect=None,
        avg_detection=None,
    ):
        avg = total_reward / num_drones
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                f"{total_reward:.4f}",
                f"{avg:.4f}",
                f"{avg_pdr:.4f}" if avg_pdr is not None else "",
                f"{avg_delay_ms:.4f}" if avg_delay_ms is not None else "",
                f"{avg_hop:.4f}" if avg_hop is not None else "",
                f"{avg_trust:.4f}" if avg_trust is not None else "",
                f"{avg_disconnect:.4f}" if avg_disconnect is not None else "",
                f"{avg_detection:.4f}" if avg_detection is not None else "",
            ])

    @staticmethod
    def load_rewards(filepath="logs/training_rewards.csv"):
        import numpy as np
        if not os.path.exists(filepath):
            return None, None
        episodes, rewards = [], []
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["total_reward"]))
        if not episodes:
            return None, None
        return np.array(episodes), np.array(rewards)


class QMetricsLogger:
    def __init__(self, filepath="logs/training_q_values.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "algorithm",
                    "q_current_mean",
                    "q_target_mean",
                    "q_overestimation_gap",
                    "q_abs_td_error",
                    "q1_current_mean",
                    "q2_current_mean",
                    "q_disagreement_mean",
                    "critic_loss1_mean",
                    "critic_loss2_mean",
                    "actor_loss_mean",
                    "actor_update_ratio",
                ])

    def log_episode(
        self,
        episode,
        algorithm,
        q_current_mean=None,
        q_target_mean=None,
        q_overestimation_gap=None,
        q_abs_td_error=None,
        q1_current_mean=None,
        q2_current_mean=None,
        q_disagreement_mean=None,
        critic_loss1_mean=None,
        critic_loss2_mean=None,
        actor_loss_mean=None,
        actor_update_ratio=None,
    ):
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                algorithm,
                f"{q_current_mean:.6f}" if q_current_mean is not None else "",
                f"{q_target_mean:.6f}" if q_target_mean is not None else "",
                f"{q_overestimation_gap:.6f}" if q_overestimation_gap is not None else "",
                f"{q_abs_td_error:.6f}" if q_abs_td_error is not None else "",
                f"{q1_current_mean:.6f}" if q1_current_mean is not None else "",
                f"{q2_current_mean:.6f}" if q2_current_mean is not None else "",
                f"{q_disagreement_mean:.6f}" if q_disagreement_mean is not None else "",
                f"{critic_loss1_mean:.6f}" if critic_loss1_mean is not None else "",
                f"{critic_loss2_mean:.6f}" if critic_loss2_mean is not None else "",
                f"{actor_loss_mean:.6f}" if actor_loss_mean is not None else "",
                f"{actor_update_ratio:.6f}" if actor_update_ratio is not None else "",
            ])


class EvalMetricsLogger:
    def __init__(self, filepath="logs/eval_metrics.csv", overwrite=False):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if overwrite or not os.path.exists(filepath):
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "scenario",
                    "policy",
                    "total_reward",
                    "avg_pdr",
                    "avg_delay_ms",
                    "avg_hop",
                    "avg_trust",
                    "avg_disconnect",
                    "avg_detection",
                    "avg_detection_accuracy",
                    "avg_detection_precision",
                    "avg_detection_recall",
                    "avg_detection_f1",
                ])

    def log_episode(self, episode, stats, scenario="Default", policy="EMARL-XAI"):
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                scenario,
                policy,
                f"{stats['total_reward']:.4f}",
                f"{stats['avg_pdr']:.4f}",
                f"{stats['avg_delay_ms']:.4f}",
                f"{stats['avg_hop']:.4f}",
                f"{stats['avg_trust']:.4f}",
                f"{stats['avg_disconnect']:.4f}",
                f"{stats['avg_detection']:.4f}" if stats['avg_detection'] is not None else "",
                f"{stats['avg_detection_accuracy']:.4f}" if stats.get('avg_detection_accuracy') is not None else "",
                f"{stats['avg_detection_precision']:.4f}" if stats.get('avg_detection_precision') is not None else "",
                f"{stats['avg_detection_recall']:.4f}" if stats.get('avg_detection_recall') is not None else "",
                f"{stats['avg_detection_f1']:.4f}" if stats.get('avg_detection_f1') is not None else "",
            ])

    @staticmethod
    def load_eval_metrics(filepath="logs/eval_metrics.csv"):
        if not os.path.exists(filepath):
            return None
        metrics = []
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append({
                    "episode": int(row["episode"]),
                    "scenario": row.get("scenario", "Default"),
                    "avg_pdr": float(row["avg_pdr"]),
                    "avg_delay_ms": float(row["avg_delay_ms"]),
                    "avg_hop": float(row["avg_hop"]),
                    "avg_trust": float(row["avg_trust"]),
                    "avg_disconnect": float(row["avg_disconnect"]),
                    "avg_detection": float(row["avg_detection"]) if row["avg_detection"] else None,
                    "avg_detection_accuracy": float(row["avg_detection_accuracy"]) if row.get("avg_detection_accuracy") else None,
                    "avg_detection_precision": float(row["avg_detection_precision"]) if row.get("avg_detection_precision") else None,
                    "avg_detection_recall": float(row["avg_detection_recall"]) if row.get("avg_detection_recall") else None,
                    "avg_detection_f1": float(row["avg_detection_f1"]) if row.get("avg_detection_f1") else None,
                })
        return metrics
