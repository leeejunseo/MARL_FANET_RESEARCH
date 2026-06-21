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
                writer.writerow(["episode", "total_reward", "avg_reward_per_drone"])

    def log_episode(self, episode, total_reward, num_drones):
        avg = total_reward / num_drones
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([episode, f"{total_reward:.4f}", f"{avg:.4f}"])

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
