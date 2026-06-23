import os
import numpy as np
import torch
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from analysis.malicious_detector import MaliciousNodeDetector
from utils.tacview_logger import TacviewLogger
from utils.metrics_logger import EvalMetricsLogger
from utils.config import load_config
from utils.model_metrics import accuracy_score, precision_score, recall_score, f1_score


def infer_actor_obs_dim_from_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    for key, value in state_dict.items():
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.shape[1]
    raise ValueError(f"Cannot infer actor obs_dim from checkpoint: {path}")


def test_inference():
    config = load_config()
    env_cfg = config["environment"]
    eval_cfg = config["evaluation"]

    print("=== 완성된 신경망 기반 전술 기동 평가 시작 ===")
    num_drones = env_cfg["num_drones"]
    env = AdvancedFANETEnv(
        num_drones=num_drones,
        R_c=env_cfg["R_c"],
        d_safe=env_cfg["d_safe"],
        max_pos=env_cfg["max_pos"],
        max_vel=env_cfg["max_vel"],
        malicious_ratio=env_cfg.get("malicious_ratio", 0.0),
        malicious_drop_rate=env_cfg.get("malicious_drop_rate", 0.4),
        trust_noise=env_cfg.get("trust_noise", 0.05),
    )

    model_prefix = eval_cfg["actor_model_prefix"]
    episode_to_load = eval_cfg["model_episode"]
    agents = []
    for i in range(num_drones):
        path = model_prefix.format(i=i, episode=episode_to_load)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Actor 모델을 찾을 수 없습니다: {path}")
        actor_obs_dim = infer_actor_obs_dim_from_checkpoint(path)
        agent = MADDPGAgent(
            env.obs_dim,
            env.state_dim,
            env.action_dim,
            num_drones,
            agent_id=i,
            actor_obs_dim=actor_obs_dim,
        )
        agent.actor.load_state_dict(torch.load(path, map_location="cpu"))
        agents.append(agent)

    print(f"[성공] {episode_to_load} 에피소드 학습 모델 로드 완료")

    obs, state = env.reset(seed=42)
    tacview = TacviewLogger("logs/swarm_test_eval.acmi")
    logger = EvalMetricsLogger("logs/test_metrics.csv")
    detector = MaliciousNodeDetector()

    print("순수 전술 기동 평가 중...")
    pdr_values = []
    delay_values = []
    hop_values = []
    trust_values = []
    disconnect_values = []
    detection_rates = []
    step_labels = []
    step_predictions = []
    episode_reward = 0.0

    for step in range(eval_cfg["max_steps"]):
        actions = []
        for i in range(num_drones):
            action = agents[i].act(obs[i], explore=False)
            actions.append(action)

        next_obs, next_state, rewards, terminated, info = env.step(np.array(actions))
        episode_reward += float(np.sum(rewards))
        tacview.log_step(float(step), env.positions)

        pdr_values.append(info.get("pdr", 0.0))
        delay_values.append(info.get("avg_delay_ms", 0.0))
        hop_values.append(info.get("avg_hop", 0.0))
        trust_values.append(np.mean(info.get("trust_scores", [0.0])))
        disconnect_values.append(info.get("disconnect_ratio", 0.0))

        if info.get("node_features") is not None and info.get("node_labels") is not None:
            features = info["node_features"]
            labels = info["node_labels"]
            proba = detector.predict_proba(features)
            predicted = (proba >= eval_cfg.get("detection_threshold", 0.5)).astype(int)
            detection_rates.append(np.mean(predicted == labels))
            step_labels.append(labels)
            step_predictions.append(predicted)

        obs = next_obs
        state = next_state
        if terminated:
            break

    all_step_labels = np.concatenate(step_labels) if step_labels else None
    all_step_predictions = np.concatenate(step_predictions) if step_predictions else None

    detection_accuracy = (
        float(accuracy_score(all_step_labels, all_step_predictions))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )
    detection_precision = (
        float(precision_score(all_step_labels, all_step_predictions, average="binary", pos_label=1))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )
    detection_recall = (
        float(recall_score(all_step_labels, all_step_predictions, average="binary", pos_label=1))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )
    detection_f1 = (
        float(f1_score(all_step_labels, all_step_predictions, average="binary", pos_label=1))
        if all_step_labels is not None and all_step_predictions is not None
        else None
    )

    stats = {
        "total_reward": episode_reward,
        "avg_pdr": float(np.mean(pdr_values)) if pdr_values else 0.0,
        "avg_delay_ms": float(np.mean(delay_values)) if delay_values else 0.0,
        "avg_hop": float(np.mean(hop_values)) if hop_values else 0.0,
        "avg_trust": float(np.mean(trust_values)) if trust_values else 0.0,
        "avg_disconnect": float(np.mean(disconnect_values)) if disconnect_values else 0.0,
        "avg_detection": float(np.mean(detection_rates)) if detection_rates else None,
        "avg_detection_accuracy": detection_accuracy,
        "avg_detection_precision": detection_precision,
        "avg_detection_recall": detection_recall,
        "avg_detection_f1": detection_f1,
    }
    logger.log_episode(1, stats)
    print("=== 평가 종료: logs/swarm_test_eval.acmi 파일 추출 완료 ===")
    print("=== 테스트 메트릭 CSV 저장 완료: logs/test_metrics.csv ===")


if __name__ == "__main__":
    test_inference()
