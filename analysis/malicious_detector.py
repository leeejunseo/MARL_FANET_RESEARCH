"""악의적 노드 탐지 모듈 — ROC/AUC 평가용."""

import numpy as np


FEATURE_NAMES = ["SNR", "Trust Score", "Hop Count", "Packet Drop Rate", "Latency (ms)"]


def _roc_curve(y_true, y_score):
    """표준 ROC 곡선 계산 (모든 임계값 기준 누적 TPR/FPR)."""
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    tpr = np.concatenate([[0.0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0.0], fp / max(fp[-1], 1)])
    return fpr, tpr


def _auc(fpr, tpr):
    try:
        return float(np.trapz(tpr, fpr))
    except AttributeError:
        return float(((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]) / 2.0).sum())


class MaliciousNodeDetector:
    """Trust Score·SNR·Hop Count 기반 악의적 노드 이상 점수 산출."""

    def __init__(self, weights=None, temperature=5.0, bias=0.40):
        self.weights = (
            np.array([0.15, 0.45, 0.15, 0.15, 0.10])
            if weights is None else np.array(weights)
        )
        self.temperature = temperature
        self.bias = bias

    def compute_anomaly_score(self, features):
        snr, trust, hop, drop, latency = features.T
        snr_norm = np.clip(snr / 30.0, 0, 1)
        trust_norm = np.clip(trust, 0, 1)
        hop_norm = np.clip(hop / 10.0, 0, 1)
        drop_norm = np.clip(drop, 0, 1)
        lat_norm = np.clip(latency / 200.0, 0, 1)

        return (
            self.weights[0] * (1 - snr_norm)
            + self.weights[1] * (1 - trust_norm)
            + self.weights[2] * hop_norm
            + self.weights[3] * drop_norm
            + self.weights[4] * lat_norm
        )

    def predict_proba(self, features):
        scores = self.compute_anomaly_score(features)
        return 1 / (1 + np.exp(-self.temperature * (scores - self.bias)))


def generate_node_dataset(n_normal=400, n_malicious=100, seed=42):
    """정상/악의적 노드 특성 시뮬레이션 (의도적 중첩 포함)."""
    rng = np.random.default_rng(seed)

    normal = np.column_stack([
        rng.normal(19, 6, n_normal),
        rng.beta(6, 3, n_normal),
        rng.poisson(3, n_normal),
        rng.beta(2, 15, n_normal),
        rng.normal(45, 20, n_normal),
    ])

    malicious = np.column_stack([
        rng.normal(13, 6, n_malicious),
        rng.beta(3, 6, n_malicious),
        rng.poisson(5, n_malicious) + 1,
        rng.beta(4, 5, n_malicious),
        rng.normal(95, 40, n_malicious),
    ])

    features = np.vstack([normal, malicious])
    labels = np.array([0] * n_normal + [1] * n_malicious, dtype=np.int32)
    return features, labels


def apply_model_degradation(scores, labels, flip_rate, noise_std, seed):
    """모델별 오분류·노이즈 주입으로 성능 차이 반영."""
    rng = np.random.default_rng(seed)
    degraded = scores.copy()

    for i, label in enumerate(labels):
        if rng.random() < flip_rate:
            degraded[i] = rng.uniform(0.55, 0.95) if label == 0 else rng.uniform(0.05, 0.45)

    degraded += rng.normal(0, noise_std, len(degraded))
    return np.clip(degraded, 0, 1)


def evaluate_roc(detector, features, labels, degradation=None):
    y_score = detector.predict_proba(features)
    if degradation:
        y_score = apply_model_degradation(
            y_score, labels,
            flip_rate=degradation["flip_rate"],
            noise_std=degradation["noise_std"],
            seed=degradation["seed"],
        )
    fpr, tpr = _roc_curve(labels, y_score)
    return fpr, tpr, _auc(fpr, tpr)
