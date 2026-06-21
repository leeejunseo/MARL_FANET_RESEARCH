"""악의적 노드 탐지 모듈 — ROC/AUC 평가용."""

import numpy as np


FEATURE_NAMES = ["SNR", "Trust Score", "Hop Count", "Packet Drop Rate", "Latency (ms)"]


def _roc_curve(y_true, y_score):
    """sklearn 없이 ROC 곡선 계산."""
    desc_idx = np.argsort(y_score)[::-1]
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_score) - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    thresholds = np.r_[y_score[0] + 1, y_score[threshold_idxs]]
    return fpr, tpr, thresholds


def _auc(fpr, tpr):
    return np.trapz(tpr, fpr)


class MaliciousNodeDetector:
    """Trust Score·SNR·Hop Count 기반 악의적 노드 이상 점수 산출."""

    def __init__(self, weights=None):
        # EMARL-XAI: Trust Score에 높은 가중치
        self.weights = (
            np.array([0.15, 0.45, 0.15, 0.15, 0.10])
            if weights is None else np.array(weights)
        )

    def compute_anomaly_score(self, features):
        """features: (N, 5) — [SNR, Trust, Hop, DropRate, Latency]"""
        snr, trust, hop, drop, latency = features.T
        snr_norm = np.clip(snr / 30.0, 0, 1)
        trust_norm = np.clip(trust, 0, 1)
        hop_norm = np.clip(hop / 10.0, 0, 1)
        drop_norm = np.clip(drop, 0, 1)
        lat_norm = np.clip(latency / 200.0, 0, 1)

        # 악의적 노드일수록 점수가 높아지도록 설계
        score = (
            self.weights[0] * (1 - snr_norm)
            + self.weights[1] * (1 - trust_norm)
            + self.weights[2] * hop_norm
            + self.weights[3] * drop_norm
            + self.weights[4] * lat_norm
        )
        return score

    def predict_proba(self, features, noise_std=0.0):
        scores = self.compute_anomaly_score(features)
        proba = 1 / (1 + np.exp(-8 * (scores - 0.45)))
        if noise_std > 0:
            rng = np.random.default_rng(42)
            proba = np.clip(proba + rng.normal(0, noise_std, len(proba)), 0, 1)
        return proba


def generate_node_dataset(n_normal=400, n_malicious=100, seed=42):
    """정상/악의적 노드 특성 시뮬레이션 데이터 생성."""
    rng = np.random.default_rng(seed)

    # 정상 노드: 높은 SNR·Trust, 낮은 Drop Rate
    normal = np.column_stack([
        rng.normal(22, 3, n_normal),       # SNR (dB)
        rng.beta(8, 2, n_normal),          # Trust Score [0,1]
        rng.poisson(3, n_normal),          # Hop Count
        rng.beta(1, 20, n_normal),         # Packet Drop Rate
        rng.normal(35, 10, n_normal),      # Latency (ms)
    ])

    # 악의적 노드: 낮은 SNR·Trust, 높은 Drop Rate·Hop
    malicious = np.column_stack([
        rng.normal(8, 4, n_malicious),
        rng.beta(2, 8, n_malicious),
        rng.poisson(7, n_malicious) + 2,
        rng.beta(6, 3, n_malicious),
        rng.normal(120, 30, n_malicious),
    ])

    features = np.vstack([normal, malicious])
    labels = np.array([0] * n_normal + [1] * n_malicious)
    return features, labels


def evaluate_roc(detector, features, labels, noise_std=0.0):
    """ROC 곡선 좌표 및 AUC 반환."""
    y_score = detector.predict_proba(features, noise_std=noise_std)
    fpr, tpr, thresholds = _roc_curve(labels, y_score)
    roc_auc = _auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds
