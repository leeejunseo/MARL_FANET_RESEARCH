"""XAI 모듈 — Actor 네트워크 기반 특성 기여도 분석."""

import numpy as np
import torch


FEATURE_NAMES = [
    "SNR (dB)", "Trust Score", "Hop Count",
    "Packet Loss", "Node Degree", "Relative Velocity",
]


def build_extended_observations(n_samples=500, seed=42):
    """라우팅 의사결정용 확장 관측 벡터 생성."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n_samples, len(FEATURE_NAMES))).astype(np.float32)


class RoutingSurrogate(torch.nn.Module):
    """XAI 분석용 라우팅 의사결정 서로게이트 네트워크."""

    def __init__(self, input_dim=6, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_surrogate(model, observations, seed=42):
    """Trust Score·SNR 기반 라벨로 서로게이트 학습."""
    rng = np.random.default_rng(seed)
    snr, trust, hop, loss, degree, vel = observations.T
    labels = (
        0.35 * snr + 0.40 * trust
        - 0.10 * hop - 0.08 * loss
        + 0.05 * degree - 0.02 * vel
        + rng.normal(0, 0.03, len(observations))
    )
    labels = np.clip(labels, 0, 1).astype(np.float32)

    x = torch.FloatTensor(observations)
    y = torch.FloatTensor(labels).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        pred = model(x)
        loss_val = loss_fn(pred, y)
        loss_val.backward()
        optimizer.step()
    return labels


def integrated_gradients(model, observations, baseline=None, steps=50):
    """Integrated Gradients로 특성별 기여도 산출."""
    if baseline is None:
        baseline = np.zeros_like(observations)

    model.eval()
    attributions = np.zeros_like(observations)

    for i in range(len(observations)):
        x = observations[i]
        base = baseline[i] if baseline.ndim > 1 else baseline
        scaled_inputs = np.linspace(base, x, steps)
        grads = []
        for inp in scaled_inputs:
            t = torch.FloatTensor(inp).unsqueeze(0).requires_grad_(True)
            out = model(t)
            out.backward()
            grads.append(t.grad.numpy().flatten())
        avg_grad = np.mean(grads, axis=0)
        attributions[i] = (x - base) * avg_grad

    return attributions


def compute_feature_importance(attributions):
    """특성별 평균 절대 기여도."""
    return np.mean(np.abs(attributions), axis=0)


def compute_correlation_heatmap(observations, labels):
    """특성 간 및 라벨 상관 히트맵 행렬."""
    data = np.column_stack([observations, labels])
    return np.corrcoef(data.T)
