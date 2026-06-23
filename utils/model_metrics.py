"""모델 평가 지표 및 교차 검증 유틸리티.

이 모듈은 분류/회귀 성능 지표, 데이터 분할, K-fold 교차 검증,
그리고 Ridge/Lasso 정규화를 위한 간단한 회귀 모델을 제공합니다.
"""

import math
import numpy as np


def train_test_split(X, y=None, test_size=0.2, random_state=None, shuffle=True):
    """데이터를 Train/Validation(Test)으로 분할합니다."""
    X = np.asarray(X)
    n_samples = X.shape[0]
    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != n_samples:
            raise ValueError("X와 y의 길이가 일치하지 않습니다.")

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    n_test = int(math.floor(n_samples * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    if y is None:
        return X_train, X_test
    return X_train, X_test, y[train_idx], y[test_idx]


def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true와 y_pred의 shape이 일치해야 합니다.")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_index[true], label_to_index[pred]] += 1
    return matrix


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true와 y_pred의 shape이 일치해야 합니다.")
    return float(np.mean(y_true == y_pred))


def _precision_recall_f1_for_label(y_true, y_pred, positive_label):
    tp = int(np.sum((y_pred == positive_label) & (y_true == positive_label)))
    fp = int(np.sum((y_pred == positive_label) & (y_true != positive_label)))
    fn = int(np.sum((y_pred != positive_label) & (y_true == positive_label)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        precision, _, _ = _precision_recall_f1_for_label(y_true, y_pred, pos_label)
        return precision
    scores = []
    for label in labels:
        precision, _, _ = _precision_recall_f1_for_label(y_true, y_pred, label)
        scores.append(precision)
    return float(np.mean(scores))


def recall_score(y_true, y_pred, average="binary", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        _, recall, _ = _precision_recall_f1_for_label(y_true, y_pred, pos_label)
        return recall
    scores = []
    for label in labels:
        _, recall, _ = _precision_recall_f1_for_label(y_true, y_pred, label)
        scores.append(recall)
    return float(np.mean(scores))


def f1_score(y_true, y_pred, average="binary", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        _, _, f1 = _precision_recall_f1_for_label(y_true, y_pred, pos_label)
        return f1
    scores = []
    for label in labels:
        _, _, f1 = _precision_recall_f1_for_label(y_true, y_pred, label)
        scores.append(f1)
    return float(np.mean(scores))


def roc_curve(y_true, y_score, pos_label=1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true와 y_score의 길이가 일치하지 않습니다.")

    desc = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc]
    y_score_sorted = y_score[desc]
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == pos_label)[threshold_idxs]
    fps = np.cumsum(y_true_sorted != pos_label)[threshold_idxs]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fn = tps[-1] - tps
    tn = fps[-1] - fps
    tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps, dtype=float)
    return fpr, tpr, np.r_[threshold_idxs, threshold_idxs[-1]]


def auc(fpr, tpr):
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr와 tpr의 shape이 일치해야 합니다.")
    return float(np.trapz(tpr, fpr))


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def kfold_split(n_samples, n_splits=5, shuffle=True, random_state=None):
    if n_splits < 2:
        raise ValueError("n_splits는 2 이상이어야 합니다.")
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        yield train_idx, test_idx
        current = stop


def cross_val_score(estimator, X, y, scoring="accuracy", cv=5, **fit_params):
    X = np.asarray(X)
    y = np.asarray(y)
    scores = []
    for train_idx, test_idx in kfold_split(len(X), n_splits=cv, shuffle=True, random_state=fit_params.pop("random_state", None)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        estimator.fit(X_train, y_train, **fit_params)
        y_pred = estimator.predict(X_test)
        if scoring == "accuracy":
            scores.append(accuracy_score(y_test, y_pred))
        elif scoring == "precision":
            scores.append(precision_score(y_test, y_pred, average="binary"))
        elif scoring == "recall":
            scores.append(recall_score(y_test, y_pred, average="binary"))
        elif scoring == "f1":
            scores.append(f1_score(y_test, y_pred, average="binary"))
        elif scoring == "mse":
            scores.append(mean_squared_error(y_test, y_pred))
        elif scoring == "mae":
            scores.append(mean_absolute_error(y_test, y_pred))
        else:
            raise ValueError(f"지원하지 않는 scoring: {scoring}")
    return np.array(scores)


class RidgeRegressor:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(n_features, dtype=float)
            y_mean = 0.0

        A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
        b = X_centered.T @ y_centered
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = y_mean - X_mean @ self.coef_ if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def penalty(self):
        return float(self.alpha * np.sum(self.coef_ ** 2))


class LassoRegressor:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0

    def _soft_threshold(self, rho, alpha):
        if rho < -alpha:
            return rho + alpha
        if rho > alpha:
            return rho - alpha
        return 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X = X - X_mean
            y = y - y_mean
        else:
            X_mean = np.zeros(n_features, dtype=float)
            y_mean = 0.0

        self.coef_ = np.zeros(n_features, dtype=float)
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                residual = y - X @ self.coef_ + self.coef_[j] * X[:, j]
                rho = np.dot(X[:, j], residual)
                self.coef_[j] = self._soft_threshold(rho, self.alpha) / np.sum(X[:, j] ** 2)
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        self.intercept_ = y_mean - X_mean @ self.coef_ if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def penalty(self):
        return float(self.alpha * np.sum(np.abs(self.coef_)))


if __name__ == "__main__":
    print("model_metrics 유틸리티가 정상적으로 로드되었습니다.")
