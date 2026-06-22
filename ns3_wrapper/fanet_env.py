import numpy as np
import gymnasium as gym

class AdvancedFANETEnv(gym.Env):
    def __init__(
        self,
        num_drones=3,
        R_c=300.0,
        d_safe=30.0,
        max_pos=1000.0,
        max_vel=20.0,
        malicious_ratio=0.0,
        malicious_drop_rate=0.4,
        malicious_behavior="drop_and_trust",
        trust_noise=0.05,
    ):
        super(AdvancedFANETEnv, self).__init__()
        self.num_drones = num_drones
        self.max_pos = max_pos
        self.max_vel = max_vel
        self.R_c = R_c
        self.d_safe = d_safe
        self.obs_dim = 9
        self.state_dim = self.obs_dim * self.num_drones
        self.action_dim = 3

        self.malicious_ratio = malicious_ratio
        self.malicious_drop_rate = malicious_drop_rate
        self.malicious_behavior = malicious_behavior
        self.trust_noise = trust_noise
        self.malicious_ids = []
        self.step_count = 0
        self.current_trust_scores = np.zeros(self.num_drones)
        self.current_node_snr = np.zeros(self.num_drones)
        self.current_hop_counts = np.ones(self.num_drones)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.positions = np.random.uniform(450.0, 550.0, (self.num_drones, 3))
        self.velocities = np.zeros((self.num_drones, 3))
        self.step_count = 0
        self.malicious_ids = []

        if self.malicious_ratio > 0.0:
            n_malicious = max(1, int(self.num_drones * self.malicious_ratio))
            self.malicious_ids = list(np.random.choice(self.num_drones, size=n_malicious, replace=False))

        self._update_static_metrics()
        return self._get_obs(), self._get_global_state()

    def step(self, actions):
        self.step_count += 1
        actions = np.clip(actions, -1.0, 1.0)
        self.velocities += actions * 2.0
        self.velocities = np.clip(self.velocities, -self.max_vel, self.max_vel)
        self.positions += self.velocities
        self.positions = np.clip(self.positions, 0.0, self.max_pos)

        rewards = np.zeros(self.num_drones)
        distances = self._compute_distance_matrix()
        connected = distances <= self.R_c
        hop_matrix = self._compute_hop_matrix(connected)
        total_pairs = self.num_drones * (self.num_drones - 1)
        delivered_links = 0.0
        disconnect_count = 0
        node_latencies = np.zeros(self.num_drones)
        node_drop_rates = np.zeros(self.num_drones)
        node_snr = np.zeros(self.num_drones)
        trust_scores = np.zeros(self.num_drones)

        for i in range(self.num_drones):
            delays = []
            drops = []
            for j in range(self.num_drones):
                if i == j:
                    continue
                if connected[i, j]:
                    is_malicious_path = i in self.malicious_ids or j in self.malicious_ids
                    delivery = 1.0 - (self.malicious_drop_rate if is_malicious_path else 0.0)
                    delay_ms = 5.0 + (distances[i, j] / self.R_c) * 45.0

                    if is_malicious_path:
                        if self.malicious_behavior == "blackhole":
                            delivery = 0.0
                            delay_ms = 300.0
                        elif self.malicious_behavior == "selective_forwarding":
                            if np.random.rand() < self.malicious_drop_rate:
                                delivery = 0.0
                                delay_ms = 300.0
                        elif self.malicious_behavior == "sybil":
                            delivery = max(0.0, delivery - 0.25)
                            delay_ms += 40.0
                        else:
                            delay_ms += 20.0

                    delivered_links += delivery
                    delays.append(delay_ms)
                    drops.append(1.0 - delivery)
                else:
                    disconnect_count += 1
                    delays.append(300.0)
                    drops.append(1.0)

            avg_dist = np.mean(distances[i, np.arange(self.num_drones) != i])
            trust_penalty = 0.0
            if i in self.malicious_ids:
                if self.malicious_behavior == "blackhole":
                    trust_penalty = 0.40
                elif self.malicious_behavior == "selective_forwarding":
                    trust_penalty = 0.30
                elif self.malicious_behavior == "sybil":
                    trust_penalty = 0.45
                else:
                    trust_penalty = 0.25

            trust = np.clip(
                1.0 - (avg_dist / self.R_c) * 0.8
                - trust_penalty
                + np.random.normal(0.0, self.trust_noise),
                0.0,
                1.0,
            )
            trust_scores[i] = trust
            node_latencies[i] = np.mean(delays)
            node_drop_rates[i] = np.mean(drops)
            node_snr[i] = np.clip(30.0 - avg_dist * 0.025, 0.0, 30.0)

        self.current_trust_scores = trust_scores
        self.current_node_snr = node_snr
        self.current_hop_counts = np.clip(np.mean(hop_matrix, axis=1), 1.0, 10.0)

        valid_hops = hop_matrix[np.isfinite(hop_matrix) & (hop_matrix > 0)]
        avg_hop = float(np.mean(valid_hops)) if len(valid_hops) > 0 else float(self.num_drones)
        pdr = float(delivered_links / total_pairs)
        avg_delay_ms = float(np.mean(node_latencies))
        disconnect_ratio = float(disconnect_count / total_pairs)

        node_features = np.column_stack([
            node_snr,
            trust_scores,
            np.maximum(1.0, np.mean(hop_matrix, axis=1)),
            node_drop_rates,
            node_latencies,
        ])
        node_labels = np.array([1 if i in self.malicious_ids else 0 for i in range(self.num_drones)], dtype=int)

        for i in range(self.num_drones):
            r_cov = 0.0
            r_conn = 0.0
            r_col = 0.0
            for j in range(self.num_drones):
                if i == j:
                    continue
                dist = distances[i, j]
                r_cov += dist / self.max_pos
                if dist < self.d_safe:
                    r_col -= 5.0 * (1.0 - (dist / self.d_safe))
                if dist > self.R_c:
                    r_conn -= 2.0 * ((dist - self.R_c) / self.max_pos) ** 2
            rewards[i] = (
                1.0 * r_cov
                + 2.5 * r_col
                + 3.0 * r_conn
                + 0.9 * trust_scores[i]
                - 0.5 * (1.0 - trust_scores[i])
            )

        info = {
            "pdr": pdr,
            "avg_hop": avg_hop,
            "avg_delay_ms": avg_delay_ms,
            "trust_scores": trust_scores.tolist(),
            "disconnect_ratio": disconnect_ratio,
            "malicious_ids": self.malicious_ids,
            "node_features": node_features,
            "node_labels": node_labels,
        }

        terminated = False
        return self._get_obs(), self._get_global_state(), rewards, terminated, info

    def _compute_distance_matrix(self):
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def _compute_hop_matrix(self, adjacency):
        n = adjacency.shape[0]
        hops = np.full((n, n), np.inf, dtype=float)
        np.fill_diagonal(hops, 0.0)
        hops[adjacency] = 1.0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    hops[i, j] = min(hops[i, j], hops[i, k] + hops[k, j])
        return hops

    def _describe_behavior(self):
        if self.malicious_behavior == "blackhole":
            return "Blackhole"
        if self.malicious_behavior == "selective_forwarding":
            return "Selective Forwarding"
        if self.malicious_behavior == "sybil":
            return "Sybil"
        return "Drop and Trust"

    def _update_static_metrics(self):
        distances = self._compute_distance_matrix()
        connected = distances <= self.R_c
        hop_matrix = self._compute_hop_matrix(connected)
        node_snr = np.zeros(self.num_drones)
        trust_scores = np.zeros(self.num_drones)

        for i in range(self.num_drones):
            avg_dist = np.mean(distances[i, np.arange(self.num_drones) != i])
            trust_scores[i] = np.clip(
                1.0 - (avg_dist / self.R_c) * 0.8
                - (0.25 if i in self.malicious_ids else 0.0)
                + np.random.normal(0.0, self.trust_noise),
                0.0,
                1.0,
            )
            node_snr[i] = np.clip(30.0 - avg_dist * 0.025, 0.0, 30.0)

        self.current_trust_scores = trust_scores
        self.current_node_snr = node_snr
        self.current_hop_counts = np.clip(np.mean(hop_matrix, axis=1), 1.0, 10.0)

    def _get_obs(self):
        obs = []
        for i in range(self.num_drones):
            norm_pos = self.positions[i] / self.max_pos
            norm_vel = self.velocities[i] / self.max_vel
            norm_trust = np.array([self.current_trust_scores[i]], dtype=np.float32)
            norm_snr = np.array([self.current_node_snr[i] / 30.0], dtype=np.float32)
            norm_hop = np.array([self.current_hop_counts[i] / 10.0], dtype=np.float32)
            obs.append(np.concatenate([norm_pos, norm_vel, norm_trust, norm_snr, norm_hop]))
        return np.array(obs, dtype=np.float32)

    def _get_global_state(self):
        return self._get_obs().flatten()
