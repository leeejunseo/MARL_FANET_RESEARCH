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
        velocity_damping=0.05,
        center_pull_coeff=0.12,
        center_reward_coeff=0.6,
        reward_cov_coeff=0.6,
        reward_col_coeff=2.5,
        reward_conn_coeff=4.0,
        reward_trust_pos_coeff=1.2,
        reward_trust_neg_coeff=0.8,
        trust_update_rate=0.2,
        trust_w_fr=0.5,
        trust_w_cr=0.3,
        trust_w_dr=0.2,
        trust_threshold=0.35,
        interference_k=1.2,
        interference_base=0.1,
        interference_distance_coeff=0.9,
        interference_malicious_boost=0.6,
        energy_init=100.0,
        energy_move_coeff=0.08,
        energy_tx_coeff=0.25,
        reward_alpha=1.2,
        reward_beta=1.0,
        reward_gamma=0.8,
        reward_delta=1.0,
        reward_w_pdr=1.0,
        reward_w_trust=1.0,
        reward_w_delay=1.0,
        reward_w_energy=0.8,
        reward_w_security=1.2,
        alert_decay=0.9,
        wall_bounce_coeff=0.6,
        wall_push_coeff=0.15,
        boundary_margin_ratio=0.08,
        boundary_avoid_accel=0.35,
        boundary_penalty_coeff=1.5,
        enable_boundary_shield=True,
        connectivity_guard_coeff=0.35,
        min_neighbor_target=2,
        malicious_avoid_coeff=0.65,
        suspicious_avoid_coeff=0.35,
        avoid_distance_factor=1.15,
    ):
        super(AdvancedFANETEnv, self).__init__()
        self.num_drones = num_drones
        self.max_pos = max_pos
        self.max_vel = max_vel
        self.R_c = R_c
        self.d_safe = d_safe
        # Keep first 9 dimensions unchanged for checkpoint compatibility.
        # [pos(3), vel(3), trust, snr, hop, energy, alert_history, comm_delay]
        self.obs_dim = 12
        self.state_dim = self.obs_dim * self.num_drones
        self.action_dim = 3

        self.malicious_ratio = malicious_ratio
        self.malicious_drop_rate = malicious_drop_rate
        self.malicious_behavior = malicious_behavior
        self.trust_noise = trust_noise
        self.velocity_damping = float(velocity_damping)
        self.center_pull_coeff = float(center_pull_coeff)
        self.center_reward_coeff = float(center_reward_coeff)
        self.reward_cov_coeff = float(reward_cov_coeff)
        self.reward_col_coeff = float(reward_col_coeff)
        self.reward_conn_coeff = float(reward_conn_coeff)
        self.reward_trust_pos_coeff = float(reward_trust_pos_coeff)
        self.reward_trust_neg_coeff = float(reward_trust_neg_coeff)
        self.trust_update_rate = float(trust_update_rate)
        self.trust_w_fr = float(trust_w_fr)
        self.trust_w_cr = float(trust_w_cr)
        self.trust_w_dr = float(trust_w_dr)
        self.trust_threshold = float(trust_threshold)
        self.interference_k = float(interference_k)
        self.interference_base = float(interference_base)
        self.interference_distance_coeff = float(interference_distance_coeff)
        self.interference_malicious_boost = float(interference_malicious_boost)
        self.energy_init = float(energy_init)
        self.energy_move_coeff = float(energy_move_coeff)
        self.energy_tx_coeff = float(energy_tx_coeff)
        self.reward_alpha = float(reward_alpha)
        self.reward_beta = float(reward_beta)
        self.reward_gamma = float(reward_gamma)
        self.reward_delta = float(reward_delta)
        self.reward_w_pdr = float(reward_w_pdr)
        self.reward_w_trust = float(reward_w_trust)
        self.reward_w_delay = float(reward_w_delay)
        self.reward_w_energy = float(reward_w_energy)
        self.reward_w_security = float(reward_w_security)
        self.alert_decay = float(alert_decay)
        self.wall_bounce_coeff = float(wall_bounce_coeff)
        self.wall_push_coeff = float(wall_push_coeff)
        self.boundary_margin_ratio = float(boundary_margin_ratio)
        self.boundary_avoid_accel = float(boundary_avoid_accel)
        self.boundary_penalty_coeff = float(boundary_penalty_coeff)
        self.enable_boundary_shield = bool(enable_boundary_shield)
        self.connectivity_guard_coeff = float(connectivity_guard_coeff)
        self.min_neighbor_target = int(min_neighbor_target)
        self.malicious_avoid_coeff = float(malicious_avoid_coeff)
        self.suspicious_avoid_coeff = float(suspicious_avoid_coeff)
        self.avoid_distance_factor = float(avoid_distance_factor)
        self.malicious_ids = []
        self.step_count = 0
        self.current_trust_scores = np.zeros(self.num_drones)
        self.current_node_snr = np.zeros(self.num_drones)
        self.current_hop_counts = np.ones(self.num_drones)
        self.current_node_delay = np.zeros(self.num_drones)
        self.current_energy = np.full(self.num_drones, self.energy_init)
        self.current_alert_history = np.zeros(self.num_drones)
        self.trust_matrix = np.full((self.num_drones, self.num_drones), 0.8, dtype=float)
        np.fill_diagonal(self.trust_matrix, 1.0)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.positions = self._sample_initial_positions()
        self.velocities = np.zeros((self.num_drones, 3))
        self.step_count = 0
        self.malicious_ids = []
        self.current_energy = np.full(self.num_drones, self.energy_init)
        self.current_alert_history = np.zeros(self.num_drones)
        self.trust_matrix = np.full((self.num_drones, self.num_drones), 0.8, dtype=float)
        np.fill_diagonal(self.trust_matrix, 1.0)

        if self.malicious_ratio > 0.0:
            n_malicious = max(1, int(self.num_drones * self.malicious_ratio))
            self.malicious_ids = list(np.random.choice(self.num_drones, size=n_malicious, replace=False))

        self._update_static_metrics()
        return self._get_obs(), self._get_global_state()

    def _sample_initial_positions(self):
        # Communication-first initialization:
        # spread enough for exploration, but keep most drones link-reachable.
        cx = np.random.uniform(self.max_pos * 0.4, self.max_pos * 0.6)
        cy = np.random.uniform(self.max_pos * 0.4, self.max_pos * 0.6)
        cz = np.random.uniform(self.max_pos * 0.45, self.max_pos * 0.55)
        center = np.array([cx, cy, cz], dtype=float)

        ring_radius = min(self.R_c * 0.33, self.max_pos * 0.14)
        z_jitter = min(self.max_pos * 0.03, self.R_c * 0.1)
        min_sep = max(self.d_safe * 1.15, self.R_c * 0.08)

        positions = []
        for i in range(self.num_drones):
            placed = False
            for _ in range(120):
                angle = (2.0 * np.pi * i / max(1, self.num_drones)) + np.random.uniform(-0.2, 0.2)
                radial = ring_radius + np.random.uniform(-0.18, 0.18) * ring_radius
                p = center + np.array(
                    [
                        radial * np.cos(angle),
                        radial * np.sin(angle),
                        np.random.uniform(-z_jitter, z_jitter),
                    ],
                    dtype=float,
                )
                p = np.clip(p, self.max_pos * 0.1, self.max_pos * 0.9)

                if not positions:
                    positions.append(p)
                    placed = True
                    break
                d = np.linalg.norm(np.array(positions) - p, axis=1)
                if np.all(d >= min_sep):
                    positions.append(p)
                    placed = True
                    break

            if not placed:
                positions.append(np.clip(center + np.random.uniform(-ring_radius, ring_radius, size=3), 0.0, self.max_pos))

        return np.array(positions, dtype=float)

    def step(self, actions):
        self.step_count += 1
        actions = np.clip(actions, -1.0, 1.0)

        # Safety shield near walls: suppress outward actions and add inward bias.
        if self.enable_boundary_shield:
            margin = max(1.0, self.max_pos * self.boundary_margin_ratio)
            for axis in range(3):
                coord = self.positions[:, axis]
                near_low = coord < margin
                near_high = coord > (self.max_pos - margin)

                # Block actions that push further outside near the walls.
                low_outward = near_low & (actions[:, axis] < 0.0)
                high_outward = near_high & (actions[:, axis] > 0.0)
                actions[low_outward, axis] = 0.0
                actions[high_outward, axis] = 0.0

                # Add a smooth inward push that grows closer to the wall.
                if np.any(near_low):
                    low_strength = 1.0 - np.clip(coord[near_low] / margin, 0.0, 1.0)
                    actions[near_low, axis] += self.boundary_avoid_accel * low_strength
                if np.any(near_high):
                    high_dist = (self.max_pos - coord[near_high])
                    high_strength = 1.0 - np.clip(high_dist / margin, 0.0, 1.0)
                    actions[near_high, axis] -= self.boundary_avoid_accel * high_strength

            actions = np.clip(actions, -1.0, 1.0)

        self.velocities += actions * 2.0

        # Suppress long-term drift: apply velocity damping and a mild pull toward map center.
        center = np.full(3, self.max_pos * 0.5, dtype=float)
        damping = np.clip(1.0 - self.velocity_damping, 0.0, 1.0)
        self.velocities *= damping
        self.velocities += (
            -self.center_pull_coeff * ((self.positions - center) / self.max_pos) * self.max_vel
        )

        # Security avoidance: push drones away from known malicious and low-trust neighbors.
        if self.malicious_avoid_coeff > 0.0 or self.suspicious_avoid_coeff > 0.0:
            avoid_radius = max(self.R_c, self.R_c * self.avoid_distance_factor)
            curr_diff = self.positions[:, None, :] - self.positions[None, :, :]
            curr_dist = np.linalg.norm(curr_diff, axis=-1)
            for i in range(self.num_drones):
                current_neighbors = int(np.sum(curr_dist[i] <= self.R_c) - 1)
                repel_vec = np.zeros(3, dtype=float)
                for j in range(self.num_drones):
                    if i == j:
                        continue

                    d_vec = self.positions[i] - self.positions[j]
                    dist = float(np.linalg.norm(d_vec))
                    if dist < 1e-6 or dist > avoid_radius:
                        continue

                    is_malicious = j in self.malicious_ids
                    is_suspicious = self.trust_matrix[i, j] < self.trust_threshold
                    if not is_malicious and not is_suspicious:
                        continue

                    if is_malicious:
                        weight = self.malicious_avoid_coeff
                        # Keep connectivity first when a node is at risk of isolation.
                        if current_neighbors < self.min_neighbor_target:
                            weight *= 0.45
                        if current_neighbors <= 0:
                            weight *= 0.30
                    else:
                        trust_gap = float(np.clip(self.trust_threshold - self.trust_matrix[i, j], 0.0, 1.0))
                        weight = self.suspicious_avoid_coeff * trust_gap
                        # Suspicious avoidance is softened under low-neighbor conditions.
                        if current_neighbors < self.min_neighbor_target:
                            weight *= 0.35
                        if current_neighbors <= 0:
                            weight *= 0.20

                    proximity = 1.0 - (dist / avoid_radius)
                    repel_vec += weight * proximity * (d_vec / dist)

                repel_norm = float(np.linalg.norm(repel_vec))
                if repel_norm > 1e-6:
                    self.velocities[i] += (repel_vec / repel_norm) * self.max_vel

        # Connectivity guard: if a drone has too few predicted neighbors,
        # nudge it back toward nearby teammates to keep communication links.
        if self.min_neighbor_target > 0 and self.connectivity_guard_coeff > 0.0:
            curr_diff = self.positions[:, None, :] - self.positions[None, :, :]
            curr_dist = np.linalg.norm(curr_diff, axis=-1)
            predicted = self.positions + self.velocities
            pred_diff = predicted[:, None, :] - predicted[None, :, :]
            pred_dist = np.linalg.norm(pred_diff, axis=-1)
            for i in range(self.num_drones):
                pred_neighbors = int(np.sum(pred_dist[i] <= self.R_c) - 1)
                curr_neighbors = int(np.sum(curr_dist[i] <= self.R_c) - 1)
                neighbors = min(pred_neighbors, curr_neighbors)
                if neighbors >= self.min_neighbor_target:
                    continue

                # Prefer trusted and non-malicious peers when reconnecting.
                trusted_normal_ids = [
                    j for j in range(self.num_drones)
                    if j != i and j not in self.malicious_ids and self.trust_matrix[i, j] >= (0.8 * self.trust_threshold)
                ]
                fallback_normal_ids = [
                    j for j in range(self.num_drones)
                    if j != i and j not in self.malicious_ids
                ]
                pool = trusted_normal_ids if trusted_normal_ids else fallback_normal_ids
                if not pool:
                    pool = [j for j in range(self.num_drones) if j != i]

                order = sorted(pool, key=lambda j: pred_dist[i, j])
                k = min(self.min_neighbor_target, self.num_drones - 1)
                close_ids = order[:k]
                if not close_ids:
                    continue

                # Use current positions as the attractor anchor to reduce overshoot.
                target = np.mean(self.positions[close_ids], axis=0)
                direction = target - self.positions[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    nearest = float(pred_dist[i, close_ids[0]])
                    stretch = max(0.0, (nearest - 0.9 * self.R_c) / max(self.R_c, 1e-6))
                    deficit = max(0.0, float(self.min_neighbor_target - neighbors))
                    pull_gain = self.connectivity_guard_coeff * (1.0 + 1.6 * stretch + 0.6 * deficit)
                    if curr_neighbors <= 0:
                        pull_gain += self.connectivity_guard_coeff
                    pull_gain = min(2.0, pull_gain)
                    self.velocities[i] += (
                        pull_gain
                        * (direction / norm)
                        * self.max_vel
                    )

        self.velocities = np.clip(self.velocities, -self.max_vel, self.max_vel)
        self.positions += self.velocities

        # Bounce back from map borders so agents do not stick to the walls.
        for axis in range(3):
            low_hits = self.positions[:, axis] < 0.0
            high_hits = self.positions[:, axis] > self.max_pos

            if np.any(low_hits):
                self.positions[low_hits, axis] = 0.0
                self.velocities[low_hits, axis] = np.abs(self.velocities[low_hits, axis]) * self.wall_bounce_coeff

            if np.any(high_hits):
                self.positions[high_hits, axis] = self.max_pos
                self.velocities[high_hits, axis] = -np.abs(self.velocities[high_hits, axis]) * self.wall_bounce_coeff

            # Soft push away from the borders even before a hard collision.
            near_low = (self.positions[:, axis] >= 0.0) & (self.positions[:, axis] < self.max_pos * 0.08)
            near_high = (self.positions[:, axis] <= self.max_pos) & (self.positions[:, axis] > self.max_pos * 0.92)
            self.velocities[near_low, axis] += self.wall_push_coeff * self.max_vel
            self.velocities[near_high, axis] -= self.wall_push_coeff * self.max_vel

        self.velocities = np.clip(self.velocities, -self.max_vel, self.max_vel)
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
        node_delivery_rates = np.zeros(self.num_drones)
        node_snr = np.zeros(self.num_drones)
        trust_scores = np.zeros(self.num_drones)
        security_risk = np.zeros(self.num_drones)
        fpr_proxy = np.zeros(self.num_drones)
        node_energy_cost = np.zeros(self.num_drones)
        node_tx_attempts = np.zeros(self.num_drones)

        # Alert memory decays over time.
        self.current_alert_history *= self.alert_decay

        # Mobility energy cost from velocity norm.
        move_cost = self.energy_move_coeff * np.linalg.norm(self.velocities, axis=1)
        self.current_energy = np.maximum(0.0, self.current_energy - move_cost)
        node_energy_cost += move_cost

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

                    # Physical link reliability under interference: P_succ = exp(-k * interference)
                    interference = self.interference_base + self.interference_distance_coeff * min(
                        1.0, distances[i, j] / max(self.R_c, 1e-6)
                    )
                    if i in self.malicious_ids or j in self.malicious_ids:
                        interference += self.interference_malicious_boost
                    p_succ = float(np.exp(-self.interference_k * interference))
                    delivery = float(np.clip(delivery * p_succ, 0.0, 1.0))

                    delivered_links += delivery
                    delays.append(delay_ms)
                    drops.append(1.0 - delivery)
                    node_tx_attempts[i] += 1.0

                    tx_cost = self.energy_tx_coeff * (1.0 + delay_ms / 300.0)
                    self.current_energy[i] = max(0.0, self.current_energy[i] - tx_cost)
                    node_energy_cost[i] += tx_cost
                else:
                    disconnect_count += 1
                    delays.append(300.0)
                    drops.append(1.0)
                    security_risk[i] += 0.05

            avg_dist = np.mean(distances[i, np.arange(self.num_drones) != i])
            node_latencies[i] = np.mean(delays)
            node_drop_rates[i] = np.mean(drops)
            node_delivery_rates[i] = 1.0 - node_drop_rates[i]
            node_snr[i] = np.clip(30.0 - avg_dist * 0.025, 0.0, 30.0)

        # Trust computation model:
        # T_ij(t+1) = (1-lambda) T_ij(t) + lambda * Psi_ij
        # Psi_ij = w1*FR_ij + w2*CR_ij + w3*DR_ij
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i == j:
                    self.trust_matrix[i, j] = 1.0
                    continue
                if connected[i, j]:
                    delivery_ij = max(0.0, 1.0 - node_drop_rates[i])
                    delay_ij = node_latencies[i]

                    fr_ij = delivery_ij
                    cr_ij = float(np.exp(-abs(delay_ij - node_latencies[i]) / 80.0))
                    dr_ij = 1.0 - (1.0 - delivery_ij)
                    psi_ij = (
                        self.trust_w_fr * fr_ij
                        + self.trust_w_cr * cr_ij
                        + self.trust_w_dr * dr_ij
                    )
                    psi_ij = float(np.clip(psi_ij, 0.0, 1.0))
                    self.trust_matrix[i, j] = (
                        (1.0 - self.trust_update_rate) * self.trust_matrix[i, j]
                        + self.trust_update_rate * psi_ij
                    )
                else:
                    self.trust_matrix[i, j] *= (1.0 - 0.5 * self.trust_update_rate)

                self.trust_matrix[i, j] = float(np.clip(self.trust_matrix[i, j], 0.0, 1.0))

        low_trust_edges = self.trust_matrix < self.trust_threshold
        np.fill_diagonal(low_trust_edges, False)

        # Low-trust neighbors are isolated in connectivity and add security risk.
        for i in range(self.num_drones):
            low_cnt = 0
            mild_low_cnt = 0
            for j in np.where(low_trust_edges[i])[0]:
                severe_low_trust = self.trust_matrix[i, j] < (0.6 * self.trust_threshold)
                if (j in self.malicious_ids) or severe_low_trust:
                    connected[i, j] = False
                    low_cnt += 1
                else:
                    # Fail-open for mild low-trust links to avoid over-fragmenting normal nodes.
                    mild_low_cnt += 1

            if low_cnt > 0:
                security_risk[i] += 0.10 * low_cnt
                self.current_alert_history[i] += float(low_cnt)
            if mild_low_cnt > 0:
                security_risk[i] += 0.03 * mild_low_cnt
                self.current_alert_history[i] += 0.3 * float(mild_low_cnt)

            malicious_neighbors = int(sum(1 for j in self.malicious_ids if j != i and connected[i, j]))
            if malicious_neighbors > 0:
                security_risk[i] += 0.15 * malicious_neighbors
                self.current_alert_history[i] += float(malicious_neighbors)

            false_alert_rate = self.current_alert_history[i] / max(1.0, self.step_count)
            fpr_proxy[i] = float(np.clip(false_alert_rate, 0.0, 1.0))

        # Node trust score used by detector/observation is the mean outgoing trust.
        trust_scores = np.mean(self.trust_matrix, axis=1)

        self.current_trust_scores = trust_scores
        self.current_node_snr = node_snr
        self.current_hop_counts = np.clip(np.mean(hop_matrix, axis=1), 1.0, 10.0)
        self.current_node_delay = node_latencies

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

            pdr_i = node_delivery_rates[i]
            delay_norm = node_latencies[i] / 300.0
            trust_i = trust_scores[i]
            energy_norm = node_energy_cost[i] / max(self.energy_init, 1e-6)
            security_i = security_risk[i]
            fpr_i = fpr_proxy[i]

            base_reward = (
                self.reward_alpha * pdr_i
                - self.reward_beta * delay_norm
                - self.reward_gamma * fpr_i
                + self.reward_delta * trust_i
            )

            detailed_reward = (
                self.reward_w_pdr * pdr_i
                + self.reward_w_trust * trust_i
                - self.reward_w_delay * delay_norm
                - self.reward_w_energy * energy_norm
                - self.reward_w_security * security_i
            )

            rewards[i] = (
                self.reward_cov_coeff * r_cov
                + self.reward_col_coeff * r_col
                + self.reward_conn_coeff * r_conn
                + self.reward_trust_pos_coeff * trust_scores[i]
                - self.reward_trust_neg_coeff * (1.0 - trust_scores[i])
                + base_reward
                + detailed_reward
            )

        centroid = np.mean(self.positions, axis=0)
        max_center_dist = np.linalg.norm(np.array([self.max_pos * 0.5] * 3, dtype=float))
        centroid_drift = float(np.linalg.norm(centroid - center) / max_center_dist)
        rewards -= self.center_reward_coeff * centroid_drift

        # Penalize staying too close to walls to avoid boundary hugging.
        margin = max(1.0, self.max_pos * self.boundary_margin_ratio)
        wall_clearance = np.min(
            np.minimum(self.positions, self.max_pos - self.positions),
            axis=1,
        )
        boundary_proximity = np.clip(1.0 - (wall_clearance / margin), 0.0, 1.0)
        rewards -= self.boundary_penalty_coeff * boundary_proximity

        # Extra penalty when a drone is almost stationary while hugging the wall.
        speed_norm = np.linalg.norm(self.velocities, axis=1) / max(self.max_vel, 1e-6)
        stuck_near_wall = (boundary_proximity > 0.7) & (speed_norm < 0.05)
        rewards[stuck_near_wall] -= 0.5

        info = {
            "pdr": pdr,
            "avg_hop": avg_hop,
            "avg_delay_ms": avg_delay_ms,
            "trust_scores": trust_scores.tolist(),
            "trust_matrix": self.trust_matrix.tolist(),
            "node_snr": node_snr.tolist(),
            "hop_counts": self.current_hop_counts.tolist(),
            "node_energy": self.current_energy.tolist(),
            "alert_history": self.current_alert_history.tolist(),
            "disconnect_ratio": disconnect_ratio,
            "connected_matrix": connected.astype(bool),
            "centroid": centroid.tolist(),
            "centroid_drift": centroid_drift,
            "avg_boundary_proximity": float(np.mean(boundary_proximity)),
            "wall_hugging_count": int(np.sum(boundary_proximity > 0.7)),
            "malicious_ids": self.malicious_ids,
            "node_features": node_features,
            "node_labels": node_labels,
            "link_source": "distance_model",
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
        self.current_node_delay = np.full(self.num_drones, 50.0)

        for i in range(self.num_drones):
            avg_dist = np.mean(distances[i, np.arange(self.num_drones) != i])
            trust_scores[i] = float(np.mean(self.trust_matrix[i]))
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
            norm_energy = np.array([self.current_energy[i] / max(self.energy_init, 1e-6)], dtype=np.float32)
            norm_alert = np.array([min(1.0, self.current_alert_history[i] / 10.0)], dtype=np.float32)
            norm_delay = np.array([min(1.0, self.current_node_delay[i] / 300.0)], dtype=np.float32)
            obs.append(
                np.concatenate(
                    [
                        norm_pos,
                        norm_vel,
                        norm_trust,
                        norm_snr,
                        norm_hop,
                        norm_energy,
                        norm_alert,
                        norm_delay,
                    ]
                )
            )
        return np.array(obs, dtype=np.float32)

    def _get_global_state(self):
        return self._get_obs().flatten()
