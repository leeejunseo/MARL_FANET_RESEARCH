import numpy as np
import gymnasium as gym

class AdvancedFANETEnv(gym.Env):
    def __init__(self, num_drones=3):
        super(AdvancedFANETEnv, self).__init__()
        self.num_drones = num_drones
        
        # 전술 시뮬레이션 물리 파라미터
        self.max_pos = 1000.0
        self.max_vel = 20.0
        
        # 임무 설계 기준 스펙
        self.R_c = 300.0     # 통신 제한 반경 (FANET 한계선)
        self.d_safe = 30.0   # 드론 간 최소 안전 거리 (충돌 임계선)
        
        # 차원 정의
        self.obs_dim = 6
        self.state_dim = self.obs_dim * self.num_drones
        self.action_dim = 3

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # 초기 위치를 센터 부근에 클러스터 형태로 배치하여 전술 대형 유도
        self.positions = np.random.uniform(450.0, 550.0, (self.num_drones, 3))
        self.velocities = np.zeros((self.num_drones, 3))
        return self._get_obs(), self._get_global_state()

    def step(self, actions):
        # 1. 물리 가속도 제어 및 경계면 제한 (Bounding Box)
        actions = np.clip(actions, -1.0, 1.0)
        self.velocities += actions * 2.0  # 가속도 스케일링
        self.velocities = np.clip(self.velocities, -self.max_vel, self.max_vel)
        self.positions += self.velocities
        self.positions = np.clip(self.positions, 0.0, self.max_pos)
        
        # 2. 대학원 수준의 다중 목적 보상 계산
        rewards = np.zeros(self.num_drones)
        
        for i in range(self.num_drones):
            r_cov = 0.0
            r_conn = 0.0
            r_col = 0.0
            
            # 다른 드론들과의 상대적 거리 행렬 계산
            for j in range(self.num_drones):
                if i == j: continue
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                
                # 가목표 1: 탐지 영역 확장 (스웜 확산 유도)
                r_cov += dist / self.max_pos
                
                # 가목표 2: 충돌 회피 페널티 (안전거리 미만 진입 시)
                if dist < self.d_safe:
                    r_col -= 5.0 * (1.0 - (dist / self.d_safe))
                    
                # 가목표 3: FANET 통신망 유지 페널티 (통신 반경 이탈 시)
                if dist > self.R_c:
                    r_conn -= 2.0 * ((dist - self.R_c) / self.max_pos) ** 2
            
            # 가중치 결합 (전술적 밸런싱)
            rewards[i] = (1.0 * r_cov) + (2.5 * r_col) + (3.0 * r_conn)
            
        terminated = False
        return self._get_obs(), self._get_global_state(), rewards, terminated, {}

    def _get_obs(self):
        obs = []
        for i in range(self.num_drones):
            norm_pos = self.positions[i] / self.max_pos
            norm_vel = self.velocities[i] / self.max_vel
            obs.append(np.concatenate([norm_pos, norm_vel]))
        return np.array(obs, dtype=np.float32)

    def _get_global_state(self):
        return self._get_obs().flatten()
