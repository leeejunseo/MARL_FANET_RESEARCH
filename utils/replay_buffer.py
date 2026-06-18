import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, action_dim, num_drones):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # 다중 에이전트의 데이터를 저장하기 위한 다차원 배열 초기화
        self.obs = np.zeros((capacity, num_drones, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_drones, obs_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_drones, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_drones), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, state, action, reward, next_obs, next_state, done):
        # 현재 포인터 위치에 경험 저장
        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # 링 버퍼(Ring Buffer) 구조로 포인터 순환
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # 무작위로 배치 크기만큼의 인덱스 추출
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[ind],
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_obs[ind],
            self.next_states[ind],
            self.dones[ind]
        )
