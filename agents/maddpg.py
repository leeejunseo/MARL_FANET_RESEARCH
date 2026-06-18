import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_drones):
        super(Critic, self).__init__()
        input_dim = state_dim + (action_dim * num_drones)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)

class MADDPGAgent:
    def __init__(self, obs_dim, state_dim, action_dim, num_drones, agent_id, lr=1e-3):
        self.agent_id = agent_id
        
        # 메인 네트워크 생성
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(state_dim, action_dim, num_drones)
        
        # 타겟 네트워크 생성 (학습 안정화용 심층 버퍼)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 옵티마이저 정의
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.action_dim = action_dim

    def act(self, obs, explore=True):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0).numpy()
        if explore:
            # 탐색성 향상을 위한 가우시안 전술 노이즈 추가
            action += np.random.normal(0, 0.1, size=self.action_dim)
        return np.clip(action, -1.0, 1.0)

    def update(self, sample_data, all_agents, gamma=0.95, tau=0.01):

    
        obs, states, actions, rewards, next_obs, next_states, dones = sample_data
        batch_size = obs.shape[0]
        num_drones = len(all_agents)
        
        # 장치 텐서 변환
        obs_t = torch.FloatTensor(obs)
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards)[:, self.agent_id].unsqueeze(1)
        next_obs_t = torch.FloatTensor(next_obs)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # -------------------------------------------------------------
        # 1. Centralized Critic 업데이트 (MSBE 손실함수 계산)
        # -------------------------------------------------------------
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(all_agents):
                next_act = agent.actor_target(next_obs_t[:, i])
                next_actions.append(next_act)
            next_actions_t = torch.cat(next_actions, dim=-1)
            
            target_Q = self.critic_target(next_states_t, next_actions_t)
            y = rewards_t + (1 - dones_t) * gamma * target_Q

        # 현재 크리틱 평가값 계산
        current_actions_flat = actions_t.view(batch_size, -1)
        current_Q = self.critic(states_t, current_actions_flat)
        
        critic_loss = nn.MSELoss()(current_Q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------------------------------------------------
        # 2. Decentralized Actor 업데이트 (Policy Gradient 기법)
        # -------------------------------------------------------------
        # 다른 에이전트의 행동은 고정하고, 자신(i번째)의 행동만 연산 그래프에 포함하여 트래킹
        eval_actions = actions_t.clone()
        eval_actions[:, self.agent_id] = self.actor(obs_t[:, self.agent_id])
        eval_actions_flat = eval_actions.view(batch_size, -1)
        
        actor_loss = -self.critic(states_t, eval_actions_flat).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------------------------------------------------------------
        # 3. Target Networks Soft Update
        # -------------------------------------------------------------
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            # 기존 update 함수 아래에 추가합니다.
    def save_models(self, path, episode):
        """학습된 액터와 크리틱 네트워크의 가중치를 .pth 파일로 저장합니다."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor_agent_{self.agent_id}_ep_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_agent_{self.agent_id}_ep_{episode}.pth")
