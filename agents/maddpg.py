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
    def __init__(self, obs_dim, state_dim, action_dim, num_drones, agent_id, lr=1e-3, use_marl=True, actor_obs_dim=None):
        self.agent_id = agent_id
        self.use_marl = use_marl
        self.obs_dim = obs_dim
        self.actor_obs_dim = actor_obs_dim if actor_obs_dim is not None else obs_dim
        
        # 메인 네트워크 생성
        self.actor = Actor(self.actor_obs_dim, action_dim)
        critic_state_dim = state_dim if use_marl else obs_dim
        critic_num_drones = num_drones if use_marl else 1
        self.critic = Critic(critic_state_dim, action_dim, critic_num_drones)
        
        # 타겟 네트워크 생성 (학습 안정화용 심층 버퍼)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 옵티마이저 정의
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.action_dim = action_dim

    def _prepare_actor_obs(self, obs):
        if obs.shape[-1] > self.actor_obs_dim:
            return obs[..., : self.actor_obs_dim]
        return obs

    def act(self, obs, explore=True):
        obs = self._prepare_actor_obs(np.asarray(obs, dtype=np.float32))
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
        # 1. Critic 업데이트 (MSBE 손실함수 계산)
        # -------------------------------------------------------------
        if self.use_marl:
            with torch.no_grad():
                next_actions = []
                for i, agent in enumerate(all_agents):
                    next_act = agent.actor_target(next_obs_t[:, i])
                    next_actions.append(next_act)
                next_actions_t = torch.cat(next_actions, dim=-1)
                target_Q = self.critic_target(next_states_t, next_actions_t)
                y = rewards_t + (1 - dones_t) * gamma * target_Q

            current_actions_flat = actions_t.view(batch_size, -1)
            current_Q = self.critic(states_t, current_actions_flat)
        else:
            local_obs = obs_t[:, self.agent_id]
            local_next_obs = next_obs_t[:, self.agent_id]
            local_actions = actions_t[:, self.agent_id]
            with torch.no_grad():
                next_local_action = self.actor_target(local_next_obs)
                target_Q = self.critic_target(local_next_obs, next_local_action)
                y = rewards_t + (1 - dones_t) * gamma * target_Q

            current_Q = self.critic(local_obs, local_actions)

        critic_loss = nn.MSELoss()(current_Q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------------------------------------------------
        # 2. Actor 업데이트 (Policy Gradient 기법)
        # -------------------------------------------------------------
        if self.use_marl:
            eval_actions = actions_t.clone()
            eval_actions[:, self.agent_id] = self.actor(self._prepare_actor_obs(obs_t[:, self.agent_id]))
            eval_actions_flat = eval_actions.view(batch_size, -1)
            actor_loss = -self.critic(states_t, eval_actions_flat).mean()
        else:
            local_obs = self._prepare_actor_obs(obs_t[:, self.agent_id])
            actor_loss = -self.critic(local_obs, self.actor(local_obs)).mean()
        
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

        with torch.no_grad():
            q_current_mean = float(current_Q.mean().item())
            q_target_mean = float(y.mean().item())
            q_overestimation_gap = q_current_mean - q_target_mean
            q_abs_td_error = float(torch.abs(current_Q - y).mean().item())

        return {
            "q_current_mean": q_current_mean,
            "q_target_mean": q_target_mean,
            "q_overestimation_gap": q_overestimation_gap,
            "q_abs_td_error": q_abs_td_error,
            "q1_current_mean": q_current_mean,
            "q2_current_mean": None,
            "q_disagreement_mean": None,
            "critic_loss1": float(critic_loss.item()),
            "critic_loss2": None,
            "actor_loss": float(actor_loss.item()),
            "actor_updated": 1.0,
        }

    def save_models(self, path, episode):
        """학습된 액터와 크리틱 네트워크의 가중치를 .pth 파일로 저장합니다."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor_agent_{self.agent_id}_ep_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_agent_{self.agent_id}_ep_{episode}.pth")
