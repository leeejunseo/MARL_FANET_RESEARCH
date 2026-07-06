import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_drones):
        super().__init__()
        input_dim = state_dim + (action_dim * num_drones)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)


class MATD3Agent:
    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        num_drones,
        agent_id,
        actor_lr=1e-3,
        critic_lr=1e-3,
        use_marl=True,
        actor_obs_dim=None,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        explore_noise_std=0.1,
    ):
        self.agent_id = agent_id
        self.use_marl = use_marl
        self.obs_dim = obs_dim
        self.actor_obs_dim = actor_obs_dim if actor_obs_dim is not None else obs_dim
        self.action_dim = action_dim

        critic_state_dim = state_dim if use_marl else obs_dim
        critic_num_drones = num_drones if use_marl else 1

        self.actor = Actor(self.actor_obs_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1 = Critic(critic_state_dim, action_dim, critic_num_drones)
        self.critic2 = Critic(critic_state_dim, action_dim, critic_num_drones)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.policy_delay = max(1, int(policy_delay))
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.explore_noise_std = float(explore_noise_std)
        self.update_step = 0

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
            action += np.random.normal(0.0, self.explore_noise_std, size=self.action_dim)
        return np.clip(action, -1.0, 1.0)

    def _target_action(self, agent, next_obs):
        action = agent.actor_target(agent._prepare_actor_obs(next_obs))
        noise = torch.randn_like(action) * self.target_policy_noise
        noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
        return torch.clamp(action + noise, -1.0, 1.0)

    def update(self, sample_data, all_agents, gamma=0.95, tau=0.01):
        self.update_step += 1

        obs, states, actions, rewards, next_obs, next_states, dones = sample_data
        batch_size = obs.shape[0]

        obs_t = torch.FloatTensor(obs)
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards)[:, self.agent_id].unsqueeze(1)
        next_obs_t = torch.FloatTensor(next_obs)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        if self.use_marl:
            with torch.no_grad():
                next_actions = []
                for i, agent in enumerate(all_agents):
                    next_actions.append(self._target_action(agent, next_obs_t[:, i]))
                next_actions_t = torch.cat(next_actions, dim=-1)
                target_q1 = self.critic1_target(next_states_t, next_actions_t)
                target_q2 = self.critic2_target(next_states_t, next_actions_t)
                target_q = torch.min(target_q1, target_q2)
                y = rewards_t + (1.0 - dones_t) * gamma * target_q

            current_actions_flat = actions_t.view(batch_size, -1)
            current_q1 = self.critic1(states_t, current_actions_flat)
            current_q2 = self.critic2(states_t, current_actions_flat)
        else:
            local_obs = obs_t[:, self.agent_id]
            local_next_obs = next_obs_t[:, self.agent_id]
            local_actions = actions_t[:, self.agent_id]

            with torch.no_grad():
                next_local_action = self._target_action(self, local_next_obs)
                target_q1 = self.critic1_target(local_next_obs, next_local_action)
                target_q2 = self.critic2_target(local_next_obs, next_local_action)
                target_q = torch.min(target_q1, target_q2)
                y = rewards_t + (1.0 - dones_t) * gamma * target_q

            current_q1 = self.critic1(local_obs, local_actions)
            current_q2 = self.critic2(local_obs, local_actions)

        critic_loss1 = nn.MSELoss()(current_q1, y)
        critic_loss2 = nn.MSELoss()(current_q2, y)

        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        if self.update_step % self.policy_delay != 0:
            return

        if self.use_marl:
            eval_actions = actions_t.clone()
            eval_actions[:, self.agent_id] = self.actor(self._prepare_actor_obs(obs_t[:, self.agent_id]))
            eval_actions_flat = eval_actions.view(batch_size, -1)
            actor_loss = -self.critic1(states_t, eval_actions_flat).mean()
        else:
            local_obs = self._prepare_actor_obs(obs_t[:, self.agent_id])
            actor_loss = -self.critic1(local_obs, self.actor(local_obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self, path, episode):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor_agent_{self.agent_id}_ep_{episode}.pth")
        torch.save(self.critic1.state_dict(), f"{path}/critic1_agent_{self.agent_id}_ep_{episode}.pth")
        torch.save(self.critic2.state_dict(), f"{path}/critic2_agent_{self.agent_id}_ep_{episode}.pth")
