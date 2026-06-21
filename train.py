import os
import numpy as np
import torch
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.tacview_logger import TacviewLogger
from utils.metrics_logger import MetricsLogger

def main():
    print("========================================================")
    print("  MARL-FANET 심층 학습 및 Tacview 시각화 연동 시스템")
    print("========================================================")
    
    num_drones = 3
    env = AdvancedFANETEnv(num_drones=num_drones)
    agents = [MADDPGAgent(env.obs_dim, env.state_dim, env.action_dim, num_drones, agent_id=i) for i in range(num_drones)]
    replay_buffer = ReplayBuffer(100000, env.obs_dim, env.state_dim, env.action_dim, num_drones)
    
    # 학습 하이퍼파라미터
    batch_size = 128
    max_episodes = 1000    # 본격적인 심층 학습을 위해 에피소드 대폭 증가
    max_steps = 60
    warmup_steps = 500     # 버퍼를 충분히 채우기 위한 초기 탐색
    save_interval = 200    # 200 에피소드마다 모델 저장
    
    global_step = 0
    model_dir = "models/weights"
    metrics_logger = MetricsLogger("logs/training_rewards.csv")
    
    print("\n[알림] 대규모 데이터 수집 및 인공지능 최적화 시작...")
    
    for episode in range(1, max_episodes + 1):
        obs, state = env.reset()
        episode_reward = 0
        
        # 마지막 에피소드에서만 Tacview 로그를 기록하여 최종 전술 확인
        record_tacview = (episode == max_episodes)
        if record_tacview:
            tacview = TacviewLogger(f"logs/swarm_final_ep{episode}.acmi")
            print("\n[알림] 최종 에피소드 전술 기동 Tacview 로그 기록 중...")
            
        for step in range(max_steps):
            global_step += 1
            
            # 탐색 및 행동 결정
            actions = []
            for i in range(num_drones):
                if global_step < warmup_steps:
                    action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
                else:
                    # 후반부로 갈수록 노이즈(탐색)를 줄여 안정적인 전술 확인
                    explore = not record_tacview 
                    action = agents[i].act(obs[i], explore=explore)
                actions.append(action)
            actions = np.array(actions)
            
            next_obs, next_state, rewards, terminated, _ = env.step(actions)
            replay_buffer.add(obs, state, actions, rewards, next_obs, next_state, terminated)
            
            # Tacview 위치 로깅 (dt=1.0초 가정)
            if record_tacview:
                tacview.log_step(time_step=float(step), positions=env.positions)
            
            obs = next_obs
            state = next_state
            episode_reward += np.sum(rewards)
            
            # 실시간 역전파 가중치 업데이트
            if global_step >= warmup_steps and replay_buffer.size >= batch_size:
                sample_data = replay_buffer.sample(batch_size)
                for agent in agents:
                    agent.update(sample_data, agents)
                    
            if terminated:
                break
                
        metrics_logger.log_episode(episode, episode_reward, num_drones)

        # 로그 출력 (10 에피소드마다)
        if episode % 10 == 0 or episode == 1:
            print(f"에피소드 {episode:04d} | 총 전술 보상 합계: {episode_reward:7.2f} | 버퍼 크기: {replay_buffer.size:6d}")
            
        # 지정된 주기마다 모델 저장
        if episode % save_interval == 0:
            for agent in agents:
                agent.save_models(model_dir, episode)
            print(f"  -> [저장 완료] 에피소드 {episode} 신경망 가중치 저장됨 ({model_dir}/)")

    print("\n========================================================")
    print("  [연구 완료] 모델 저장 및 Tacview ACMI 로그 추출 완료")
    print("========================================================")

if __name__ == "__main__":
    main()
