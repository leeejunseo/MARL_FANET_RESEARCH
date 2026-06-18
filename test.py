import numpy as np
import torch
from ns3_wrapper.fanet_env import AdvancedFANETEnv
from agents.maddpg import MADDPGAgent
from utils.tacview_logger import TacviewLogger

def test_inference():
    print("=== 완성된 신경망 기반 전술 기동 평가 시작 ===")
    num_drones = 3
    env = AdvancedFANETEnv(num_drones=num_drones)
    agents = [MADDPGAgent(env.obs_dim, env.state_dim, env.action_dim, num_drones, agent_id=i) for i in range(num_drones)]
    
    # 1000번째 에피소드에서 저장된 최적 전술 가중치 로드
    model_dir = "models/weights"
    episode_to_load = 1000
    try:
        for i, agent in enumerate(agents):
            agent.actor.load_state_dict(torch.load(f"{model_dir}/actor_agent_{i}_ep_{episode_to_load}.pth", weights_only=True))
        print("[성공] 1000 에피소드 학습 모델 로드 완료")
    except Exception as e:
        print(f"[오류] 모델을 찾을 수 없습니다: {e}")
        return

    # 동일한 전술 비교를 위해 시드 고정
    obs, state = env.reset(seed=42)
    tacview = TacviewLogger("logs/swarm_test_eval.acmi")
    
    print("순수 전술 기동 평가 중...")
    for step in range(60):
        actions = []
        for i in range(num_drones):
            # explore=False 로 설정하여 노이즈 없는 100% 최적화된 행동만 도출
            action = agents[i].act(obs[i], explore=False)
            actions.append(action)
        
        next_obs, next_state, rewards, terminated, _ = env.step(np.array(actions))
        tacview.log_step(float(step), env.positions)
        
        obs = next_obs
        state = next_state
        if terminated:
            break
            
    print("=== 평가 종료: logs/swarm_test_eval.acmi 파일 추출 완료 ===")

if __name__ == "__main__":
    test_inference()
