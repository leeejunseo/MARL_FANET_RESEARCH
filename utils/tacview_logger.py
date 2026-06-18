import os
import numpy as np
from datetime import datetime

class TacviewLogger:
    def __init__(self, filepath="logs/swarm_tactics_advanced.acmi"):
        self.filepath = filepath
        self.Rc = 300.0  # FANET 통신 한계 반경 설정
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write(f"0,ReferenceTime={datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            f.write("0,Title=MARL UAV Swarm Tactical Simulation (Advanced)\n")

    def log_step(self, time_step, positions):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"#{time_step:.2f}\n")
            
            # 실시간 통신 상태 계산 로직
            connected = [True] * len(positions)
            for i in range(len(positions)):
                is_isolated = True
                for j in range(len(positions)):
                    if i == j: continue
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= self.Rc:
                        is_isolated = False
                        break
                connected[i] = not is_isolated

            # 3D 렌더링 속성 적용
            for i, pos in enumerate(positions):
                drone_id = f"{100+i:X}"
                
                # 통신이 끊어지면 색상을 Yellow로 변경하여 시각적 경고
                color = "Red" if connected[i] else "Yellow"
                
                if time_step == 0.0:
                    # 첫 스텝: 기체 타입(UAV) 및 반투명 정찰 반경(Radius) 초기 설정
                    f.write(f"{drone_id},T={pos[0]:.2f}|{pos[1]:.2f}|{pos[2]:.2f},Name=UAV_{i},Type=Air+FixedWing+UAV,Color={color},Radius=150\n")
                else:
                    # 이후 스텝: 위치 및 동적 상태(색상) 지속 업데이트
                    f.write(f"{drone_id},T={pos[0]:.2f}|{pos[1]:.2f}|{pos[2]:.2f},Color={color}\n")