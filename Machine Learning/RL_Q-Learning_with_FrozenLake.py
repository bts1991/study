import gymnasium as gym
import numpy as np
import random
import time

# 환경 생성
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

# Q-table 초기화
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# 하이퍼파라미터
alpha = 0.8 ## Q 업데이트 시, 미래 Q를 반영할 비율
gamma = 0.95 ## 할인율, 클수록 미래의 Reward에 충실
epsilon = 1.0 ## 클수록 Random 하게 움직임, 처음에는 무조건 Exploration 후 점점 Exploitation
epsilon_min = 0.01
epsilon_decay = 0.995 ## epsilon에 곱해서 epsilon을 점점 작아지게 만듦
# episodes = 2000
episodes = 100
## 에이전트가 환경과의 상호작용을 시작해서 종료(목표 도달 또는 실패)할 때까지의 한 사이클
## 에이전트가 시작 위치 S에서 시작해, 움직이며 목적지 G로 가거나 구멍 H에 빠지면 에피소드 종료.
## 총 2000번의 학습 사이클을 반복하겠다는 의미
total_reward = 0

# 학습 루프
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    print("======",f"[Episode {episode}]","=======================================================")
    while not done: ## 에피소드가 끝나지 않았다면
        if random.uniform(0, 1) < epsilon: ##  균등분포, 정규분포 아님
            action = env.action_space.sample() ## 왼쪽(0), 오른쪽(2), 아래(1), 위(3)
            print(f"[탐험] S:{state}, 무작위 행동 A:{action}")
        else:
            action = np.argmax(q_table[state]) ## 배열 내 가장 큰 인덱스, 즉 가장 높은 보상을 선택
            print(f"[이용] S:{state}, 최적 행동 A:{action}")

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        ## 특정 상태 state에서 특정 행동 action을 했을 때 앞으로 받을 것으로 예상되는 누적 보상의 기대값
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
 
        print(f"총 누적 보상: {total_reward}")
        # 업데이트 과정 출력
        print(f"S:{state}, A:{action} → S':{next_state}, R:{reward}")
        print(f"Q({state},{action}) 업데이트: {old_value:.4f} → {q_table[state, action]:.4f}")    

        state = next_state
        

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("✅ 학습 완료된 Q-table:")
print(q_table)

# # ------------------------------------------------------
# # 🧭 정책 시각화 함수
# # ------------------------------------------------------
# def render_policy_from_q_table(q_table, size=4):
#     arrow_dict = {
#         0: '←',  # Left
#         1: '↓',  # Down
#         2: '→',  # Right
#         3: '↑',  # Up
#     }

#     print("\n🧭 학습된 정책 (화살표로 시각화):\n")
#     for i in range(size):
#         row = ''
#         for j in range(size):
#             state = i * size + j
#             if np.sum(q_table[state]) == 0:
#                 row += '■ '  # 학습되지 않은 상태
#             else:
#                 action = np.argmax(q_table[state])
#                 row += arrow_dict[action] + ' '
#         print(row)

# render_policy_from_q_table(q_table)

# # ------------------------------------------------------
# # 🎬 최적 정책으로 이동 시뮬레이션
# # ------------------------------------------------------
# def play_best_policy(q_table, env):
#     state, _ = env.reset()
#     done = False
#     total_reward = 0

#     print("\n🎬 최적 정책으로 에이전트 이동:\n")
#     while not done:
#         env.render()
#         action = np.argmax(q_table[state])
#         time.sleep(0.5)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         total_reward += reward
#         state = next_state

#     print(env.render()) 
#     print(f"\n🏁 종료! 총 보상: {total_reward}\n")

# play_best_policy(q_table, env)
