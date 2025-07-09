import gym
import numpy as np
import matplotlib.pyplot as plt

# 하이퍼파라미터
EPISODES = 1000           # 학습할 에피소드 수
ALPHA = 0.1               # 학습률 (learning rate)
GAMMA = 0.99              # 할인율 (future reward 반영 정도)
EPSILON_START = 1.0       # ε-greedy 시작값
EPSILON_MIN = 0.01        # ε 최소값
EPSILON_DECAY = 0.995     # ε 매 에피소드 감소 비율

# ε-greedy 정책 함수
def select_action(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon: # np.random.rand()는 NumPy 라이브러리에서 0 이상 1 미만의 난수를 생성하는 함수, 균등분포(Uniform Distribution) 를 따름
        return np.random.randint(n_actions) # 탐험 (exploration), random.randint()는 정수 난수를 생성하는 함수
    return np.argmax(Q[state]) # 이용 (exploitation)

# Q-learning 알고리즘
def train_q_learning(env):
    Q = np.zeros((env.observation_space.n, env.action_space.n)) # Q-table 초기화, state * action (500 * 6) 크기의 배열
    
    rewards = []
    epsilon = EPSILON_START

    for ep in range(EPISODES):
        state, _ = env.reset() # 환경 초기화
        # reset()은 두 개의 값을 반환하는데 뒤의 값은 사용하지 않겠다는 의미
        print('================================')
        print("ep: ",ep)
        print("state: ",state)
        total_reward = 0
        done = False

        step = 0
        
        while not done : # 한 에피소드 동안
            print('***********************************************')  
            print('step: ', step) 
            print('state: ', state)
            action = select_action(Q, state, epsilon, env.action_space.n) # action 선택
            print("action: ", action)
            next_state, reward, terminated, truncated, _ = env.step(action) # 환경에 action을 전달하고, 다음 상태/보상/종료 여부를 받음
            # 최신 Gym은 step()에서 5개의 값을 반환
            print("env.step(action)ction: ", next_state, reward, terminated, truncated)
            done = terminated or truncated # 둘 중 하나라도 True면 에피소드 종료로 판단
            print("done: ", done)
            print("Q_next: ", Q[next_state],next_state )
            Q[state][action] += ALPHA * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
            )
            print("Q: ", Q)

            state = next_state
            print("next state: ", next_state)
            total_reward += reward
            print("Total Reward: ", total_reward)
            step +=1

        rewards.append(total_reward) # 총 보상 저장
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return rewards # 각 에피소드의 누적 보상 리스트

# SARSA 알고리즘
def train_sarsa(env):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    epsilon = EPSILON_START

    for ep in range(EPISODES):
        state, _ = env.reset()
        action = select_action(Q, state, epsilon, env.action_space.n)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = select_action(Q, next_state, epsilon, env.action_space.n)

            Q[state][action] += ALPHA * (
                reward + GAMMA * Q[next_state][next_action] - Q[state][action]
            )

            state, action = next_state, next_action
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return rewards

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# 실행 및 시각화
def run_experiment():
    env = gym.make("Taxi-v3")
    q_rewards = train_q_learning(env)
    sarsa_rewards = train_sarsa(env)

    # # 시각화
    # plt.figure(figsize=(10, 6))
    # plt.plot(q_rewards, label="Q-Learning")
    # plt.plot(sarsa_rewards, label="SARSA")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("SARSA vs Q-Learning on Taxi-v3")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    
    # 이동평균 계산
    q_ma = moving_average(q_rewards, window_size=50)
    sarsa_ma = moving_average(sarsa_rewards, window_size=50)

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(q_ma, label="Q-Learning (Moving Avg)")
    plt.plot(sarsa_ma, label="SARSA (Moving Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Moving Average)")
    plt.title("SARSA vs Q-Learning on Taxi-v3 (Smoothed)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
