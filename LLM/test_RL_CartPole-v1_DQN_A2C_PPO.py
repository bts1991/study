import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터
EPISODES = 100 # 하나의 에피소드는 막대기를 넘어뜨리거나 최대 시간(500 step)에 도달할 때까지의 시퀀스
# 에피소드 마다 스텝의 개수가 달라질 수 있음, 메모리에 저장되는 개수가 달라짐
GAMMA = 0.99
LR = 1e-3 # 학습률(Learning Rate), 1e-3은 0.001

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0] # 관측 공간(state space)의 차원 수, 상태 벡터가 4차원[카트 위치, 카트 속도, 막대 각도, 막대 각속도] → state_dim = 4, 
# print(env.observation_space)
print("state_dim: ", state_dim)
action_dim = env.action_space.n # 행동 공간(action space)의 개수, action_dim = 2
print("action_dim: ",action_dim)

# 공통 신경망
class MLP(nn.Module): # PyTorch의 신경망 모듈 nn.Module을 상속한 클래스
    def __init__(self, input_dim, output_dim): # input_dim: 입력 벡터의 차원, output_dim: 출력 벡터의 차원 (예: 행동의 개수 등)
        super().__init__() # nn.Module의 초기화를 호출
        self.model = nn.Sequential( # 여러 층을 순차적으로 쌓을 때 편리
            nn.Linear(input_dim, 128), nn.ReLU(), # 각 nn.Linear는 완전 연결층 (fully connected layer)
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x): # 입력 x는 순차적으로 정의된 self.model을 거쳐감
        return self.model(x)

# KK = MLP(20,10)
# print(KK)
# KK2 = MLP(20,10).to(device)
# print(KK2)

# DQN 알고리즘
class DQNAgent:
    def __init__(self):
        self.q_net = MLP(state_dim, action_dim).to(device) # 입력 데이터를 받아 비선형 변환을 여러 번 거쳐, 원하는 출력 벡터로 매핑하는 신경망 모델을 만드는 역할
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = deque(maxlen=10000) # 경험을 저장할 경험 리플레이 버퍼. (s, a, r, s', done) 형태로 저장. 최대 1만개 저장, maxlen 를 넘어가면 오래된 데이터를 자동으로 제거
        # deque: double-ended queue(양방향 큐) 자료구조, 양쪽 끝에서 데이터를 추가하거나 제거할 수 있는 큐(queue)**입니다.(일반적인 리스트보다 삽입/삭제 연산이 훨씬 빠름)
        self.batch_size = 64 # 학습 시 한 번에 샘플링할 미니배치 크기
        self.epsilon = 0.8 #1.0 # 행동 선택 시 무작위 선택할 확률. 처음엔 100% 탐색부터 시작
        self.eps_min = 0.01 # epsilon은 학습이 진행되며 점점 감소하지만 최소 이 값까지 유지
        self.eps_decay = 0.995 # 학습할 때마다 epsilon *= 0.995로 점점 줄여감 (탐색 → 활용)

    def act(self, state):
        print('==================== action 시작=====================')
        k = np.random.rand()
        print(k, ' < ', self.epsilon, 'then exploration, else exploitation')
        if k < self.epsilon: # epsilon의 확률로
            print('action1: ', env.action_space.sample())
            return env.action_space.sample() # 가능한 행동들 중 무작위로 선택
        print('torch.state(카트위치, 카트속도, 막대각도, 막대각속도): ', torch.FloatTensor(state))
        print('torch.state.shape: ', torch.FloatTensor(state).shape)
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # .unsqueeze(0): 차원 추가 → shape: [1, state_dim] (배치 크기 1)
        print('torch.state+unsqueeze: ', state)
        print('torch.state+unsqueeze.shape(0): ', state.shape)
        print('Q_value 들: ', self.q_net(state))
        print('Q_value 의 Max: ', self.q_net(state).argmax())
        print('숫자형 Q_Max: ', self.q_net(state).argmax().item())
        return self.q_net(state).argmax().item()
        # 신경망을 통해 각 행동에 대한 Q값을 예측 -> tensor([[Q_0, Q_1, ..., Q_n]])
        # .argmax()sms Q값이 가장 큰 인덱스 (즉, 행동)를 선택
        # .item()은 파이토치 텐서를 파이썬 숫자형으로 바꿔 반환

    def memorize(self, s, a, r, s_, d): # 경험 리플레이 메모리에 transition 데이터를 저장
        print('==================== memorize 시작=====================')
        print('memorize(s, a, r, s_, d): ', (s, a, r, s_, d))
        self.memory.append((s, a, r, s_, d)) #s_: 다음 상태, d: 종료 여부(done; true or false)

    def train(self):
        print('==================== train 시작=====================')
        print('length of memory: ', len(self.memory), ' VS ', self.batch_size)
        if len(self.memory) < self.batch_size: # 저장된 경험이 batch_size보다 적으면 → 학습 X (학습 불가능)
            return
        minibatch = random.sample(self.memory, self.batch_size) # 경험 리플레이 메모리에서 무작위로 batch_size 개만큼 샘플링
        # batch_size 개수만큼 한 번에 학습
        print('minibatch.lenth: ', len(minibatch))
        s, a, r, s_, d = zip(*minibatch) # 샘플링한 데이터를 각각 상태, 행동, 보상, 다음 상태, 종료 여부로 분리
        s = torch.FloatTensor(s).to(device) # [batch_size, state_dim]
        # print('s: ', s)
        print('s_shape: ', s.shape)
        a = torch.LongTensor(a).unsqueeze(1).to(device) # [batch_size, 1]
        # print('a: ', a)
        print('a_shape: ', a.shape)
        r = torch.FloatTensor(r).unsqueeze(1).to(device) # 	[batch_size, 1]
        # print('r: ', r)
        print('r_shape: ', r.shape)
        s_ = torch.FloatTensor(s_).to(device) # [batch_size, state_dim]
        # print('s_: ', s_)
        print('s_ _shape: ', s_.shape)
        d = torch.FloatTensor(d).unsqueeze(1).to(device) # [batch_size, 1] (0: 진행중, 1: 종료)
        # print('d: ', d)
        print('d_shape: ', d.shape)
        
        print('q_net: ', self.q_net)
        print('q_val_s: ', self.q_net(s))
        print('q_val_s.shape: ', self.q_net(s).shape)
        q_vals = self.q_net(s).gather(1, a)
        # self.q_net(s): 모든 행동에 대한 Q값 [batch_size, action_dim]
        # gather(1, a): 우리가 실제로 실행한 행동 a에 대한 Q값만 선택
        print('q_val_s_a: ', q_vals)
        print('q_val_s_a.shape: ', q_vals.shape)

        q_next = self.q_net(s_).max(1)[0].unsqueeze(1)
        
        print('q_val_nxs: ', self.q_net(s_))
        print('q_val_nxs.shape: ', self.q_net(s_).shape)
        print('q_val_nxs.max1: ', self.q_net(s_).max(1))
        print('q_val_nxs.max1[0]: ', self.q_net(s_).max(1)[0])
        print('q_val_nxs_max: ', q_next)
        print('q_val_nxs_max.shape: ', q_next.shape)
        # 다음 상태 s_에 대해 모든 Q값 계산 후, 최대값 선택
        # max(1)[0]: dim=1 (행 기준), [0]은 Q값만 (index는 [1])
        target = r + GAMMA * q_next * (1 - d) # done == 1인 경우 → 종료 상태 → 다음 Q값 무시
        
        print('target: ', target)

        loss = nn.MSELoss()(q_vals, target.detach()) # detach(): target은 gradient 계산에서 제외 (고정값으로 사용)
        print('loss: ', loss)
        self.optimizer.zero_grad() # zero_grad(): 이전 gradient 초기화
        loss.backward() # backward(): 손실 함수로부터 gradient 계산
        self.optimizer.step() # step(): optimizer가 파라미터 업데이트

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        # epsilon을 점점 줄여감 (예: 1.0 → 0.995 → 0.99 → ...)
        # eps_min보다 작아지지 않도록 제한


# A2C 에이전트
class A2CAgent:
    def __init__(self):
        self.actor = MLP(state_dim, action_dim).to(device) # 각 행동에 대한 확률 분포
        self.critic = MLP(state_dim, 1).to(device) # Action value fuction approximation
        # 주어진 상태 s에 대해 미래의 누적 보상(return)을 예측하는 가치 함수
        # 출력: V(s) 또는 Q(s, a) 형태의 scalar 값
        # 학습 목표: actor가 얼마나 "잘" 행동했는지 판단 기준 제공
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR
        ) # actor와 critic의 모든 파라미터를 하나의 옵티마이저로 함께 업데이트

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # 상태를 float형 텐서로 변환
        print("state.tensor: ", state)
        logits = self.actor(state) # 액션에 대한 점수 (정책 확률 분포의 logit)
        print("logits: ", logits)
        print("probs:", F.softmax(logits, dim=-1))
        dist = torch.distributions.Categorical(logits=logits)
        print("dist: ", dist)
        # Categorical: 다항 분포로부터 행동 샘플링
        action = dist.sample() # 위 확률 분포에 따라 행동 하나를 무작위로 선택
        print("action: ", action)
        return action.item(), dist.log_prob(action)
        # log_prob(action): 샘플링된 행동의 로그확률 (나중에 정책 경사 계산에 사용됨)
        # ln(π(a|s)), 즉 action이 선택될 확률의 로그 값을 계산
        
    def train(self, trajectory): # 학습 목표: 좋은 행동의 확률을 높이고, 나쁜 행동의 확률을 낮추는 것
        states, actions, rewards, log_probs = zip(*trajectory) # 에피소드 전체 데이터를 분해해서 각 항목으로 리스트화
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        # 누적 보상, 뒤에서부터 앞으로 계산하여 리스트 형태로 저장
        # print('returns: ', returns)

        # states = torch.FloatTensor(states).to(device)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        log_probs = torch.stack(log_probs).to(device)
        
        print('states.shape: ', states.shape)
        print('actions.shape: ', actions.shape)
        print('returns.shape: ', returns.shape)
        print('log_probs.shape: ', log_probs.shape)

        values = self.critic(states) # V(s)
        print('values.shape: ', values.shape)
        advantage = returns - values # A(s, a), Advantage = 실제 return - 예측된 value, 이 값을 통해 정책 업데이트 시 방향성을 반영

        actor_loss = -(log_probs * advantage.detach()).mean()
        # Advantage가 클수록 log_prob를 크게 하도록(즉, 해당 행동의 확률을 높이도록) 학습 => Loss 함수를 최소화 하도록 유도
        # detach()는 Critic 네트워크에 그래디언트가 흘러가지 않게 막음
        critic_loss = nn.MSELoss()(values, returns)
        # 예측한 V(s)와 실제 Gₜ 사이의 MSE 손실
        loss = actor_loss + critic_loss
        # print('loss: ', loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print('<<<< train_end >>>>')


# PPO 에이전트
class PPOAgent:
    def __init__(self, clip_eps=0.2):
        self.actor = MLP(state_dim, action_dim).to(device)
        self.critic = MLP(state_dim, 1).to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR
        )
        self.clip_eps = clip_eps

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        print("state.tensor: ", state)
        logits = self.actor(state)
        print("logits: ", logits)
        print("probs:", F.softmax(logits, dim=-1))
        dist = torch.distributions.Categorical(logits=logits)
        print("dist: ", dist)
        action = dist.sample()
        print("action: ", action)
        return action.item(), dist.log_prob(action), dist

    def train(self, trajectory):
        states, actions, rewards, old_log_probs = zip(*trajectory)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G # G: 누적 할인 보상 (return)
            returns.insert(0, G)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        old_log_probs = torch.stack(old_log_probs).detach().to(device)

        print('states.shape: ', states.shape)
        print('actions.shape: ', actions.shape)
        print('returns.shape: ', returns.shape)
        print('old_log_probs.shape: ', old_log_probs.shape)

        for _ in range(4):  # K epochs
            logits = self.actor(states)
            # print('logits: ', logits)
            print('logits.shape: ', logits.shape)
            print('logits[0]: ', logits[0])
            dist = torch.distributions.Categorical(logits=logits)
            print('dist: ', dist)
            new_log_probs = dist.log_prob(actions)
            print('new_log_probs.shape: ', new_log_probs.shape)
            ratio = torch.exp(
                new_log_probs - old_log_probs)
            values = self.critic(states)
            advantage = returns - values.detach()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        print('<<<< train_end >>>>')


# 실험 실행 함수
def run(agent_class, label):
    scores = []
    agent = agent_class()
    for ep in range(EPISODES):
        print('==================== [',label,'] episode: ', ep,' =====================')
        state, _ = env.reset()
        print('init.state: ', state)
        done = False
        score = 0
        trajectory = [] # A2C, PPO에서 학습에 사용하는 (상태, 행동, 보상, 로그확률) 저장용
        count = 0 # 디버깅용 루프 카운터 (DQN에서만 사용)
        while not done: # 한 스텝, 한 스텝 진행하다가 특정 state에서 더 이상 진행할 수 없으면 done = True를 반환
            if label == "DQN":
                count = count + 1
                print('==================== [',label,'] episode: ', ep,' count: ', count,' =====================')
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                print('End? ', done)
                agent.memorize(state, action, reward, next_state, done)
                agent.train()
            else:
                count = count + 1
                print('==================== [',label,'] episode: ', ep,' count: ', count,' =====================')
                action, *log = agent.act(state)
                print('action: ', action)
                print('log_prob: ', *log)
                next_state, reward, done, _, _ = env.step(action)
                print("next_state, reward: ", (next_state, reward))
                print('End? ', done)
                if label == "A2C":
                    trajectory.append((state, action, reward, log[0]))
                    # 학습에 사용할 데이터를 수집
                    # trajectory는 한 에피소드 동안의 모든 transition을 저장한 리스트
                else:
                    trajectory.append((state, action, reward, log[0]))  # PPO
            score += reward
            state = next_state
        scores.append(score)
        if label != "DQN":
            print('<<<< train_start >>>>')
            agent.train(trajectory) # 한 에피소드 끝나면 한 번에 학습
        if (ep + 1) % 10 == 0:
            print(f"[{label}] Episode {ep+1}: Avg Score = {np.mean(scores[-10:]):.2f}")
    return scores


dqn_scores = run(DQNAgent, "DQN")
a2c_scores = run(A2CAgent, "A2C")
ppo_scores = run(PPOAgent, "PPO")

# # 시각화
# plt.plot(dqn_scores, label="DQN")
# plt.plot(a2c_scores, label="A2C")
# plt.plot(ppo_scores, label="PPO")
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.legend()
# plt.title("Performance Comparison on CartPole")
# plt.grid()
# plt.show()

# 이동 평균 함수 정의
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# 이동 평균 계산
dqn_ma = moving_average(dqn_scores)
a2c_ma = moving_average(a2c_scores)
ppo_ma = moving_average(ppo_scores)

# 이동 평균 적용된 에피소드 인덱스
episodes_ma = np.arange(len(dqn_ma))

# 시각화: 원본 + 이동 평균 비교
plt.figure(figsize=(10, 6))

# 원본
plt.plot(dqn_scores, label="DQN (raw)", alpha=0.3, color='blue')
plt.plot(a2c_scores, label="A2C (raw)", alpha=0.3, color='orange')
plt.plot(ppo_scores, label="PPO (raw)", alpha=0.3, color='green')

# 이동 평균
plt.plot(episodes_ma, dqn_ma, label="DQN (MA)", color='blue')
plt.plot(episodes_ma, a2c_ma, label="A2C (MA)", color='orange')
plt.plot(episodes_ma, ppo_ma, label="PPO (MA)", color='green')

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Performance Comparison with Moving Average (CartPole)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()