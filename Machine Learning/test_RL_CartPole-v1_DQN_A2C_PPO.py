import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터
EPISODES = 300 # 하나의 에피소드는 막대기를 넘어뜨리거나 최대 시간(500 step)에 도달할 때까지의 시퀀스
GAMMA = 0.99
LR = 1e-3 # 학습률(Learning Rate), 1e-3은 0.001

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0] # 관측 공간(state space)의 차원 수, 상태 벡터가 4차원[카트 위치, 카트 속도, 막대 각도, 막대 각속도] → state_dim = 4, 
print(env.observation_space)
print("state_dim: ", state_dim)
action_dim = env.action_space.n # 행동 공간(action space)의 개수, action_dim = 2
print("action_dim: ",action_dim)

# # 공통 신경망
# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 128), nn.ReLU(),
#             nn.Linear(128, 128), nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, x):
#         return self.model(x)


# # DQN 알고리즘
# class DQNAgent:
#     def __init__(self):
#         self.q_net = MLP(state_dim, action_dim).to(device)
#         self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
#         self.memory = deque(maxlen=10000)
#         self.batch_size = 64
#         self.epsilon = 1.0
#         self.eps_min = 0.01
#         self.eps_decay = 0.995

#     def act(self, state):
#         if np.random.rand() < self.epsilon:
#             return env.action_space.sample()
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         return self.q_net(state).argmax().item()

#     def memorize(self, s, a, r, s_, d):
#         self.memory.append((s, a, r, s_, d))

#     def train(self):
#         if len(self.memory) < self.batch_size:
#             return
#         minibatch = random.sample(self.memory, self.batch_size)
#         s, a, r, s_, d = zip(*minibatch)
#         s = torch.FloatTensor(s).to(device)
#         a = torch.LongTensor(a).unsqueeze(1).to(device)
#         r = torch.FloatTensor(r).unsqueeze(1).to(device)
#         s_ = torch.FloatTensor(s_).to(device)
#         d = torch.FloatTensor(d).unsqueeze(1).to(device)

#         q_vals = self.q_net(s).gather(1, a)
#         q_next = self.q_net(s_).max(1)[0].unsqueeze(1)
#         target = r + GAMMA * q_next * (1 - d)

#         loss = nn.MSELoss()(q_vals, target.detach())
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# # A2C 에이전트
# class A2CAgent:
#     def __init__(self):
#         self.actor = MLP(state_dim, action_dim).to(device)
#         self.critic = MLP(state_dim, 1).to(device)
#         self.optimizer = optim.Adam(
#             list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR
#         )

#     def act(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         logits = self.actor(state)
#         dist = torch.distributions.Categorical(logits=logits)
#         action = dist.sample()
#         return action.item(), dist.log_prob(action)

#     def train(self, trajectory):
#         states, actions, rewards, log_probs = zip(*trajectory)
#         returns = []
#         G = 0
#         for r in reversed(rewards):
#             G = r + GAMMA * G
#             returns.insert(0, G)

#         states = torch.FloatTensor(states).to(device)
#         actions = torch.LongTensor(actions).to(device)
#         returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
#         log_probs = torch.stack(log_probs).to(device)

#         values = self.critic(states)
#         advantage = returns - values

#         actor_loss = -(log_probs * advantage.detach()).mean()
#         critic_loss = nn.MSELoss()(values, returns)
#         loss = actor_loss + critic_loss

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


# # PPO 에이전트
# class PPOAgent:
#     def __init__(self, clip_eps=0.2):
#         self.actor = MLP(state_dim, action_dim).to(device)
#         self.critic = MLP(state_dim, 1).to(device)
#         self.optimizer = optim.Adam(
#             list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR
#         )
#         self.clip_eps = clip_eps

#     def act(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         logits = self.actor(state)
#         dist = torch.distributions.Categorical(logits=logits)
#         action = dist.sample()
#         return action.item(), dist.log_prob(action), dist

#     def train(self, trajectory):
#         states, actions, rewards, old_log_probs = zip(*trajectory)
#         returns = []
#         G = 0
#         for r in reversed(rewards):
#             G = r + GAMMA * G
#             returns.insert(0, G)

#         states = torch.FloatTensor(states).to(device)
#         actions = torch.LongTensor(actions).to(device)
#         returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
#         old_log_probs = torch.stack(old_log_probs).detach().to(device)

#         for _ in range(4):  # K epochs
#             logits = self.actor(states)
#             dist = torch.distributions.Categorical(logits=logits)
#             new_log_probs = dist.log_prob(actions)
#             ratio = torch.exp(new_log_probs - old_log_probs)
#             values = self.critic(states)
#             advantage = returns - values.detach()

#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
#             actor_loss = -torch.min(surr1, surr2).mean()
#             critic_loss = nn.MSELoss()(values, returns)
#             loss = actor_loss + critic_loss

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()


# # 실험 실행 함수
# def run(agent_class, label):
#     scores = []
#     agent = agent_class()
#     for ep in range(EPISODES):
#         state, _ = env.reset()
#         done = False
#         score = 0
#         trajectory = []
#         while not done:
#             if label == "DQN":
#                 action = agent.act(state)
#                 next_state, reward, done, _, _ = env.step(action)
#                 agent.memorize(state, action, reward, next_state, done)
#                 agent.train()
#             else:
#                 action, *log = agent.act(state)
#                 next_state, reward, done, _, _ = env.step(action)
#                 if label == "A2C":
#                     trajectory.append((state, action, reward, log[0]))
#                 else:
#                     trajectory.append((state, action, reward, log[0]))  # PPO
#             score += reward
#             state = next_state
#         scores.append(score)
#         if label != "DQN":
#             agent.train(trajectory)
#         if (ep + 1) % 10 == 0:
#             print(f"[{label}] Episode {ep+1}: Avg Score = {np.mean(scores[-10:]):.2f}")
#     return scores


# dqn_scores = run(DQNAgent, "DQN")
# a2c_scores = run(A2CAgent, "A2C")
# ppo_scores = run(PPOAgent, "PPO")

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
