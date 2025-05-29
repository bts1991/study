# Reinforcement Learning
## 목표
- 에이전트가 환경과 상호작용하면서 보상을 최대화하는 최적의 정책(policy)을 학습

| 구성 요소                     | 설명                                 |
| ------------------------- | ---------------------------------- |
| **에이전트 (Agent)**          | 학습하고 행동을 결정하는 주체                   |
| **환경 (Environment)**      | 에이전트가 상호작용하는 세계                    |
| **상태 (State)**            | 현재 환경의 상황 정보                       |
| **행동 (Action)**           | 에이전트가 취할 수 있는 선택지                  |
| **보상 (Reward)**           | 에이전트의 행동에 대한 피드백 (정수나 실수)          |
| **정책 (Policy)**           | 상태에서 행동을 결정하는 전략 $\pi(a \mid s)$   |
| **가치함수 (Value Function)** | 미래 보상의 기대값 추정 (예: $V(s), Q(s, a)$) |
| **모델 (Model)** *(선택적)*    | 환경의 동작을 모사하는 함수 (모델 기반 RL에서 사용)    |

## Q-Learning
- Greedy action: Q값이 큰 방향으로 나아간다.
- Exploration: 더 좋은 길을 탐험한다.
  - $\epsilon$-Greedy
    - $\epsilon$은 0과 1의 값
    - $\epsilon$이 커질 수록, Random하게 움직이는 경향이 커짐
  - 장점
    - 새로운 Path를 찾을 수 있다
    - 새로운 맛집(더 많은 보상을 주는)을 찾을 수 있다.
  - Decaying $\epsilon$
    - $\epsilon$의 크기를 점점 0으로 줄여나감
    - 처음에는 탐험을 열심히 하다가, 길이 밝아지면 Greedy 해진다.
- Exploration VS Exploitation(Greedy action)
  - trade-Off 발생
  - $\epsilon$만큼은 Random 하게 움직이고
  - 나머지는 Greedy 하게 움직인다.
- Discount factor. $\gamma$
  - 여러 path 중 더 효율적인 것은? 에 대한 해답
    - $\gamma$는 0과 1 사이의 값
    - 목적지로부터 한 걸음 멀어질수록(시작점 방향으로) $\gamma$를 계속 곱함
    - 따라서, $\gamma$가 덜 곱해진 방향을 더 효율적이라고 인식하게 됨
  - 현재 VS 미래 중 무엇이 더 중요한가? 에 대한 해답
    - $\gamma$가 작을수록 현재 시점에서 먼 미래 Reward의 가치가 떨어짐
    - 따라서, 현재 시점의 Reward에 더 충실하게 됨
- Q-update
  - t+1 시점의 Q값을 t 시점으로 가져올 때, 그대로 $\gamma$를 곱하지 않고, '부드럽게' 가져온다.
  - $$Q(s_t, a_t) \leftarrow (1 - \alpha) \cdot Q(s_t, a_t) + \alpha \cdot \left( R_t + \gamma \cdot \max_{a_{t+1}} Q(s_{t+1}, a_{t+1}) \right)$$
    - $Q(s_t, a_t)$는 기존의 Q값과 새로운 정보의 조합으로 업데이트 됨
      - 새로운 정보란?
        - 현재 받은 보상 $R_t$와
        - 다음 상태에서 가장 큰 Q값인 $\max_{a_{t+1}} Q(s_{t+1}, a_{t+1})$

    - $\alpha$는 learning rate로 기존의 Q값과 새로운 정보의 비중을 결정
      - $\alpha$가 클수록 새로운 정보에 더 민감

## MDP(Markov Decision Process)
- Decision는 action(a0, a1, a2...) 이다.
    s0 → a0 (s와 a는 하나의 세트)
    ➯ s1 → a1
    ➯ s2 → a2
  - 주요 특징1: 모든 s와 a가 랜덤하다.(어떤 확률 분포를 가지고 있다.)
  - 주요 특징2: 다음 상태는 현재 상태와 행동만으로 결정됨 (과거는 필요 없음)
     1. P(a1|s0,a0,s1) ➯ P(a1|s1)
        1. s0에서 a0이라는 행동을 했고, s1으로 넘어왔다. 이 상황에서 a1일 확률은?
        2. 이 때, s0과 a0이 지워진다... 왜?
           1. 이미 s0과 a0에 의한 정보가 s1으로 흡수되었기 때문
           2. 따라서, s1을 알고 있다면, s0과 a0은 필요 없다.
     2. P(s2|s0, a0, s1, a1) ➯ P(s2|s1, a1)
        1. s0에서 a0이라는 행동을 했고, s1으로 넘어왔고 a1이라는 행동을 했다. 이 상황에서 s2일 확률은?
        2. s0과 a0는 멀리 있으니 지워지고, s1과 a1의 세트는 s2를 결정하기 때문에 남겨져야 한다.
     3. 정책이란?
        1. $P(a_t|s_t)$: $s_t$상태에서 $a_t$행동에 대한 확률 분포
        2. MDP에서 action을 결정
     
- 강화학습의 Goal는 Maximize Expected Return
  - Return의 정의
    - $$G_t=R_t+\gamma R_{t+1}+\gamma ^2 R_{t+2}+\gamma ^3 R_{t+3}+\dotsb$$
      - $\gamma$는 할인율
      - $G_t$는 discount 된 reward의 합
    - Why "Expected" ? 
      - action이나 state가 모두 random이기 때문에, 여기서 발생하는 reward도 random
    - $R_t=R(s_t,a_t)$
      - 현재 상태 $s_t$에서 에이전트가 행동 $a_t$를 선택하고 그 결과로 환경이 $s_{t+1}$로 천이되었을 때, 그 환경이 제공하는 보상값
  - 따라서, Expected Return을 최대화 하는 action 즉, policy라고 부르는 $P(a_t|s_t)$을 찾는 것이 최종 목표

## 상태 가치 함수 V, 행동 가치 함수 Q, Optimal policy
- 개념
  - State Value Fuction(상태 가치 함수)
    - 지금 State부터 기대되는 Return
    - 즉 현재 상태가 의미하는 가치
  - Action Value Function(행동 가치 함수)
    - 지금 action으로 부터 기대되는 Return
  - Optimal policy
    - State Value Fuction을 Maximize 하는 것
    - 지금으로부터 기대되는 Return을 최대화 하는 것이기에, 과거는 신경쓰지 않음
- 수식
  - $$G_t=R_t+\gamma R_{t+1}+\gamma ^2 R_{t+2}+\gamma ^3 R_{t+3}+\dotsb$$
    - $$E[x] = \int x \, p(x) \, dx$$
      - 의미: 확률 변수 x의 기댓값은 x 값과 그 확률 밀도 p(x)를 곱한 것을 전 범위에 걸쳐 적분한 값
      - x: 확률 변수
      - p(x): x에 대한 확률 밀도 함수 (PDF; Probability Density Function)
      - dx: 작은 x의 변화량
      - p(x)⋅dx 는 p(x)와 dx가 이루는 확률 밀도 함수 내에서의 어떤 면적이므로 곧 확률
      - x에 대해 적분한다는 것은?
        - 그 변수가 가질 수 있는 모든 값을 다 고려해서, 그 값에 따라 어떤 함수의 값을 **합산(평균)**하는 것
  - 상태 가치 함수
    - $$V(s_t) \triangleq \int G_{t} \, p(a_t, s_{t+1}, a_{t+1}, \dots | s_t) \, da_t \cdot da_{t+1} \cdots$$
    - 의미
      - 시점 t에서 상태 $s_t$에 있을 때, 미래에 받을 것으로 기대되는 총 보상(return)의 기댓값
    - $G_t$: 시점 t부터 시작해서 이후에 받는 누적 보상
    - $p(a_t, s_{t+1}, a_{t+1}, \dots | s_t)$
      - 조건부 확률 밀도 함수
      - $s_t$ 에서 이후에 일어날 모든 가능한 행동($a_{t}, a_{t+1}, \dots$)과 상태 전이 ($s_{t+1}, s_{t+2}, \dots$)로 만들어지는 연속 확률 분포
        - 'path1': 's0 → a0 → s1 → a1 → s2', 'prob': 0.4, 'G': 10
        - 'path2': 's0 → a0 → s1 → a0 → s2', 'prob': 0.2, 'G': 8
        - 'path3': 's0 → a1 → s2', 'prob': 0.3, 'G': 6
        - 'path4': 's0 → a1 → s1 → a0 → s2', 'prob': 0.1, 'G': 9
    - $da_t \cdot da_{t+1} \cdots$
      - 모든 가능한 행동과 상태 전이 경로 하나하나에 대해 적분함을 의미
  - 행동 가치 함수
    - $$Q(s_t, a_t) \triangleq \int G_t \, p(s_{t+1}, a_{t+1}, s_{t+2}, a_{t+2}, \dots \mid s_t, a_t) \, ds_{t+1} \, da_{t+1} \cdots$$
    - 의미
      - 현재 시점에서 어떤 행동을 했다는 가정 하에 총 보상의 기댓값
  - Optimal policy
    - 상태 가치 함수인 $V(s_t)$를 maximize 하는 [$P(a_{t},s_{t})$, $P(a_{t+1},s_{t+1})$, $P(a_{t+2},s_{t+2})$, $P(a_{t+3},s_{t+3})$, $\dots
    $]

## Bellman equation



질문1: Q값 업데이트 공식을 반복하면 행동 가치 함수 Q가 나오는 것인가?

