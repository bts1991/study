# Jean-Ho Kim†, Eun-Hong Park†, Ha Young Kim*, "Optimizing Stock Price Prediction via Macro context and Filter-DQN Framework," Journal of The Korea Society of Computer and Information (한국컴퓨터정보학회논문지), vol. 30, no. 01, Jan. 2025.(KCI).

## 한계
첫째, 국제와 국내 지표를 통합적으로 충분히 활용하지 못함
    - 한국의 KOSPI 지수는 미국, 유럽, 일본 등 주요 국제 시장의 움직임에 민감
    - 국제 시장과의 복잡한 상호작용을 충분히 반영하지 못한 채 주식 시장 내재적인 변동성만을 설명하거나 상승과 하락의 분류로 접근
둘째, 다량의 변수 중 중요한 정보를 효과적으로 선별하지 못함
    - 차원적이고 이질적인 변수들이 포함될 경우 불필요하거나 상관성이 낮은 변수로 인하여 모델의 복잡성이 증가하고 예측 성능이 저하되는 문제
    - 고차원 데이터에서 변수 선택이 이루어지지 않을 경우 과적합 위험이 존재 하며, 모델의 복잡성 증가로 인하여 학습 속도가 저하될 가능성
  
## 개선
주요 수출 대상국의 시장 동향과 국제적인 거시경제 변수를 포함
    - 국제 거시경제 지표와 주요 수출 대상국의 주식시장 지표를 포함해 다양한 변수로 구성되어 있는 데이터 셋에서 유의미한 변수를 선별
Filter-DQN 방법을 제안
    - 다변량 금융 시계열 데이터에서 상호 정보량(Mutual Information, MI)[10]을 기반으로 가장 유의미한 변수의 하위 집합을 강화 학습(Reinforcement Learning) 방식으로 찾아내는 Filter-DQN을 제안
  
## 방법론
1. 변수 간의 상호 정보를 DQN(Deep Q-Network)[11]의 에이전트(Agent)에게 보상(Reward)으로써 부여함으로써 학학습시키는 Filter-DQN 기법을 통해 다변량 데이터에서 유의미한 변수의 하위 집합을 선별
   1. Filter-DQN Framework: 기존 필터 기법에서 활용되었던 상호 정보 이론과 DQN을 결합한 프레임워크
2. 선별된 변수들을 입력으로 활용하여 Transformer 기반 시계열 예측 모델[12-14] 을 학습시키고, 이를 통해 KOSPI 종가 지수의 변동성을 정교하게 예측
3. 절차
   1. 훈련 데이터의 무작위 시점 시계열 정보를 상태로써 입력
      1. $s_t \in \mathbb{R}^F$: F개의 변수로 구성된 시계열 상태 벡터입니다. 예를 들어, 여러 기술적 지표 값이 포함된 시계열 입력
      2. 시점 선택: $t \sim \text{Unif}(0, L - 1)$
         1. 전체 시계열 길이 L 중 무작위 시점 t를 균등 확률로 선택
      3. 해당 시점의 상태 획득: $s_t = x(t, 0)$
         1. 시점 t의 상태 $s_t$는 시계열 입력 x의 t시점 벡터

   2. 에이전트는 어떤 변수를 선택 할지 행동을 결정
      1. $a_t \in \mathbb{R}^F$: 어떤 변수들을 선택할지를 나타내는 벡터. 예를 들어, 어떤 피처(feature)를 선택해서 다음 단계 예측에 사용할지를 선택
      2. 행동 $a_t$ (변수 선택 벡터) 생성
            1. 탐색 (Exploration)
               1. $a_t \sim \text{Bernoulli}(0.5)^F$
               2. 여기서 F는 변수의 개수로, F 차원에 대해 베르노이 분포가 있음을 의미
            2. 활용 (Exploitation)
               1. $a_t = \arg\max_a \, Q(s_t, a)$
               2. Q-네트워크로부터 행동 가치가 최대가 되는 a 선택
   3. 행동에 의해 선택된 변수와 목적(타겟) 변수와의 상호 정보량을 기반으로 보상을 부여
      1. $r_t$: 선택한 변수들로 예측한 값의 품질을 기반으로 부여되는 보상. Mutual Information(상호정보량)을 사용해 정의
      2. 선택된 변수 $x^{F'\times L}$와 타겟 변수 $x^{F_{\text{target}}\times L}$ 간의 상호정보량을 계산
         1. **타겟 변수**를 어떻게 정의하지?
         2. $r_t = \mathrm{MI}\left(x^{F'\times L} ,\ x^{F_{\text{target}}\times L} \right)$
            1. 상호정보량은 서로 간의 정보의존성을 측정. 값이 클수록 예측에 유용한 변수임을 의미
               1. $\mathrm{MI}(X; Y) = D_{\mathrm{KL}} \left( P(X, Y) \,\|\, P(X)P(Y) \right)$
               2. $P(X, Y)$엔트로피와 $P(X)P(Y)$의 엔트로피의 차이
            2. 상호정보량은 무조건 0보다 크거나 같음
         3. 상호정보량이 크다는 것은 "**내가 선택한 변수들을 보면, 타겟이 어떤 값일지 많이 알 수 있다**"는 의미
   4. 반복되며 목적 변수와 높은 상호 정보를 가지게 되는 변수들에 대한 행동 가치를 에이전트가 학습
      1. Filter-DQN은 보상을 최대화하는 Q-네트워크의 정책을 학습하는 것이 목표
         1. State vector를 input으로 하고, 여러 층을 거쳐 Q-value를 산출
         2. 산출된 Q-value와 타겟값 과의 손실함수를 학습에 활용
      2. 안정적인 학습을 위해 메인 네트워크($Q_{main}$)와 이를 복제한 타깃 네트워크($Q_{target}$)로 구성
         1. 타깃 네트워크는 일정 스텝마다 메인 네트워크와 동기화 되도록 설계
         2. $Q_{target}$은 지도학습에서의 정답과 같은 역할은 하지만, 실제 정답이 아닌 '그럴듯한 정답값'으로, 학습을 위한 기준점 역할을 하는 조금 더 안정적인 추정값으로 간주된다.
      3. 최종적으로 Filter-DQN은 메인 네트워크와 타겟 네트워크의 MSE 손실 함수를 최소화하며 최적 정책을 학습
         1. $y_{\text{target}} = r_t + \gamma \, \max_{a_{t+1}} Q_{\text{target}}(s_{t+1}, a_{t+1})$
            1. $\gamma \, \max_{a_{t+1}} Q_{\text{target}}(s_{t+1}, a_{t+1})$: t+1 시점의 Q값 중 가장 큰 것
         2. $L(\theta) = \mathbb{E}\left[\left(y_{\text{target}} - Q_{\text{main}}(s_t, a_t)\right)^2\right]$
            1. $\theta$ = $P(a_t|s_t)$: $s_t$상태에서 $a_t$행동에 대한 확률 분포
   5. 상위 k개의 변수를 필터링하고, 이를 통해 생성된 하위 변수 집합을 출력
   6. Transformer 기반 예측 모델[12-14]에 입력되어 시간 흐름에 따른 변동성과 변수 간의 상관관계를 학습

## 기대 효과
계산 비용을 절감하고 예측 성능을 극대화

## 실험
1. Features of the Dataset: 3가 카테고리로 분류
2. 데이터의 기간
   1. 전체: 1996년 1월 4일부터 2024년 8월 30일까지
   2. 훈련(80%), 검증(10%), 테스트(10%)
      1. 훈련: 1996년 1월 4일부터 2018년 4월 24일까지 5651일의 데이터
      2. 검증: 2018년 4월 24일부터 2021년 1월 6일까지 663일의 데이터
      3. 테스트: 2021년 1월 6일부터 2024년 8월 30일까지 663일의 데이터
3. 변수 개수: 40개
4. 평가 지표: MSE, MAE, MAPE(Mean Absolute Percentage Error)
   1. $$\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{Y}_i - Y_i \right)^2$$
   2. $$\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| \hat{Y}_i - Y_i \right|$$
   3. $$\mathrm{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{\hat{Y}_i - Y_i}{Y_i} \right|$$
5. 예측 모델: Nonstationary-Transformer,Autoformer, Crossformer


## 성능 평가
1. Feature 별 스피어만 상관계수 측정
   1. Filter-DQN을 거치지 않은 전체 데이터 셋은 Fig. 2의 위와 같이 상관관계가 뚜렷하지 않은 변수들이 포함되어 있음을 확인할 수 있다. 그러나 제안 방법을 통해 선택된 변수들은 높은 양 또는 음의 상관관계를 가지는 변수들로 선별
2. 특히 Nonstationary-Transformer에서 제안된 방법이 실제 추세와 가장 유사하게 예측


## 나의 생각
코스피를 예측하기 위해 다양한 변수들을 고려(미국주가지수, 일본주가지수, 금가격, 원유가격등 국제적 지수 + 이동평균선 등의 기술적 지표)하였고
많은 변수 중 중요한 변수들을 유의미하게 선택하여 예측을 위한 입력으로 활용


## Feature Selection
예측 모델과는 독립적으로 계산된 통계량을 기반으로 상관성이 높은 변수를 선별하는 방식
상호 정보량은 두 변수 간 상호 의존성을 측정하는 지표로, 변수 선택을 최적화하고 예측 성능을 향상시키는 데 활용
상호 정보량을 활용하여 노이즈에 민감한 기존 상호 정보량의 단점을 극복하고 예측 정확도를 향상

### 상호 정보량

## Deep Q-Network
DQN는 행동 가치학습(Q-Learning) 알고리즘을 심층 신경망(Deep Neural Network)으로 확장
에이전트가 주어진 환경(Environment)에서 행동(Action)과 그에 따른 상태 천이(State Transition), 보상을 반복 학습하여 누적 보상을 극대화

MDP(Markov Decision Process)로 정의된 이산 상태 공
간에서 순차적 의사결정 문제를 모델링

### 강화학습 기초
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

### MDP(Markov Decision Process)
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

### 상태 가치 함수 V, 행동 가치 함수 Q, Optimal policy
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

### Bellman equation