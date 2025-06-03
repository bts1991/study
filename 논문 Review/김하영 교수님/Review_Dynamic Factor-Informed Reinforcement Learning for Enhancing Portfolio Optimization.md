# Literature review
- recent studies have primarily focused on enhancing technical aspects such as model 
architecture through the application of deep learning or reinforcement learning, knowledge of factor 
portfolios, grounded in modern portfolio theory, remains paramount.
## Traditional portfolio theory
- Modern Portfolio Theory (MPT)(Markowitz 1952): “투자 자산을 똑똑하게 나눠서, 위험은 줄이고 수익은 높이자”
  - establishing a framework for optimizing asset allocation
  - the mean (average) and standard deviation, to quantify the return and risk inherent in stocks: 평균 수익률이 동일해도, 더 들쭉날쭉 한 자산의 위험이 크다
  - considering covariance: assessment of risk within stock portfolios
    - standard deviation: 한 자산의 수익률 변동성
    - covariance: 두 자산 간 수익률의 연동성
      - 공분산: 두 변수에 대해 편차를 서로 곱하고 더한 후 n-1로 나눈 값
        - 분산: 하나의 변수에 대해 편차 제곱 평균
      - 두 자산이 같은 방향으로 움직여야 공분산이 커짐

⬇️
- asset pricing model (CAPM)(Sharpe 1964): compensates investors for assuming systematic risk, i.e., market risk, alongside total risk.
  - 결론: 각 자산이 가져야 할 정당한(균형 잡힌) 수익률 계산
  - 전제: 자본시장이 효율적이고 모든 투자자가 합리적으로 행동한다
  - 빙법: 개별 자산이 시장 전체와 얼마나 함께 움직이는지를 **체계적 위험(systematic risk)**에 대한 민감도로 측정하여 얼마의 추가 보상이 필요한지를 '위험 프리미엄(risk premium)'으로 정의
  - 공식: $E(R_i)=R_f+\beta_i \sdot (E(R_m)-R_f)$
    - $E(R_i)$: 자산 i의 기대 수익률
    - $R_f$: 무위험 수익률(예: 국채 금리)
    - $E(R_m)$: 시장 전체의 기대 수익률(예: S&P500)
    - $\beta_i$: 자산 i의 베타 값
      - 공식: $βᵢ = \frac{\mathrm{Cov}(Rᵢ, Rₘ)}{\mathrm{Var}(Rₘ)}$
      - 개별 자산과 시장의 상관관계를 시장의 변동성 단위로 정규화해서 표현
    - $E(R_m)-R_f$: 시장 리스크 프리미엄
      - 위험을 감수했을 때 시장에서 기대할 수 있는 초과 수익률
  - 한계: 단일 요인(시장 수익률)만 고려려

⬇️
- Arbitrage Pricing Theory (APT) (Ross 1976)
  - 결론: 자산의 기대 수익률은 여러 가지 경제적 위험 요인(factors)에 의해 결정된다, foundation for factor-based portfolio investment strategies
  - 전제: absence of arbitrage opportunities, 시장에서 무위험 이익(차익거래)이 존재하지 않는다, 동일한 위험을 가진 자산은 동일한 수익률을 가져야 한다
  - 방법: 인플레이션, 금리, 산업 생산, 유가, 환율 등 다양한 거시경제 요인들이 자산 수익률에 영향을 준다고 봄
  - 공식: $E(R_i) = R_f + b_1 F_1 + b_2 F_2 + \cdots + b_n F_n$
  
| 기호      | 의미                        |
| -------- | ------------------------- |
| $E(R_i)$ | 자산 i의 기대 수익률              |
| $R_f$    | 무위험 수익률                   |
| $F_n$    | 위험 요인 (factor)의 프리미엄      |
| $b_n$    | 자산이 그 요인에 얼마나 민감한지 (감마 값) |

⬇️
- The Fama-French three-factor model(Fama and Kenneth 1993)
  - 결론: portfolios comprising value stocks with a high book-to-market ratio and small-cap stocks with a low market capitalization exhibit superior returns
  - 방법: incorporating value and size factors in conjunction with the CAPM

⬇️
- Carhart (1997) introduced the momentum factor, another crucial risk factor
  - capture the persistence of past returns into the future

⬇️
- Fama and French (2015) introduced the concept of profitability(=quality) and investment
  - profitability factor: net income to capital
    - with a high profitability factor tend to have higher expected returns
  - investment factor
    - "설비 투자도 많이 하고, 빠르게 성장하는 회사가 수익도 좋다"는 생각을 반박
    - 오히려 과도한 투자는 낭비나 수익성 악화로 이어질 수 있음

⬇️
- the low volatility factor (Baker et al. 2011)
  - stocks with lower volatility exhibit higher expected returns
### Limitaion
- they rely on static models that lack the adaptability required to respond to rapid market shifts (Bertola 1988)
- not fully capture the complexity of modern, dynamically evolving markets characterized by data irregularities and rapid fluctuations
## related works
- leveraged DL and RL techniques for portfolio optimization ( Heaton et al., 2017; Wang et al., 2021; Ma et al., 2021; Li et al., 2021): outperformed traditional financial portfolio methods , but not incorporating financial domain expertise
- Yang et al. (2020) back tested Dow Jones 30 stocks
  - ensemble trading strategy: incorporates three actor-critic-based algorithms: Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gadient (DDPG)
- Wu et al. (2021) leverages neural networks, specifically Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN)
  - predicting future stock prices
  - predictions were used as inputs for the RL model
- Wang et al. (2021) proposed a novel ensemble portfolio optimization framework
  - Mode decomposition (모드 분해) + Bi-LSTM (양방향 LSTM) + RL-based model
    - 모드 분해: 시계열 데이터를 여러 주기(주기성 패턴)로 나누어 분석
    - 양방향 LSTM: 과거와 미래 방향의 정보를 모두 고려
- Sun et al. (2024) improved the architecture of the PPO agent
  - introducing a GraphSAGE-based feature extractor
  - capture complex non-Euclidean relationships between market indices, industry indices, and stocks
    - PPO agent: 강화학습 알고리즘, 안정적이고 효율적인 학습을 위해 정책(policy)의 급격한 변화를 제한
    - GraphSAGE-based feature extractor: 그래프 신경망(GNN)의 한 종류로,
각 노드(예: 종목, 지수)의 정보를 이웃 노드(관련 종목 등)로부터 **효율적으로 추출 및 요약(aggregate)**함
    - non-Euclidean relationships: 금융 시장의 구조는 **복잡한 연결 관계(예: 업종 내 상관성, 섹터 간 영향 등)**로 단순 직선 거리를 정의할 수 없음
- 딥러닝만 사용하는 게 아니라, 다양한 방법들과 통합한 대안적 접근 방식도 존재
  -  Chen et al. (2019) introduced a fuzzy grouping genetic algorithm
  -  Chen et al. (2021) devised a grouping trading strategy portfolio
- integrates existing finance theories with DL or RL
  - Jang and Seong (2023) combined modern portfolio theory and RL and solved a multi-mode problem using Tucker decomposition
  - Tucker decomposition
    - 고차원 텐서(3차원 이상)를 저차원 구조로 분해하는 텐서 분해 기법
    - PCA의 고차원 확장판
    - 복잡한 데이터를 **핵심 요인(core tensor)**과 **축별 요인(factor matrices)**로 압축해 분석
### Limitaion
- struggle to adapt to rapidly changing market conditions, integrate domain-specific financial insights, and maintain interpretability
- hinders flexibility in rapidly changing market environments.
- lacks the use of the most important factors in the financial 
market


# Methodology
- Integrating advanced techniques, such as Deep Learning (DL) and Reinforcement Learning (RL), with traditional factor strategies
- a novel hybrid portfolio investment method that integrates reinforcement learning with dynamic factors, called the dynamic factor portfolio model.
- The proposed model comprises two modules
- account for macroeconomic conditions and the unique characteristics of individual stock prices
## Dynamic factor portfolio model
- integrates traditional factor investment methodologies with RL techniques
- respond dynamically to shifting market conditions while leveraging the predictive capabilities of key investment factors
- comprises two primary components: the Dynamic Factor Module (DFM) and the Price Score Module (PSM)
- achieve adaptive, balanced portfolio allocations that maximize the Sharpe ratio while remaining robust to changing market conditions.
### Dynamic factor module
- 의미
  - computes scores based on key factors—size, value, beta, quality, and investment
  ➡️ updates each factor’s weight based on recent performance data
  ➡️ adaptively prioritize 
factors that are most relevant to current market conditions
    - Value: the relative valuation of an asset
      - Ex. price-to-earnings (P/E), price-to-book (P/B) ratios
      - Lower ratios indicate a higher potential for undervaluation.
    - Size: the market capitalization(시가 총액) of an asset, distinguishing between small-cap and large-cap stocks.
      - Small-cap stocks often exhibit higher growth potential
    - Beta: asset's sensitivity to overall market movements
      - a beta greater than 1 indicating higher volatility than the market and less than 1 indicating lower volatility.
    - Quality: profitability, earnings stability, and financial health
      - High-quality stocks are generally more resilient during market downturns.
    - Investment: Captures growth in capital expenditures or reinvestment rates, linked to the asset’s potential for future growth.
- 사용 모델
  - Temporal Attention-LSTM (TALSTM) = LSTM + Attention
  - 과거 시점 중, 어느 시점이 중요한지를 분석하는 것이 중점
  - 개요: macro market data ➡️ calculate factor importance weights (𝑀) ➡️ the impact of each risk factor on portfolio performance ➡️ integrating factor importance weights 𝑀 and factor data ➡️ 시장 상황을 반영한 다섯 가지 요인으로부터 계산된 자산별 값을 동적으로 나타내는 요인 점수(dynamic factor scores)를 학습
- 과정
  - Step 1: LSTM hidden states 생성
    - LSTM에 input으로 x(4x1) 입력하여 K개의 hidden states(32x1) 생성
      - data dimension (𝑃) = 4(the number of macroeconomic variables)
      - look back window size (K) = 18
  - Step 2: Attention score 계산
    - $$e_k = W_a^\top \cdot \tanh\left( W_b \cdot [\mathbf{h}_k ; \mathbf{h}_K] + W_c \cdot \mathbf{x}_k \right)$$
    - $W_c$: 18 x 4 (K x P)
    - $W_b$: 18 x 64 (K x 2H)
    - $W_a^T$: 1 x 18 (1 x K)
    - $\begin{aligned} e_k 
    &= (1 * 18)\cdot tahn((18*64\cdot [64*1])+(18*4)\cdot (4*1))\\
    &=(1*18)(18*1)\\
    &=(1)
    \end{aligned}$
  - Step 3: Softmax → Attention weights($\alpha_k$)
  - Step 4: Context vector 계산
    - $$c = \sum_{k=1}^{K} \alpha_k \cdot h_k$$ 
    - $c=\alpha_k\cdot (32*1)$
    - 현재 시점 기준으로, 과거 18개 시점의 데이터를 분석한 요약 결과
  - Step 5: Dense layer + tanh 적용
    - Dense layer를 통과하면 **출력 차원이 요인의 수 P=5**로 조정됨
    - 최종적으로 얻는 벡터 $( \mathbf{M} \in \mathbb{R}^5)$ 는 다음과 같습니다
      - $M = [M_{\text{value}},\; M_{\text{size}},\; M_{\beta},\; M_{\text{quality}},\; M_{\text{investment}}]$
      - 각 요인이 현재 시장 상황에서 얼마나 중요한지를 나타냄
  - Step 6: 종목별 Factor Score(stock 개수 * 5)와 Factor Importance(5、)와 multiplication
    - ➡️ 최종결과: DynamicFactorScore（stock 개수、） 



### Price score module
- consolidates stock price data, evaluating both inter-stock relationships and individual price patterns within a portfolio.
  - 과정: **weight-shared depthwise convolution** ➡️ **pointwise convolution**
- providing real-time price signals and stock-level insights.
- capture realtime price fluctuations
#### weight-shared depthwise convolution
- extract historical price information
- 과정
  - Input: each stock’s historical price data(t x n)
  - Kernel: 하나의 필터를 여러 자산에 동일하게 공유(k x 1）
  - Output: capture common patterns in price movements (t' x n)
    - t' = t-k+1
    - 점점 시간 축이 감소함(t ➡️ t' ➡️ t'')
- 결과: identifying correlations between price trends across different assets
#### pointwise convolution
- capture inter-asset relationships
- pointwise 과정
  - shape: C, H, W
  - Input: depthwise conv의 결과(t' x n)
    - ➡️ Reshpe: n x t' x 1
  - Kernal: 채널 수가 n개인 1 x 1 (stock의 수와 동일하게)
  - Output: n x t' x 1
    - ➡️ Reshpe: t' x n
  - (결국, (n x t')와 (n x n)의 곱과 동일)
- 결과: inter-stock correlations
- 왜 "pointwise"인가?
  - 커널 크기가 공간적으로 (혹은 시간적으로) **한 지점(1×1)**만 보기 때문
  - 커널 크기가 1×1인 합성곱 연산
  - 공간/시간은 고정한 상태로, 다수의 채널을 가로지르며 연산
### Integrated score module
- calculates the stock weight score
  - 과정: Factor scores + Price scores ➡️ normalized investment weights ➡️ SoftMax
- 수식
  - $$\text{Stock Weight Score} = W_p^T \tanh(W_F{FS} + W_T{PS})$$
  - (1,m) tanh((s,m)(m,)+(s,m)(m,))
  - = (1,m)(s, )???????
    - m: the number of assets, s: rebalancing period
## Model optimization
- train the DFPM with portfolio weights optimized for the Sharpe ratio
- rooted in Markov Decision Process (MDP) ➡️ maximizes the expectation of the sum of all discounted rewards
  - Agent Receives $s_t$ ➡️ Chooses an Action $a_t$
    - action: variable $w_i$(투자비율)
    - state: prices of assets, factors, and macroeconomic data(Input data)
    - rewards: ROE, Sharpe ratio
      - Sharpe ratio: primary reward metric to maximize portfolio profitability
## Data

# Experiment
- employing various portfolio selection methods and reward objectives to optimize the model.
## Baselines
## Implementation and hyperparameters
## Evaluation metrics

# Experiment Result
- With dynamic factor-informed knowledge, the proposed model can make portfolio decisions adaptively based on market conditions. 
- consistently outperformed traditional portfolio strategies and State-of-the-Art (SOTA) RL methods in risk-adjusted metrics, such as the Sharpe ratio and fAPV
- offers interpretability by identifying critical factors across varying market scenarios

## The effect of the dynamic factors
- adapts its portfolio decisions based on a comprehensive perspective that considers both macroeconomic trends and individual asset behaviors
- ability to capture market dynamics and deliver practical advantages in portfolio management
## Comparison with traditional portfolio strategies
## Comparison with state-of-the-art RL methods
## Comparisons of portfolio selection methods
## Comparison according to the reward objective
## Cross-Validation Results Analysis
## Analysis of factor importance

# Conclusion
- propose a novel RL-based portfolio optimization framework: the first framework that directly incorporates these five key factor indicators into an RL model for portfolio optimization.
- explainable RL framework: allowing for the assessment of factor importance under varying market conditions
- demonstrate the effectiveness of dynamic factors: DFPM outperforms traditional portfolio methods, reveal that DFPM surpasses recent SOTA RL models

# Literature review
## Traditional portfolio theory
- leverage a diverse set of factors that influence asset prices
- an asset’s return embodies a premium related to a specific risk factor
- prioritize investments in stocks undervalued in relation to this risk factor
- Ex: 시장 위험 요인(market risk factor); 자산의 **체계적 위험(systematic risk)**을 나타내며 **CAPM의 베타 계수(beta coefficient)**로 표현
## portfolio investment strategies
- leverage a diverse set of factors that influence asset prices
- factor portfolio strategies
  - using risk factors such as asset undervaluation and profit realization
## limitations
- not fully capture the complexity of modern, dynamically evolving markets characterized by data irregularities and rapid fluctuations
# 목적: effectively utilize the knowledge of factor investment strategies,
- When integrated with traditional factor strategies, DL and RL offer 
a more adaptive approach by combining stable relationships based on financial theories with the flexibility to respond to real-time market changes. 
- This hybrid approach captures static factor-based insights and dynamic market behaviors, effectively addressing the complexities of irregular time series and volatile financial environments.
# 방법: Dynamic Factor Portfolio Model(DFPM)
- a novel hybrid portfolio investment method
- integrates reinforcement learning with dynamic factors
  - dynamic factors: size, value, beta, investment, quality
- comprises two modules
  - a dynamic factor module: calculates a score 🠔 factors reflecting the macro market
  - a price score module: calculates a score 🠔 prices expressing the relationship between assets and their future value
- DFM and PSM outputs are combined to train the DFPM with portfolio weights optimized for the Sharpe ratio
  - achieving a dynamic balance between risk and return
- portfolio decisions based on a comprehensive perspective that considers both macroeconomic trends and individual asset behaviors
## Dynamic Factor Module (DFM)
- calculates adaptive scores for established investment factors
  - factors: size, value, beta, quality, and investment
    - derived from factor investment strategies and dynamically
    - adjusted by the DFM
      - based on prevailing market conditions, capturing the evolving importance of each factor
- integrate fundamental investment insights while making it responsive to broad economic trends that influence portfolio allocation- 
## Price Score Module (PSM)
- complements DFM
  - analyzing asset price data
  - assessing both inter-asset correlations and individual price patterns
  - thereby providing real-time price signals and stock-level insights

# 실험
## various portfolio selection methods
## reward objectives

# 기여: make portfolio decisions adaptively based on market conditions
- respond dynamically to shifting market conditions while leveraging the predictive capabilities of key investment factors, thereby enhancing portfolio performance
- enhancing profitability
- dynamically scores the five major factors based on market conditions
- explainable RL framework
  -  assessment of factor importance under varying market conditions.

# 성능
## outperforms both traditional portfolio investment strategies and existing reinforcement learningbased strategies
## offers interpretability by identifying critical factors across varying market scenarios


### factor portfolios
- 특정 투자 요인(Factor) 에 따라 자산을 선별하고 구성한 포트폴리오
- 자산가격결정모형인 Fama-French 3-factor / 5-factor 모델에서 매우 중요한 역할

| 팩터             | 팩터 포트폴리오 구성 방법         |
| -------------- | ---------------------- |
| Size           | 시가총액이 작은 주식만 모아 구성     |
| Momentum       | 최근 6개월 상승률이 높은 종목으로 구성 |
| Low Volatility | 변동성이 낮은 종목만 선별하여 구성    |

#### Fama-French 3-factor / 5-factor 모델
- 확장된 자본자산가격결정모형(CAPM의 확장)
  - CAPM은 수익률이 시장 위험(Market Risk, β)만으로 결정된다 가정
  - 그러나 실무에서는 소형주, 가치주 등이 시장보다 높은 초과수익을 기록하므로, 이를 설명하기 위한 요인(factor)을 추가
- CAPM
  - $$E(R_i)=R_f+\beta_i \sdot (E(R_m)-R_f)$$
    - $E(R_i)$: 자산 i의 기대 수익률
    - $R_f$: 무위험 수익률(예: 국채 금리)
      - 투자자가 아무 위험 없이 얻을 수 있는 수익률
    - $E(R_m)$: 시장 전체의 기대 수익률(예: S&P500)
    - $\beta_i$: 자산 i의 베타 값
      - 시장 전체 대비 자산의 민감도 (시장의 1% 수익 변화 시 자산이 얼마나 변하는가)
    - $E(R_m)-R_f$: 시장 리스크 프리미엄
      - 위험을 감수했을 때 시장에서 기대할 수 있는 초과 수익률

### factor investment strategies
- **자산(주식, 채권 등)의 수익률을 설명할 수 있는 요인(Factor)**들을 기반으로 자산을 선별하고 포트폴리오를 구성하는 전략
- 과거 데이터에서 수익률에 체계적인 영향을 준다고 밝혀진 특정 공통 요인(factor) 들에 따라 종목을 분류하고, 해당 팩터에 노출된 자산에 집중적으로 투자하는 방식
- 유형
  - 단일 팩터 전략: 하나의 팩터만 고려 (예: 저PBR 주식만 모아서 투자)
  - 멀티 팩터 전략: 여러 팩터를 결합하여 투자 (예: 저PBR + 고ROE + 저투자비용 조합)

| 팩터                    | 설명                                                   |
| --------------------- | ---------------------------------------------------- |
| **Size (크기)**         | 시가총액 기준으로, 일반적으로 소형주는 대형주보다 더 높은 기대수익을 가짐            |
| **Value (가치)**        | 저평가된 주식(예: PBR, PER이 낮은 주식)이 장기적으로 더 높은 수익을 가져올 수 있음 |
| **Beta (시장 민감도)**     | 자산이 전체 시장 변동에 얼마나 민감한지를 나타냄                          |
| **Investment (투자성향)** | 자산을 많이 투자(지출)한 기업은 수익률이 낮고, 절약형 기업은 수익률이 높을 가능성      |
| **Quality (재무 건전성)**  | 수익성, 안정성 등이 뛰어난 기업이 더 좋은 성과를 내는 경향  