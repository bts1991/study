# Literature review
- recent studies have primarily focused on enhancing technical aspects such as model 
architecture through the application of deep learning or reinforcement learning, knowledge of factor 
portfolios, grounded in modern portfolio theory, remains paramount.
## Traditional portfolio theory
- Modern Portfolio Theory (MPT)(Markowitz 1952):  investing and managing multiple assets
⬇️
- asset pricing model (CAPM)(Sharpe 1964): compensates investors for assuming systematic risk, i.e., market risk, alongside total risk.
⬇️
- factor portfolio strategies(L'Her et al. 2002): leverage a diverse set of factors that influence asset prices.
⬇️
⬇️
⬇️
⬇️
⬇️
### Limitaion
- they rely on static models that lack the adaptability required to respond to rapid market shifts (Bertola 1988)
- not fully capture the complexity of modern, dynamically evolving markets characterized by data irregularities and rapid fluctuations
## related works
- leveraged DL and RL techniques for portfolio optimization ( Heaton et al., 2017; Wang et al., 2021; Ma et al., 2021; Li et al., 2021): outperformed traditional financial portfolio methods , but not incorporating financial domain expertise

# Methodology
- Integrating advanced techniques, such as Deep Learning (DL) and Reinforcement Learning (RL), with traditional factor strategies
- a novel hybrid portfolio investment method that integrates reinforcement learning with dynamic factors, called the dynamic factor portfolio model.
- The proposed model comprises two modules
## Dynamic factor portfolio model
- integrates traditional factor investment methodologies with RL techniques
- respond dynamically to shifting market conditions while leveraging the predictive capabilities of key investment factors
- comprises two primary components: the Dynamic Factor Module (DFM) and the Price Score Module (PSM)
### Dynamic factor module
- adaptive scores for established investment factors, including size, value, beta, 
quality, and investment.
- factors are derived from factor investment strategies and dynamically adjusted by the DFM based on prevailing market conditions, capturing the evolving importance of 
each factor.
- making it responsive to broad economic trends that influence portfolio allocation
### Price score module
- analyzing asset price data, assessing both inter-asset correlations and individual 
price patterns
- providing real-time price signals and stock-level insights.
### Integrated score module
- Within the RL framework, DFM and PSM outputs are combined
## Model optimization
- train the DFPM with portfolio weights optimized for the Sharpe ratio
-  achieving a dynamic balance between risk and return
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