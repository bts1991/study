# Introduction
- rapid growth of multimedia platforms(news outlets and social media) ➡️ comprehensive evaluations about identifing potential investment
- real-world investment: subtle insights from financial news ➡️ significant profit indicators
- stock price signals inherently contain randomness(랜덤 워크 이론) ➡️ predicting the deterministic component(예측 가능한 요소) brought about by news
- two feature modalities in financial time-series forecasting
  - multivariate time-series features
    - opening price, closing price, and volume
  - discrete tabular features
    - technical indicators calculated based on historical trading signals
      - 가격·거래량 등 시계열 데이터를 기반으로 일정한 계산식(공식)을 적용하여 도출된 숫자형 요약 지표들
      - MA (이동평균선): 일정 기간 동안의 평균 가격. 상승/하락 추세 식별
      - MACD: 장·단기 이동 평균선 간 차이를 통해 추세 전환 포착
      - 왜 "이산형(discrete)"일까?
        - 하루 단위, 혹은 특정 윈도우(5일, 14일, 30일 등)를 기준으로 하나의 고정된 수치 값을 출력
        - 연속적인 시계열이 아닌, 고정된 시점에서의 하나의 값으로써 **이산적(discrete)
  - underlying assumption: trading signals for all stocks are mutually exclusive
    - 기존 연구들이 주식 간의 **상호작용(interaction)**을 무시하고 있다는 점을 비판
  - all stocks belong to the same financial system ➡️ each stock is inevitably affected by peer stocks
    - Why? momentum spillover effect(모멘텀 전이 효과)
      - factor movement:주식 수익률에 영향을 주는 공통 요인(Market factor, Value, Growth, Small/Big, Momentum)이 변하면서 개별 주식 또는 여러 섹터에 영향을 미침
      - lead-lag effect: 한 자산(또는 지표)의 가격 움직임이 시간적으로 선행(lead)하고, 다른 자산이 뒤따라서 반응(lag)하는 현상
        - t: 기술주 ETF 급등 ➡️ t+1: 삼성전자 상승
      - portfolio rebalancing: ETF나 기관 포트폴리오의 리밸런싱 ➡️ 포함된 여러 종목에 동시적으로 매수 또는 매도 압력
    - 현실에서는 주식 간 다음과 같은 상관관계(correlation) 또는 **영향 전이(spillover)**가 존재
      - 삼성전자 주가 상승 → SK하이닉스 주가도 반응
      - 테슬라 실적 발표 → LG에너지솔루션 영향
- advent of graph neural networks (GNNs): conceptualized the stock market as a complex network to model peer interactions
  - node: each stock
  - edge: relationship which is hard-coded microstructure
    - time-series correlation, supply chain, news co-occurrence
    - 학습을 통해 자동으로 만들어진 게 아니라, 기존의 외부 지식이나 규칙에 따라 미리 고정된 방식으로 구성
- static GNN structure: struggles to adapt to the dynamics of the real-world financial market
- graph attention networks (GATs): attempt to model adaptive interactions between stocks as stock attention networks
  - a long-tailed feature distribution in Real-world financial data for algorithmic trading
    - 소수의 종목은 매우 풍부한 정보를 가지고 있지만, 대부분의 종목은 거의 정보가 없는 불균형한 분포
    - 일부 인기 종목에 뉴스·데이터가 몰려 있고, 대부분의 종목은 정보가 거의 없음
  - Figure 1: Example of the long-tailed feature distribution
    - the news (known as tail features) dynamically covers only a fraction of stocks
    - investment insights from news may be overwhelmed by massive price-related head features
    - a biased attention effect: Breaking news from the financial sector that impact the overall stock market receive insufficient cross-sector attention.
      - 이러한 뉴스는 다른 섹터들로 충분히 확산되지 못함
- two main challenges
  - 1. the long tail effect in feature distribution(feature imbalanced problem)
    - bias toward the dominated head features
  - 2. resampling is not effective to overcome feature unbalance
    - why? future market landscapes are diverse and intricate
- turning point
  - events that affect specific stocks ➡️ an instantaneous dominance over their movements
    - But, few studies incorporated it into the model
  - near-equivalence between news and stock price movements
    ➡️ new resampling strategy: bypasses news and directly generates large amounts of putative(추정되는) news sentiments (treated as news) from stock price movements (labels) for data augmentation. 
  - graph attention mechanism to be pretrained
    ➡️ adapt to more complex and extreme news coverage
    ➡️ improving the model's generalization performance and addressing the long-tail feature distributions overlooked by existing methods.
  - redesign of the attention model architecture and the training strategy ➡️ seamless integration of pretraining and fine-tuning
    - simultaneously accommodate both news information and the generated putative news
    - transform the news into potential stock movements (or news sentiments)
- propose: prompt-adaptive trimodal model (PA-TMM)
  - prompt: 다른 종목들의 뉴스 감성이나 움직임을 요약한 벡터
  - designing two subnetworks
    - cross-modal fusion module: integrating trimodal features and extracting the news-induced sentiments as prompts for other stocks
    - graph dual-attention module: dynamically inferring the stock attention network by a graph dual-attention mechanism
      - circumvents(우회) direct similarity measurement of heterogeneous(서로 다른) representations(news, prices)
      - overcome biased attention
  - an equivalence resampling (EQSamp) strategy
    - tackle the tail feature scarcity problem
    - data augmentation by establislng a direct connection between market sentiments in news and stock movements by considering the dominant impact of news
  - pretrain our model using augmented data with generated prompts
    - proactively adapting to extreme feature unbalance
  - fine-tune the model with real-world data
    - mainly focusing on understanding news representations
- contributions
  - a learning framework named PA-TMM: effectively captupes news propagation dynamics by graph learning
  - targeted pretraining method named movement prompt adaptation (MPA)
    - respond to tai led news sensitively
    - prevents it from overfitting due to over-reliance on stocks canying news
  - EQSarnp strategy
    - financial data augmentati on when pretraining to overcome the news scarcity problem
    - enhancing tl1e generalization ability of GNNs on feature-imbalanced datasets
# Related Work
## Time-Series Stock Prediction
- encoding an individual stock ➡️ a sequential latent representation ➡️ downstream tasks
  - RNNs based: LSTM, GRU, Transformer
  - capture the underlying tune-varying patterns from multiple time steps
- encode the time series for each stock using RNNs
  - PEN [40], MAN-SF [12], and MTR-C [41]
- mningling different types of market factors
  - relational event-driven stock trend forecasting (REST) [44]
    - utilizes the event information from the company's announcements
- produce powerful high-frequency stock factors
  - Digger-Guider [45]
  - significantly improve stock trend prediction performance
- But, 여전히 문제가 되는 Assumption
  - the trading signals of all stocks are mutually exclusive
  - Why? financial markets are highly internally coupled, momentum spillover
## Graph-Based Stock Prediction
- 시장 현실: the movement of each entity is inevitably in fluenced by its peer entities [46]
  - a lead-lag effect in the stock market
- conceptualize the stock market as a graph
  - To model this intraindustry phenomenon
  - node: each entity
  - edges: relations
    - industry category [2], supply chain [23], business partnership [6], price correlation [3], lead-lag correlation [2], and causal effect [21]
- GNNs
  - THGNN [3] generates a temporal and heterogeneous graph for graph aggregation operation. ESTIMATE [2], utilizin g hypergraphs based on industry classifications, captures nonpairwise correlations among stocks. SAMBA [49] models dependencies between daily stock features by utilizing a bidirectional Mamba block and an adaptive graph convolution module
  - THGNN [3]: 하이퍼그래프를 통해 시간 및 관계 기반 특성을 통합
  - ESTIMATE [2]: 하이퍼그래프 및 웨이블릿 attention을 통해 주식 간 상관관계 포착
  - SAMBA [49]: 양방향 Mamba 블록 및 적응형 그래프 합성 모듈을 활용
- aggregating peer influences ➡️ update node representations to capture neighbor-induced movement
## News-Based Stock Prediction
- financial news [40], [50], [51] or social media posts [7], [20]
  - external information beyond the trading market
- the graph convolutional networks (GCNs)
  - multi-source aggregated classification (MAC) [l]
  - aggregate the effects of news on related companies
- aggregate various features such as technical indicators and textual news
  - NumHTML [52] and multi-view fusion network (MFN) [53]
- adapt to the market dynamics
  - the time-varying structure of stock networks: combining stock interactions and news information
  - AD-GAT [15] and DANSMP [6]: 




- 정형 재무 정보 외에 뉴스나 소셜 미디어 정보를 활용해 추가적인 인사이트 획득
  - MAC (Multi-source Aggregated Classification) [1]: 기술적 지표와 뉴스 텍스트를 결합하여 예측
  - NumHTML [52], MFN (Multi-view Fusion Network) [53]: 다양한 피처(뉴스, 기술지표 등)를 통합
- 뉴스와 가격 신호의 상호작용을 반영한 그래프 기반 모델
  - AD-GAT [15], DANsmp [6]: 뉴스 및 시계열 기반 엣지를 반영한 GAT 변형
  - MSMF [54]: modality 보완성과 다양성을 고려해 구성한 블렌딩 네트워크
- 롱테일 문제에 대해 충분한 고려가 없으며, 다양한 모달리티를 단순히 **병합(concatenation)**하는 방식
- 뉴스와 가격 정보 간 상호작용을 깊이 있게 활용하기 어렵고, 뉴스 전파(news propagation) 메커니즘을 모델링하기에도 한계
- 본 논문은 롱테일 분포를 명시적으로 고려한 pretraining 전략을 통해 그래프 attention 모델의 일반화 성능을 향상시키는 방식을 제안
# Problem Statement
- 일반적으로 시계열 예측(forecasting)은 회귀 문제로 간주
- 주가 방향성을 예측하는 것이 그 자체의 가격을 예측하는 것보다 더 중요한 문제로 간주
- 기존 연구들은 주식의 가격 시계열을 활용하여, 특정 시점의 **특징(features)**을 기반으로 그 다음 날의 가격이 상승할지 하락할지를 분류하는 모델을 학습하는 방식으로 접근
- 
# PA-TMM Architecture
## Cross-Modal Fusion Module
## Graph Dual-Attention Module
## Computational Complexity
# Model Optimization
## Model Pretraining: MPA
## Model Fine-Tuning
# Experiments
## Evaluation Setup
## Stock Movement Prediction
## Ablation Study
## Backtesting Profitability
## Stress Test During Market Crash
## Parameter Sensitivity Analysis
## Case Study on Exploring Stock Attention Networks
# Conclusion