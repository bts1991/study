Intro

	안녕하세요.

	2025학년도 2학기 석사과정으로 MLCF 연구실에 지원한 방태성입니다.

	저는 정형 데이터와 비정형 데이터를 융합하여, 장단기적인 주가 흐름을 예측하는 연구에 관심을 가지고 있습니다.

	이번 랩테스트에서는 제 연구 관심과 밀접하게 연관된 논문인
	**"Responding to News Sensitively in Stock Attention Networks via Prompt-Adaptive Trimodal Model"**을 리뷰 주제로 선정하였습니다.

	이 논문은 2025년 6월, IEEE Transactions on Neural Networks and Learning Systems에 게재되었으며,

	시계열 데이터, 기술적 지표, 뉴스라는 세 가지 tri-modal 정보를 통합하고,
	뉴스가 시장에 미치는 영향을 민감하게 반영하기 위해 프롬프트 기반 학습 구조를 도입하여
	주가의 방향성과 변동성을 효과적으로 예측하는 모델을 제안하고 있습니다.


Outline
	발표 목차는 논문의 흐름과 동일하게 ~~~~ 의 순으로
	
Introduction-1

	초반부는 백그라운드의 변화
	그에 따라 등장하는 새로운 챌린지를 

	멀티미디어 플랫폼의 빠른 증가
	주가 예측을 하는데 결정적 요소
	
	특히, 뉴스와 관련

	연구자들의 관심

	//

	기존 주가 예측 여러 모델
	각각의 단점

	//

	여기서 중요한 2가지 챌린지 언급

	일부 주식만 뉴스 정보를 갖는 현실을 반영

	브레이킹 뉴스가 주목을 받지 못하고,
	가격 정보에 attention이 편향되는 문제

	뉴스 정보 자체의 부족으로 일반화 성능이 떨어지는 문제


Introduction-2

	논문에서 제시한 Figure 1
	일부 종목에만 뉴스가 존재하는 현실을 도식화

	피쳐 디스트리뷰션

	제이피모건, 웰파고만 뉴스 정보

	tail이 길어짐

	Stock Attnetion Network

	MS, Google, Apple이 
	
	뉴스 정보에 주목하지 않고 있는 현상


Introduction-3

	뉴스의 본성이 가지는 Idiosyncratic 특징이 터닝포인트

	브레이킹 뉴스

	즉각적으로 주가 움직임에 영향력을 행사

	보잉의 예시

	//

	이를 활용해 PA-TMM 제시

	//

	주요 기여사항

Related Work

	기존 관련 연구들 크게 3가지

	Time-Series Stock Prediction

		RNN 이용
		타임 시리즈 패턴 분석

		추가적인 통합

		문제: 상호 배타성의 기본 가정이 

	Graph-Based Stock Prediction

		주식시장을 그래프

		종목간 상호관계 포착

		문제: 정적인 상호관계를 기반
		(현실과 일치하지 않음)


	News-Based Stock Prediction

		뉴스 정보를 통합

		어텐션 네트워크로 동적인 상호관계를 구축

		문제: 특정 종목에만 뉴스가 존재하는 롱테일 효과 간과


Problem Statement

	본 모델을 분류 문제로 정의

	현실에서 특정 주가 예측 어려움

	어제와 오늘 주가 비교

	//

	3가지 input feature

	T-1 시점의 뉴스 텍스트 데이터

	T 기간 동안의 주식 거래 데이터
		고가, 저가, ....

	T-1 시점에 테크니컬 인디케이터
		Moving Average Indicators...

	
PA-TMM Architecture

	본 논문의 아키텍쳐는 2개 서브 네트워크

	Fine-tuning 전 ~ 사전학습 전략

	(큰 흐름 설명)

	먼저 Cross-Modal Fusion Module

	3개의 모달리티 통합, sentiment prompt와 hybrid embedding의 2가지 벡터를 출력

	//

	Graph Dual-Attention Module은

	sentiment prompt와 hybrid embedding를 입력으로 받아

	Attention Network를 거쳐

	Movement Predcition 출력

	// 

	학습 과정은

	Pretraning 후 Fine-tuning 순서로 진행

	//

	Pretraning 에서는 

	Sentiments Prompt를 사용하지 않고

	~~ 통과해서

	Movement Prompt 출력

	Graph Dual-Attention Module을 통해 

	학습

	//

	Fine-tuning 에서는

	Cross-Modal Fusion Module, Graph Dual-Attention Module을 거쳐

	학습


PA-TMM Architecture: Cross-Modal Fusion Module




	
	Cross-Modal Fusion Module, Graph Dual-Attention Module의 2개의 subnetwork로 구성되어 있고, 
	MPA를 통한 Pretraining과 Fine-tuning을 통해 모델을 학습시킵니다.
	
	먼저 Cross-Modal Fusion Module에 대해 설명드리면,
		특정 종목의 뉴스가 없을 경우 2개의 modality로 표현되지 않도록 news position 을 수도 뉴스로 채웁니다.
		그리고, 뉴스가 없는 종목을 Nonactivation Subset 𝑉^((0)), 그렇지 않은 종목을 Activation Subset 𝑉^((1))로 구분합니다.
		
		이후 3개의 modality를 벡터로 인코딩 합니다. 뉴스 정보는 버트를 통해 m으로 표현하고, Transaction 정보는 Bi-LSTM을 활용해 p 로 표현하고, Technical Indicator는 Tabnet 라이브러리를 활용해 q로 표현합니다.
	
		그리고 Modal Decomposition을 통해 3개의 벡터를 4개의 다른 space로 projection 합니다. 뉴스 벡터인 m은 wum과 wvm 행렬을 곱하고, Transaction vector와 Technical Indicator 벡터는 concat 한 이후 wup와 wvp 행렬을 곱한 후 비선형 활성화 함수를 통과하여 각각 news-specific 벡터인 um, news-shared 벡터인 vm, price-specific 벡터인 up, price-shared 벡터인 vp로 표현합니다. 여기서 최종 결과 벡터들의 길이는 d_r로 모두 같습니다.

		이 때, specific vector와 shared vector가 같은 정보를 담지 않도록 하기 위해 Orthogonal Loss를 사용합니다.
		Orthogonal Loss는 뉴스 벡터와 가격 벡터 각각에 대해 modal-specific 선형 변환 행렬과 modal-shared 선형 변환 행렬이 서로 직교하도록 함으로써 구조적으로 각 선형 변환 행렬을 통과한 벡터가 각각 다른 정보를 갖도록 유도합니다. wum과 wvm이 직교하고, wup와 wvp가 직교할수록 각 행렬의 곱의 크기가 최소화됨으로써 Orthogonal Loss가 작아지도록 모델을 학습합니다.
		
        그리고 Modal Integration을 통해 주가 움직임에 영향을 주는 Sentiment Prompts, Hybrid Embeddings을  산출합니다.
        먼저 Sentiment Prompts는 news-specific 벡터와 price-shared 벡터의 element-wise 곱을 통해 price가 news에 대한 noice filter 역할을 하고, 다시 concat, 선형변환, 소프트맥스를 거쳐 주가의 하락과 상승을 비율로 표현하는 길이가 2인 벡터가 만들어 집니다. 이 때, Sentiment Prompts는 뉴스 정보를 가지고 있는 activation subset인 v1에만 존재합니다.
        그리고 Hybrid Embeddings은 가격 정보와 뉴스 정보가 동일하게 중요하다는 점에 착안해 두 벡터를 더한 후, 다시 concat, 선형변환, 비선형활성화함수를 거처 길이가 d_h인 벡터를 만들어냅니다.



Bi-LSTM
	- 양방향으로 LSTM을 적용
	- LSTM은 과거 시점에서 미래 시점으로만 학습하지만, 
	- Bi-LSTM 미래 시점에서 과거 시점으로 방향으로도 학습하여 더 풍부한 표현을 담음
포아송 분포(Poisson distribution)
	- 위 시간(또는 면적, 부피 등) 내에서 사건이 평균적으로 𝜆번 발생한다고 할 때, 실제로 𝑘번 발생할 확률 분포
	- 평균과 분산이 같다는 것이 포아송 분포의 큰 특징이며, 평균 발생 횟수만 알면 분포를 구할 수 있음
	- 콜센터: 1시간에 평균 5건의 전화가 온다면, 1시간 동안 정확히 3건이 올 확률은?
Moving Average Indicators
	- 주가의 일정 기간 동안 평균을 구해 추세의 방향을 부드럽게 보여주는 지표
	- SMA (Simple Moving Average): 단순 이동 평균. 예: 5일 종가의 평균
	- EMA (Exponential Moving Average): 최근 데이터에 더 많은 가중치를 주는 이동 평균
Momentum Indicators
	- 주가의 변화 속도를 측정하여, 주가가 과매수(overbought) 혹은 과매도(oversold) 상태인지 판단하는 데 사용
	- 가격 상승/하락의 강도와 속도를 판단
	- RSI (Relative Strength Index)	: 주가의 상승폭과 하락폭을 비교해 과매수/과매도 판단 (보통 70 이상 과매수, 30 이하 과매도)
	- ROC (Rate of Change): 주가 변화 비율을 통해 추세 강도를 평가
Volatility Indicators
	- 주가의 흔들림 정도, 즉 리스크나 불확실성의 크기를 측정
	- 가격이 얼마나 급격히 오르내리는지를 판단하여 시장 불안정성을 예측
	- ATR (Average True Range): 일정 기간 동안의 주가 움직임 범위의 평균값
Volume Indicators
	- 거래량을 기반으로 매수/매도 세력의 강도, 즉 시장 참여자들의 의지를 파악
	- 가격 변동이 거래량과 함께 일어날 때 더 신뢰성이 높다고 판단
	- OBV (On Balance Volume): 거래량에 상승/하락을 가중하여 수급 흐름 파악
GNN GCN GATs 차이
	- GNN: 그래프 구조의 데이터를 처리하기 위한 딥러닝 모델의 총칭
		-  노드와 엣지로 구성된 비정형(non-Euclidean) 데이터에서 노드 간 관계와 구조를 학습
		- **이웃 노드의 정보를 집계(Aggregation)**하고 **자기 정보와 결합(Update)**하여 노드 표현을 업데이트함
	- GCN: GCN은 GNN의 대표적인 구현 중 하나로, "그래프에서의 합성곱" 연산을 정의한 모델
		- 주변 노드들의 특징을 평균처럼 취합하여 노드 임베딩을 갱신
		- 모든 이웃에게 동일한 가중치를 부여
	- GATs: 이웃 노드들에 대해 서로 다른 중요도를 학습하는 GNN
		- attention mechanism을 통해 더 중요한 노드에 더 많은 가중치를 부여
		- 중요한 이웃을 강조, 덜 중요한 이웃은 무시
Grid Search
	- Random Search가 무작위로 샘플링하는 것과 달리
	- 지정한 하이퍼파라미터 후보들의 모든 조합을 체계적으로 탐색하여 가장 성능이 좋은 조합을 찾는 방법
	- 격자(grid) 형태로 가능한 모든 조합을 나열
	- 계산 비용이 큼
Glorot(글로로) Initialization
	- Xavier Initialization와 같은 것으로 
	- 입력과 출력의 분산을 균형 있게 유지하기 위해, 초기 가중치를 특정 분포에서 랜덤하게 샘플링하는 방법
	- 정규분포 또는 균등분포를 사용
AdamW Optimizer
	- Adam에서 weight decay를 분리하여 오직 weight 크기에 비례하여 학습률에 맞춰 별도로 감쇠
	- Adam은 매 스텝마다 학습률을 자동으로 조정하면서 각 파라미터에 맞는 개별 학습률을 적용하는 최적화 기법
		- 산을 내려갈 때, **몸무게(=파라미터 값 자체)**가 계속 불어나면 균형을 잃고 미끄러지게 됩니다. 이걸 방지하기 위해 몸무게를 조절하는 기법이 바로 **Weight Decay (가중치 감쇠)**입니다.
		- 초기 학습이 빠르고 효율적
		- 수식1: g_t ← g_t + λw_t:
			- 여기서 단순히 기울기로만 업데이트하지 않고 L2 정규화를 통해 파라미터의 값이 너무 커지지 않도록 감쇠시킴(패널티 부여)
			- 하지만, 이미 adaptive learning rate을 사용하고 있어서 L2 정규화가 의도대로 작동하지 않을 수 있음
		- 수식2: w_t+1 = w_t − η⋅AdamUpdate(g_t)
	- AdamW
		- “gradient는 gradient대로 두고, 몸무게는 그냥 별도로 줄이자!”
		- 수식: w_(t+1) = w_t − η⋅(AdamUpdate(g_t))−η⋅λw_t
			- gradient update와 decay가 분리됨 → weight 자체를 직접 감소시킴
		- 정규화 효과가 더 정확하고, 과적합 방지도 더 효과적