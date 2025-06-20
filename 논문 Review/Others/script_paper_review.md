Intro

	안녕하세요.

	2025학년도 2학기 석사과정으로 MLCF 연구실에 지원한 방태성입니다.

	저는 정형 데이터와 비정형 데이터를 융합하여, 장단기 주가 흐름을 예측하는 연구에 관심을 가지고 있습니다.

	이번 랩테스트에서는 제 연구 관심과 밀접하게 연관된 논문인
	**"Responding to News Sensitively in Stock Attention Networks via Prompt-Adaptive Trimodal Model"**을 리뷰 주제로 선정하였습니다.

	이 논문은 2025년 6월, IEEE Transactions on Neural Networks and Learning Systems에 게재되었습니다.


Outline
	발표 목차는 논문의 흐름과 동일하게 ~~~~ 의 순으로
	
Introduction-1

	초반부는 백그라운드의 변화
	그에 따라 등장하는 새로운 챌린지를 

	멀티미디어 플랫폼의 빠른 증가
	주가 예측을 하는데 결정적 요소
	
	특히, 뉴스와 관련

	연구자들의 관심 증가

	//

	기존 주가 예측 여러 모델, 한계 존재

	각각의 단점

	//

	여기서 중요한 2가지 챌린지 언급

	일부 주식만 뉴스 정보를 갖는 현실을 반영

	브레이킹 뉴스가 주목을 받지 못하고,
	가격 정보에 attention이 편향되는 문제

	뉴스 정보 자체의 부족으로 일반화 성능이 떨어지는 문제


Introduction-2

	논문에서 제시한 Figure 1
	
	모든 종목이 갖는 정보

	일부 종목이 갖는 정보

	전체적 영향을 주는 뉴스임에도

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

	기존 연구들 크게 3가지

	Time-Series Stock Prediction

		RNN 이용
		타임 시리즈 패턴 분석

		추가적인 통합

		문제: 상호 배타성의 기본 가정이 

		(market factors [42]: moving average convergence/divergence (MACD), price-tobook ratio (P/B), relative strength index (RSI))
		(investment behaviors [43]: 사람들의 집합적 투자 행동에서 추출된 주식 속성이 주가 예측 과제에서 높은 효과성을 갖고 있음)
		

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


(상세 설명)

	Cross-Modal Fusion Module

		3대의 모달리티 중 2개만 존재하는 경우

		수도뉴스로 채움

		2개의 서브셋으로 분리

	Representation Learning

		(그림 먼저)
	
		3개의 모달리티 정보를

		벡터로 임베팅

		각각 버트, Bi-LSTM, TabNet

		각 벡터의 크기는 ~

	Modal Decomposition

		(그림 먼저)

		3개의 모달리티 벡터

		디컴포즈하여

		4개의 다른 벡터를 출력

		//

		decompose 과정 설명	(컨캣과 행렬곱 등)

		서로 다른 4공간에 프로젝션

		출력 결과 설명(모두 dr 크기를 갖는 벡터)

		specific 정보와 shared 정보를 분리

		//

		Orthogonal Loss 설명

		modal-specific 공간과 modal-shared 공간을 서로 직교하는 것이 목표

		곱한 결과의 Frobenius Norm을 최소화


	Modal Integration
	
		(그림 먼저)

		디컴포지션 결과를 인터그래이션

		센티, 하이 결과 출력

		//

		Sentiment Prompts
		
		뭐가 중심인지

		element-wise 곱

		price: noise filter

		과정/과정

		길이가 2인 벡터

		감소, 증가 확률

		이것 자체로 예측 결과

		//

		Hybrid Embeddings

		뭐가 중심인지

		Equally crucial -> Addition Operation

		선형 변환 후 비선형 활성화함수

		길이가 dh인 벡터

	Stock Polarized Activation
	
		(그림 먼저)

		입력: Sentiment Prompts, Hybrid Embeddings

		노트 벡터를 임베딩

		//

		영향력이 큰 뉴스 정보

		일부 종목에만 (비대칭성)

		분리하여 임베딩

		함수 설명

		//

		Activated 노드에 대해 

		두 노드를 유사할수록 가깝게, 다를수록 더 멀게

		부호함수와 cos distance 설명

		// 

	Interaction Inference
	
		노드들의 관계성을 추론

			어텐션, 바이파알타잇

		Fig 3
			파셜리 바이파알타잇

			항상 타겟은 Nonactivated

			activated는 주기만 함

		//

		ni가 타겟, nj가 소스
		
		노드간 Message Flux 비중

		//

		컨캣, 선형변환, 리키렐루, 벡터 곱 => 스칼라값


	Information Exchange
	
		어텐션스코어로 엣지와 메세지벡터를 구현

		두 노드 컨캣.... => 엣지 구현

		//
		
		엣지에 어텐션 스코어 곲한 가중합

		activated node의 메세지 벡터와 nonactivated node의 메세지 벡터는 합치지 않고 컨캣

		노드간 상호작용을 summary


	Output Mapping

		최종 결과로 Movement를 Prediction

		activated stock 

		뉴스 정보가 지배적

		nonactivated stock

		노드 벡터에 메세지 벡터 컨캣


	Discussion

		activated stock, nonactivated stock을 분리한 partially bipartite graph

		nonactivated stock가 전달받는 Message vectors 중요한 역할 

			ablation experiments 에서 한번 더 증명 

		activated and nonactivated nodes 분리하는 코스트 높지만


	Computational Complexity
	
		Cross-Modal Fusion Module
		
		Bi-LSTM 단계에서 

		나머지는

		Graph Dual-Attention Module
		
		Interactions Inference 단계에서

		최종 시간 복잡도는


Model Optimization

	(큰 그림) 다음 슬라이드 이미지 활용

		(파란선) Movement Prompt Adaptation 통한 pretraning 설명
		(붉은선) Fine-tuning

	Movement Prompt Adaptation: Equivalence Resampling
	
		Equivalence Resampling로 Data Augmentation 

		실제 주가 움직임으로 prompts 생성

		long tail effect 해결, 일반화 성능 개선

		//

		포아송 분포 활용

		엡실론(균등 분포)

		Mutation Probability 로 인버팅

		//

		하루에 50회 반복

	Pretraining Objectives
	
		𝐿_𝑚𝑜𝑣^((0))+𝛽𝐿_𝑜𝑟𝑡+𝛾𝐿_𝑝𝑜𝑙 3개의 손실함수로 구성

		𝐿_𝑚𝑜𝑣(0)은 nonactivated stock 에 대한 손실함수
		
		binary cross entropy를 활용

		정답을 맞출 확률이 클수록 손실함수가 작아지도록 구현

		3 손실함수의 가중합을 최소화


	Fine-Tuning Objectives
	
		𝐿_𝑚𝑜𝑣^((1))+𝐿_𝑚𝑜𝑣^((0))+𝛽𝐿_𝑜𝑟𝑡+𝛾𝐿_𝑝𝑜𝑙

		activated stock 에 대한 손실함수
		
		binary cross entropy를 활용
		
		4개의 손실함수의 가중합을 최소화

Experiments

	Evaluation Setup
	
		Datasets
		
			순서대로

		Compared Baselines
		
			순서대로 읽으면서

		Evaluation Metrics
		
			4가지~

			2개 먼저

			백테스팅

			나머지 2개

		Implementation Details
		
			오른쪽 표 참고

		Trading Portfolios
			
			백테스팅 probablilty 평가를 위해
			
			20개로 구성

	Stock Movement Prediction

		다른 모델들과 실험 결과 비교
	
		ACC와 MCC 비교에서 수치상 차이가 크지 않음

		DM 검정을 통해 유의 수준 확인

		나스닥 최소 10%,  SNP는 최소 5% 수준에서 유의미한 차이를 분석

	Analysis

		실험 결과 분석

		순서대로 읽으면서

		결국 PA-TMM 이 짱

	Ablation Study

		일부 구성을 제외하고 실험 진행
	
		모델 아키텍쳐의 효과성

			표 설명

			글 설명	

		MPA의 효과성

			표 설명

			글 설명

	Backtesting Profitability

		실제 투자 모의 실험

		표 설명

		글 설명

	Stress Test During Market Crash
	
		마켓 크러쉬

		그래프 설명

	Parameter Sensitivity Analysis
	
		윈도우 사이즈, Mutation Probability, 노드와 엣지의 디멘션

		그래프와 함께 설명

	Case Study on Exploring Stock Attention Networks
	
		사례 설명

		그림과 함께 설명

Conclusion


이 논문을 통해 저는, 뉴스나 사회적 인식과 같은 정성적 정보가 실제로 주가에 영향을 미치고, 이를 학습 가능한 형태로 정제하고 활용할 수 있다는 점에 깊은 인사이트를 얻었습니다.

제 연구 역시 기업의 내재 가치를 정량적 지표뿐 아니라 정성적 요인—예를 들어 뉴스, 임직원 리뷰, 산업 전망 등—을 반영한 중장기 주가 예측 모델을 목표로 하고 있습니다.

향후에는 이 논문에서 제시한 프롬프트 기반 학습 전략이나 attention 구조를 제 가치 기반 예측 모델에 접목시켜, 저평가된 우량 종목을 조기에 식별하고 장기 수익 가능성을 정량화하는 프레임워크로 발전시키고자 합니다.

단순한 예측 정확도를 넘어서, 실제 투자 의사결정에 신뢰성 있는 분석을 제공하는 모델을 구축하는 것이 제 연구의 최종 목표입니다. 감사합니다.












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


# MSMF: Multi-Scale Multi-Modal Fusion for Enhanced Stock Market Prediction

멀티모달 데이터 분석 분야에서 본 연구는 의미 있는 성과를 달성하였습니다. 기존의 주식 시장 예측 방법과 비교했을 때, 제안한 모델은 서로 다른 모달리티(데이터 유형)를 효과적으로 통합하여 만족스러운 결과를 도출해냈습니다. 또한 본 모델은 멀티모달 학습에서 발생하는 핵심적인 문제들을 해결하기 위해 여러 모듈을 설계하여 제안하였습니다.

우선, 데이터 샘플링 주기의 불일치 문제를 해결하기 위해 **모달리티 보완 모듈(Modality Completion module)**을 설계했습니다. 이 모듈은 DBN(Deep Belief Network) 기반의 네트워크를 활용하여 알려진 모달 데이터의 분포에 기반해 누락된 데이터를 정확하게 보완함으로써 모달리티 간 이질성을 효과적으로 처리합니다.

다음으로, 전역(Global) 정보와 국소(Local) 정보의 추출 간 충돌 문제를 해결하기 위해 **다중 스케일 인코더(Multi-scale Encoder)**를 개발했습니다. 이 인코더는 다양한 스케일의 특징을 추출할 수 있어, 모델이 전역적 흐름과 세부적 패턴을 동시에 포착하고 다양한 모달 데이터를 더욱 잘 통합할 수 있게 합니다.

추가로, 모달 간 정보 교환의 어려움을 해결하기 위해 **다중 스케일 정렬 모듈(Multi-scale Alignment module)**을 도입했습니다. 이 모듈은 서로 다른 형태와 크기의 모달 특징들을 동일한 크기로 변환하여, 자연스러운 상호작용과 정보 통합을 가능하게 하여 멀티모달 분석의 효과를 높입니다.

또한, 중복되거나 충돌되는 정보를 제거하기 위해 블랭크 러닝(Blank Learning) 개념을 제안했습니다. 특징에 대해 softmax 연산을 적용하고, 확률값을 기준으로 상위 특징을 선별함으로써 중요하지 않거나 모순되는 정보를 제거하고 모델의 성능을 향상시킵니다.

더불어, **다중 세분화 게이트(Multi-Granularity Gates)**를 도입하여, 서로 다른 과제가 동일한 모달 내에서도 국소 정보와 전역 정보의 비율을 다르게 설정할 수 있도록 하였습니다. 아울러, 외부 관점에서 각 과제는 동일한 모달의 표현에 대해 서로 다른 가중치를 부여할 수 있도록 설계하였습니다.

마지막으로, **과제 중심 예측 계층(Task-oriented Prediction Layer)**을 설계하여, 특징 융합 과정 중 전역 및 국소 정보의 손실을 방지하고, 정보의 열화(degradation)를 막으며 모델의 수렴 속도를 높일 수 있도록 하였습니다.

이러한 각 모듈과 혁신적인 아이디어의 설계를 통해, 본 연구는 멀티모달 데이터 분석 분야에서 중요한 진전을 이루었습니다. 제안된 각 구성요소는 멀티모달 학습의 주요 과제를 해결하며, 주식 시장 예측과 같은 실질적인 문제에 효과적인 솔루션을 제공합니다. 우리는 본 연구의 성과가 향후 멀티모달 데이터 분석 분야의 연구 및 응용에 긍정적인 영향을 미칠 것이라 확신합니다.

# Stock Movement Prediction Based on Bi-typed Hybrid-relational Market Knowledge Graph via Dual Attention Networks

6. 결론 및 향후 연구 방향
본 논문에서는 주가 이동 예측(Stock Movement Prediction, SMP) 문제를 다루었습니다. 실제 금융 시장에서의 **모멘텀 확산 효과(momentum spillover effect)**를 모델링하기 위해, 우리는 이중 타입(Bi-typed) 및 **혼합 관계(Hybrid-relational)**를 갖춘 **새로운 시장 지식 그래프(Market Knowledge Graph, MKG)**를 구축하였습니다. 이후, 이 MKG에서 모멘텀 확산 특성을 학습할 수 있도록, **상호 클래스 어텐션(inter-class attention)**과 동일 클래스 어텐션(intra-class attention) 모듈을 모두 갖춘 **새로운 이중 어텐션 네트워크(DANSMP)**를 제안하였습니다.

우리 방법의 성능을 평가하기 위해, CSI100E와 CSI300E라는 두 개의 새로운 데이터셋을 구축하였으며, 이 데이터셋에 대한 실험 결과는, 제안한 DANSMP 모델이 이중 타입의 혼합 관계 MKG를 활용하여 주가 예측 성능을 효과적으로 향상시킬 수 있음을 입증하였습니다. 또한, Ablation Study를 통해 임원 정보의 활용과 기업 간 암묵적 관계의 추가가 성능 향상에 주요한 기여를 한다는 점을 다시 확인할 수 있었습니다.

향후 연구로는, 임원에 관한 웹 미디어 정보를 탐색하는 방향이 흥미로운 과제가 될 수 있습니다. 예를 들어:

범죄 혐의, 건강 문제 등 뉴스에서 다루는 부정적인 사실 정보,

Twitter, Weibo 등 소셜미디어에서의 부적절한 발언

등이 있을 수 있습니다. 우리는 이러한 임원 관련 사건 정보가 그래프 기반 방법에 통합될 경우, 주가 이동 예측 성능을 더욱 향상시키는 데 도움이 될 것이라고 믿습니다