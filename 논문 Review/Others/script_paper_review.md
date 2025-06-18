First

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
	발표 목차는 논문의 흐름과 동일하게 ~~~~ 의 순으로 진행하겠습니다.
	
Introduction
	도입부는 Multimedia Platforms의 빠른 증가에 대한 얘기로 시작됩니다. Financial news & social media 에서 제공되는 정보가 중요한 투자 신호의 역할을 하기 시작했고, 여전히 randomness가 존재하는 주식 시장에서 주가 흐름을 예측하는데 deterministic components로서 주목을 받고 있다고 얘기합니다.
	
	하지만, 기존의 모델들이 Financial news를 충분히 활용하고 있지 못한다고 지적합니다.
		Time-series Forecasting Model들은 모든 종목들이 서로 독릭접이라는 가정아래 Momentum spillover와 같은 종목 간의 상호작용을 고려하지 못했고,
		Graph Neural Network 기반 모델들을 Hard-coded microstructure로 인해 종목간 관계성을 충분히 포착하지 못했고,
		Graph Attention Networs는 종목간 관계를 고려하는 부분은 개선했지만, 다량의 가격 정보 속에서 뉴스 정보에 attention이 Biased되었다고 합니다.
	
	여기서 두가지 중요한 Challenges를 언급합니다.
		실제 현실에서는 일부 종목만 주식 정보를 갖고 있다는 전제 하에서 
		
		먼저, Feature Distribution에 나타나는 Long Tail Effect 로 인해 뉴스 정보에 덜 주목하게 된다는 것과
		
		데이터 자체가 부족하여 일반화에 문제가 있다는 점입니다.

	논문에서 제시한 Figure 1에서는
		모든 종목들이 Time-series features and Technical indicators 가지고 있는데 반해, JPM, WFC만 뉴스 정보를 가지고 있는 Long Tail Effect가 나타나고 있고, 
	
		아래 그림을 보면 JPM과 WFC에 대한 뉴스가 시장 전반에 영향을 줌에도 불구하고, MS나 Google, Apple은 주목하고 있지 않고 있음을 보여줍니다.
		
	Introduction의 다음 부분에서는 이에 대한 터닝 포인트로 Financial News 의 특이한 본성에 대해 얘기합니다.
		특정 종목에 대한 이벤트가 뉴스로 전달되면 시장 전반에 즉각적으로 지배적인 영향을 미친다는 것입니다. 그에 대한 예시로 2024년 1월 미국의 Federal Aviation Administration이 보잉 737 모델의 170기 이상을 비행할 수 없도록 명령했고, 이로 인해 보잉의 주가가 8% 하락했으며, 연이어 보잉의 주요 경쟁자인 에어버스의 주가는 소폭 상승했습니다.
		
	하지만 이러한 특성이 기존에는 반영되지 못했음을 지적하며, 기존의 문제들을 해결하기 위한 rompt-Adaptive Trimodal Model (PA-TMM)을 제안합니다.
		이 모델은 Cross-Modal Fusion Module, Graph Dual-Attention Module, Movement Prompt Adaptation 그리고 Pretraining, Fine-tuning으로 구성되어 있으며, 

		각각을 통해 Sentiments Prompts, Stock Attention Network, Movement Prompt를 구현하고 Pretraining, Fine-tuning을 사용해 financial news에 더 민감하게 반응할 수 있게 됩니다
	
	본 논문의 기여는 다음과 같은 아이디어를 제시합니다. 
		먼저, New Resampling Strategy를 통해 data를 augmentation 하여 long-tailed feature distribution를 해결하는 것입니다. 
		다음으로 financial news까지 고려한 Trimodal 방식을 적용하는 것입니다. 
		마지막으로, 종목간 dynamic interaction을 반영하기 위해 고정된 네트워크가 아닌 attention을 적용한 네트워크를 구현하는 것입니다.

Related Work
	관련 연구들 중 Time-Series Stock Prediction은 RNN을 활용하여 time series pattern을 분석합니다. 이후 market factors [42], investment behaviors [43], technical indicators 를 통합하는 연구가 진행되었지만, 종목간의 관계가 상호 배타적이라는 가정을 벗어나지 못했습니다.
	
	실제 기업들은 하나의 시장 경제체제에서 서로 연결되어 있다는 개념을 반영한 Graph-Based Stock Prediction 에서는 Graph Neural Networks를 이용해 종목 간의 관계성을 고려했습니다. 하지만 노드간 관계의 강도를 고려하지 않는 static network를 기반으로 했다는 한계가 있습니다.
	
	이후, News라는 Modality를 반영하기 시작하면서, Graph Attention Networks 를 같이 적용해 종목 간의 관계성도 개선했지만, 실제 현실에서 뉴스와 관련된 주식의 수가 매우 적은 long-tail effect를 해결하지는 못했습니다.
		
Problem Statement
	본 연구는 문제를 해결하기 위한 예측 방법을 회귀가 아닌 분류로 접근합니다. 하루 전 날짜의 주가와 비교하여 오늘이 주가가 올랐는지 내렸는지를 분류하는 방식입니다.
	
	다음으로, Input feature 로 3가지의 modality를 사용합니다. 
	Textual News, Historical transaction features, Technical Indicators 입니다.
	
	각각 뉴스 텍스트 데이터, 종가/시가/거래량 등의 주식 거래 데이터, 기술 분석을 통해 계산된 이동평균지표와 모멘텀지표 등이 있습니다.
		
PA-TMM Architecture
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



 