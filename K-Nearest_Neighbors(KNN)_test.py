import numpy as np
from collections import Counter

# KNN 클래스는 k개의 최근접 이웃(K-nearest neighbors)을 기준으로 예측하는 알고리즘을 구현

class KNN:
    def __init__(self, k=3):
        self.k = k
        
        # k는 하이퍼파라미터이며, 분류를 위해 얼마나 많은 이웃을 고려할지 설정

    def fit(self, X_train, y_train): # 단순히 데이터를 저장하는 기능만 합니다. KNN은 학습이라는 과정이 별도로 없습니다 (lazy learner).
        self.X_train = np.array(X_train)  # (n_train, d) 훈련 샘플들의 특징(feature) 데이터 (2차원 배열)
        print('x_train_shape: ', self.X_train.shape)
        self.y_train = np.array(y_train)  # (n_train,) 각 샘플에 대한 레이블

    def predict(self, X_test):
        X_test = np.array(X_test)  # (n_test, d)
        print('x_test_shape: ', X_test.shape)

        # 거리 행렬 계산: (n_test, n_train) -> 모든 테스트 샘플과 모든 훈련 샘플 간의 유클리드 거리 행렬을 만들기
            # n_test(3,2), n_train(6,2) -> 거리 행렬 계산(3,6,2) -> 최종 거리 행렬(3,6)
        dists = np.sqrt(
            np.sum((X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2) # 브로드캐스팅
        )
            # 요약: 브로드캐스팅(차원 확장) -> 유클리드 거리 계산(원소간 뺄셈 -> 제곱 -> 합산 -> 루트)
        
        
        # 브로드캐스팅이란? 다양한 형상 또는 랭크의 배열끼리 산술 연산을 할 때, 서로 호환될 수 있도록 형상을 맞춰주는 과정
        # 왜 브로드캐스팅을 하는가? 모든 테스트 샘플과 훈련 샘플 간의 거리를 한 번에 계산하기 위해서
            # 가능한 이유 
                # X_test.shape = (n_test, d)
                # X_train.shape = (n_train, d)
                # → 이를 각각 (n_test, 1, d)와 (1, n_train, d)로 바꾸면
                # → 두 배열은 **(n_test, n_train, d)**로 자동 확장됨
                # → 즉, 모든 x_test와 x_train 쌍에 대해 차를 계산할 수 있음.
        # X_test[:, np.newaxis, :] => 브로드캐스팅을 위해 차원을 하나 추가하는 것(각 샘플을 독립적으로 다루기 위해)
            # (3, 2) -> (3, 1, 2)
            # np.newaxis = None 
        # 브로드캐스팅 규칙
            # 두 배열의 뒤에서부터 차원끼리 비교
            # 같거나, 한쪽이 1이면 → 브로드캐스트 가능
            # 두 배열을 비교: X_test_expanded (3, 1, 2), X_train_expanded (1, 6, 2)
            # 브로드캐스팅 가능하여 실행: NumPy는 내부적으로 아래처럼 자동 확장해서 연산을 수행
                # (3, 1, 2) → (3, 6, 2), (1, 6, 2) → (3, 6, 2)
            
        
        # 브로드캐스팅을 하지 않으면? 이중 루프 필요
            # for x_test in X_test:  # [2, 3]
            #    for x_train in X_train: # 각각 [1,2], [3,4], [5,6]
            #        dist = np.linalg.norm(x_test - x_train)
                                                                # dist = sqrt((2-1)^2 + (3-2)^2) = sqrt(1 + 1) = 1.41
                                                                # dist = sqrt((2-3)^2 + (3-4)^2) = sqrt(1 + 1) = 1.41
                                                                # dist = sqrt((2-5)^2 + (3-6)^2) = sqrt(9 + 9) = 4.24
                                                                # x_test = [2,3] 에 대한 거리 리스트 = [1.41, 1.41, 4.24]
            
                # 각 테스트 샘플 x_test와 훈련 샘플 x_train 간의 거리를 하나하나 계산하는 과정
                # 모든 테스트 샘플에 대해, 모든 훈련 샘플과의 거리(유클리드 거리) 를 계산
                # 먼저 "누가 가까운지" 알기 위해
                # 유클리드 거리 = sqrt((y1-x1)^2 + (y2-x1)^2+ ... + (yd-xd)^2) (벡터 차이를 제곱해서 모두 더하고, 제곱근을 씌움)
        
        print('dist.shape: ', dists.shape)
        print('dist: ', dists)

        # 가장 가까운 k개 인덱스 선택: (n_test, k)
        knn_indices = np.argsort(dists, axis=1)
        print('knn_indices1: ', knn_indices)
        knn_indices = np.argsort(dists, axis=1)[:, :self.k] # 행은 모두 추출, 열은 k번째 까지만 추출
        print('knn_indices2: ', knn_indices)
            # np.argsort: 가까운 순으로 정렬
            # axis=1이므로, 축이 같은 것들끼리 연산하면 테스트 샘플별 연산이 됨
            # 각 테스트 샘플에 대해 거리 순으로 정렬된 인덱스 반환
            # 가장 가까운 k개의 인덱스만 자릅니다 → (n_test, k)
                # NumPy의 슬라이싱(Slicing) 문법
                    # arr[start:stop:step]
                        # start: 시작 인덱스 (포함)
                        # stop: 끝 인덱스 (불포함)
                        # step: 간격 (기본값 1)
                            # a = np.array([0, 1, 2, 3, 4, 5])
                            # print(a[1:4])      # [1 2 3]
                            # print(a[:3])       # [0 1 2]
                            # print(a[::2])      # [0 2 4]
                            # print(a[::-1])     # [5 4 3 2 1 0] (역순)
                    # 다차원 슬라이싱 문법
                        # arr[axis0_slice, axis1_slice, axis2_slice, ...]
                        # 각 축마다 start:stop:step 형식으로 슬라이싱
                        # 콤마(,)로 축을 구분
                        # :는 해당 축 전체를 의미
                        # 음수 인덱스는 뒤에서부터: a[-1] → 마지막 요소
        

        # 인덱스를 이용해 라벨 추출, 각 x에 해당하는 y값 추출
        knn_labels = self.y_train[knn_indices]  # (n_test, k)
        print("knn_lables: ", knn_labels)
        
            # X_train: 훈련용 입력값 (예: 사람 키와 몸무게)
            # y_train: 각 입력값에 대한 정답 레이블 (예: "남성", "여성" 혹은 클래스 번호 0, 1)
            # X_test: 예측하고 싶은 새 입력값(어느 클래스인지 모름 → 예측이 필요)

        # 각 row에서 다수결 투표
        predictions = np.array([
            Counter(row).most_common(1)[0][0] # Counter().most_common(1): 각 row에서 가장 빈도가 높은 레이블(y값) 선택
            for row in knn_labels
            
            # 1단계: Counter(row)
                # 결과: Counter({1: 2, 0: 1})
            # 2단계: .most_common(1)
                # most_common(n)은 등장 빈도순으로 상위 n개의 (값, 개수) 튜플을 반환
                # 결과: [(1, 2)]
            # 3단계: [0]
                # 리스트에서 첫 번째 요소인 튜플 하나를 꺼냄
                # 결과:[(1, 2)][0] → (1, 2)
            # 4단계: 또 [0]
                # 튜플에서 0번째 요소 = 가장 많이 등장한 레이블 1
                # 결과: (1, 2)[0] → 1            
            
        ])
        # 가장 가까운 k개의 샘플의 **정답(y)**을 가져옵니다 → 이게 바로 y_train
        # 그 중 가장 많이 등장한 레이블을 예측값으로 사용합니다
        
        print('predictions: ', predictions)

        return predictions

# 예제

X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]] # (6,2)
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[5, 5], [2, 2], [7, 6]] # (3,2)

model = KNN(k=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Predictions:", pred)  # 예: [1 0 1]