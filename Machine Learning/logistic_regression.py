import numpy as np
import matplotlib.pyplot as plt

## 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

## 예시 데이터
# x: 시험 점수, y: 합격 여부 (1은 합격, 0은 불합격)
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])  # (9, 1)
y = np.array([[0], [0], [0], [0], [1], [1], [1], [1], [1]])  # (9, 1)

# 절편을 위한 1 추가 (편향 term)
X = np.hstack((np.ones((x.shape[0], 1)), x))  # (9, 2)

## 비용 함수 (Binary Cross Entropy Loss)
def compute_loss(y, y_hat):
    m = len(y)
    return - (1/m) * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

## 학습 (경사하강법)
def logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))  # 초기 가중치 (2x1)

    for i in range(epochs):
        z = X @ weights
        y_hat = sigmoid(z)
        loss = compute_loss(y, y_hat)
        
        # 경사 계산 및 업데이트
        grad = (1/m) * (X.T @ (y_hat - y))
        weights -= lr * grad

        if i % 100 == 0:
            print(f"epoch {i}, loss: {loss:.4f}")
    
    return weights

## 학습 실행
weights = logistic_regression(X, y, lr=0.1, epochs=1000)

## 예측 함수
def predict(x, weights):
    x = np.hstack((np.ones((x.shape[0], 1)), x))  # 절편 추가
    return sigmoid(x @ weights)

## 시각화
x_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = predict(x_test, weights)

plt.scatter(x, y, label='Training Data', color='red')
plt.plot(x_test, y_pred, label='Logistic Curve')
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel("시험 점수")
plt.ylabel("합격 확률")
plt.title("로지스틱 회귀 수식 기반 구현")
plt.legend()
plt.grid(True)
plt.show()
