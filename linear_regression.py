import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ... 이후 시각화 코드
plt.title("순수 Python 선형회귀")

# 1. 데이터 준비
X = np.array([1, 2, 3, 4, 5])  # 공부 시간
y = np.array([30, 50, 65, 75, 85])  # 시험 점수

# 2. 평균 계산
x_mean = np.mean(X)
y_mean = np.mean(y)

# 3. 기울기(w)와 절편(b) 수식 적용
numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean)**2)
w = numerator / denominator
b = y_mean - w * x_mean

print(f"기울기 w: {w:.2f}")
print(f"절편 b: {b:.2f}")

# 4. 예측 함수 정의
def predict(x):
    return w * x + b

# 5. 예측 및 시각화
X_test = 6
y_pred = predict(X_test)
print(f"공부 {X_test}시간 → 예상 점수: {y_pred:.2f}")

# 회귀선 시각화
x_range = np.linspace(1, 6, 100)
y_range = predict(x_range)

plt.scatter(X, y, color='blue', label='실제 데이터')
plt.plot(x_range, y_range, color='red', label='회귀 직선')
plt.scatter([X_test], [y_pred], color='green', label='예측값 (6시간)')
plt.xlabel("공부 시간 (시간)")
plt.ylabel("시험 점수")
plt.title("순수 Python 선형회귀")
plt.legend()
plt.grid(True)
plt.show()