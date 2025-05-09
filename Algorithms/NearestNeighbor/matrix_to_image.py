import numpy as np
import matplotlib.pyplot as plt

# 데이터
X_test = np.array([[5, 5], [2, 2], [7, 6]])  # (3, 2)
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])  # (6, 2)

# 3차원 확장 (형태 확인용)
X_test_expand = X_test[:, np.newaxis, :]    # (3, 1, 2)
X_train_expand = X_train[np.newaxis, :, :]  # (1, 6, 2)

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X_test 점: z=0
ax.scatter(X_test[:, 0], X_test[:, 1], zs=0, zdir='z', c='red', s=100, label='X_test')

# X_train 점: z=1
ax.scatter(X_train[:, 0], X_train[:, 1], zs=1, zdir='z', c='blue', s=100, label='X_train')

# 연결선 (거리 계산 직관적으로 보기)
for i in range(len(X_test)):
    for j in range(len(X_train)):
        x_vals = [X_test[i, 0], X_train[j, 0]]
        y_vals = [X_test[i, 1], X_train[j, 1]]
        z_vals = [0, 1]
        ax.plot(x_vals, y_vals, z_vals, c='gray', linestyle='dotted', linewidth=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Group (Test=0, Train=1)')
ax.legend()
ax.set_title("3D Visualization of X_test and X_train")

plt.show()
