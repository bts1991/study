import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.01
epochs = 5

# 2. 데이터셋 로딩 (MNIST)
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 선형 모델 정의
# MNIST 이미지: 28x28 픽셀 → 784 차원 input
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)  # 입력: 784, 출력: 숫자 0~9 (10 클래스)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 이미지(28x28)를 1차원 벡터로 변환
        return self.linear(x)

model = LinearNet()

# 4. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 5. 학습 루프
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# 6. 테스트 정확도 측정
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
