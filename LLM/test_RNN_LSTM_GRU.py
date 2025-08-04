import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
import os

# 데이터 전처리
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter): # 텍스트를 토큰화(tokenize)하여 한 줄씩 넘겨주는 제너레이터를 만드는 것
    for label, line in data_iter: # 예: (3, "Breaking news: stock prices are rising...")
        yield tokenizer(line) # line (텍스트 문자열)을 tokenizer로 분해해서 단어 목록을 생성

train_iter = AG_NEWS(split='train') # torchtext.datasets.AG_NEWS에서 훈련 데이터셋만 가져옴
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
# build_vocab_from_iterator(...): 전체 토큰 리스트들을 모아 단어 사전(vocab) 을 생성
# {"wall": 4, "st": 5, ".": 6, "bears": 7, ...}
# specials=["<pad>", "<unk>"]: 특별 토큰 추가, <pad>: 시퀀스를 동일 길이로 맞추기 위한 패딩 토큰, <unk>: 사전에 없는 단어(unknown)를 나타내는 토큰
vocab.set_default_index(vocab["<unk>"])
# vocab은 기본적으로 사전에 없는 단어가 들어오면 KeyError 발생
# 사전에 없는 단어에 대해 항상 <unk> 인덱스를 반환하도록 설정

def text_pipeline(x):
    return vocab(tokenizer(x))
# tokenizer(x): 단어 리스트로 분해 (예: ["stocks", "are", "rising", "fast", "."])
# vocab(...): 각 단어를 사전에서 인덱스로 변환 (예: [42, 8, 17, 65, 6])
label_pipeline = lambda x: x - 1  # 0~3 (4개 클래스)
# AG_NEWS는 1~4 범위의 정수 라벨을 가짐
# PyTorch 모델에서는 보통 0부터 시작하는 클래스 라벨을 기대하기 때문에, 1 → 0, 2 → 1, ..., 4 → 3 으로 변환

def collate_batch(batch): # 여러 개의 샘플(batch)을 하나의 배치로 정리(padding 포함)
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(torch.tensor(label_pipeline(label), dtype=torch.long))
        # label_pipeline(label): AG_NEWS 등에서는 라벨이 14 → 이를 03 으로 변환
        # torch.tensor(..., dtype=torch.long): 라벨을 LongTensor로 변환 (PyTorch 분류 모델의 요구 사항)
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.long)
        # text_pipeline(text): "Stocks are rising" → ["stocks", "are", "rising"] → [14, 23, 7]
        # torch.tensor(...): 이 정수 리스트를 텐서로 변환
        # 결과: 길이가 서로 다른 텍스트 텐서들 생성됨
        text_list.append(processed_text)
    return pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"]), torch.tensor(label_list)
            # [
            #   [14, 23, 7],
            #   [51, 9],
            #   [3]
            # ]
            # → pad_sequence →
            # [
            #   [14, 23, 7],
            #   [51, 9, 0],
            #   [3,  0, 0]
            # ]


# 모델 정의
class TextRNN(nn.Module):
    def __init__(self, rnn_type="RNN", vocab_size=20000, embed_dim=100, hidden_dim=100, output_dim=4):
        super().__init__() # 부모 클래스 nn.Module의 생성자 호출, PyTorch 모델을 정의할 때 항상 포함되어야 함
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        # Embedding 레이어는 텍스트를 정수 인덱스 → 밀집 벡터(dense vector) 로 변환
        # padding_idx: <pad> 토큰은 학습하지 않도록 마스킹
        if rnn_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")
        self.fc = nn.Linear(hidden_dim, output_dim)
        # 최종적으로 RNN/LSTM/GRU의 출력(hidden state)을 받아서 분류 결과로 변환
        # hidden_dim: 마지막 hidden state의 차원
        # output_dim: 분류 클래스 개수 (AG_NEWS는 4개 → output_dim=4)


    def forward(self, x):
        x = self.embedding(x) # 각 인덱스를 임베딩 벡터로 변환
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        # 모든 시퀀스 중 마지막 time step (seq_len-1) 의 hidden state만 추출, 
        # 즉, 각 문장의 최종 의미 요약 벡터를 가져옴 → out.shape = [batch_size, hidden_dim]
        return self.fc(out) # 최종적으로 클래스별 로짓(logits) 출력, 로짓은 일종의 점수로 softmax 전 값

# 학습 및 평가
def train_eval(model, train_loader, test_loader, epochs=1):
    device = torch.device("cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
    # "epochs 횟수만큼 반복하라", **"반복 변수는 안 쓸 거니까 이름 붙이지 마라"**는 뜻입니다.
        model.train()
        for text, label in train_loader:
            text, label = text.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
    # 역전파(backpropagation) 와 gradient 계산 비활성화
    # 평가 단계에서는 학습하지 않기 때문에 메모리와 연산 효율을 높이기 위해 사용
    # 이 블록 안에서는 모델의 파라미터가 업데이트되지 않음

        for text, label in test_loader: # 테스트 데이터를 미니배치 단위로 하나씩 불러옴
            text, label = text.to(device), label.to(device) # 데이터를 모델이 위치한 디바이스로 옮김
            preds = model(text) # 모델에 입력을 주고, 예측 결과를 얻음
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
    return correct / total

# 학습 함수 (train만 수행)
def train(model, train_loader, epochs=1):
    device = torch.device("cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for text, label in train_loader:
            text, label = text.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

# 평가 함수 (정확도만 반환)
def evaluate(model, test_loader):
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for text, label in test_loader:
            text, label = text.to(device), label.to(device)
            preds = model(text)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
    return correct / total


# 데이터 로딩
train_iter, test_iter = AG_NEWS(split=('train', 'test'))
train_dataloader = DataLoader(train_iter, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=64, shuffle=False, collate_fn=collate_batch)

continue_training = True  # ← 추가 학습 여부 설정
additional_epochs = 1     # ← 이어서 학습할 횟수

# 성능 비교
results = {}
for rnn_type in ["RNN", "LSTM", "GRU"]:
    print(f"\n=== {rnn_type} 모델 시작 ===")
    model = TextRNN(rnn_type=rnn_type, vocab_size=len(vocab))

    model_path = f"{rnn_type}_model.pt"

    start = time.time()
    
    if os.path.exists(model_path):
        print("🔁 기존 모델 로드")
        model.load_state_dict(torch.load(model_path))
        
        if continue_training:
            print("➕ 추가 학습 수행 중...")
            train(model, train_dataloader, epochs=additional_epochs)
            torch.save(model.state_dict(), model_path)
        else:
            print("📊 평가만 수행")
        
        acc = evaluate(model, test_dataloader)  # ❗ 평가만 수행
        
    else:
        print("🆕 모델 학습 시작")        
        train(model, train_dataloader, epochs=1)  # ❗ 학습r)
        torch.save(model.state_dict(), model_path)
        acc = evaluate(model, test_dataloader)    # ❗ 학습 후 평가        
        results[rnn_type] = {"accuracy": round(acc, 4), "training_time_sec": round(end - start, 2)}

    end = time.time()

    results[rnn_type] = {
        "accuracy": round(acc, 4),
        "training_time_sec": round(end - start, 2)
    }

print("\n=== 성능 비교 ===")
for model, res in results.items():
    print(f"{model}: Accuracy = {res['accuracy']}, Time = {res['training_time_sec']} sec")
