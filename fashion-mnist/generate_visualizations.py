"""
Fashion MNIST CNN 시각화 이미지 생성 스크립트

생성되는 이미지:
1. 샘플 이미지 (10개 카테고리별)
2. 학습 곡선 (accuracy & loss)
3. 혼동 행렬 (Confusion Matrix)

모든 이미지는 images/ 폴더에 저장됩니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import os

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 이미지 저장 폴더 생성
os.makedirs('images', exist_ok=True)

# ============================================================
# 1. 데이터 로딩
# ============================================================
print("📂 데이터 로딩 중...")

train_images_path = 'data/train-images-idx3-ubyte'
train_labels_path = 'data/train-labels-idx1-ubyte'
test_images_path = 'data/t10k-images-idx3-ubyte'
test_labels_path = 'data/t10k-labels-idx1-ubyte'

def load_mnist_images(path: str):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(path: str):
    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

print(f"✅ 데이터 로딩 완료: 훈련 {len(train_images)}개, 테스트 {len(test_images)}개")

# ============================================================
# 2. 데이터 전처리
# ============================================================
print("🔧 데이터 전처리 중...")

train_images_normalized = train_images.astype(np.float32) / 255.0
test_images_normalized = test_images.astype(np.float32) / 255.0

train_images_normalized = train_images_normalized[:, np.newaxis, :, :]
test_images_normalized = test_images_normalized[:, np.newaxis, :, :]

print("✅ 전처리 완료")

# ============================================================
# 3. CNN 모델 정의
# ============================================================
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================
# 4. 모델 학습
# ============================================================
print("🚀 모델 학습 시작...")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"   장치: {device}")

model = FashionMNISTCNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008)

# 데이터로더 생성
train_images_tensor = torch.from_numpy(train_images_normalized)
train_labels_tensor = torch.from_numpy(train_labels).long()
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 학습 history 저장
history = {
    'train_loss': [],
    'train_acc': []
}

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    history['train_loss'].append(avg_loss)
    history['train_acc'].append(accuracy)

    print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("✅ 학습 완료!")

# ============================================================
# 5. 모델 평가
# ============================================================
print("📊 모델 평가 중...")

test_images_tensor = torch.from_numpy(test_images_normalized)
test_labels_tensor = torch.from_numpy(test_labels).long()
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
correct_predictions = 0
total_samples = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct_predictions / total_samples
print(f"✅ 테스트 정확도: {accuracy:.2f}%")

# ============================================================
# 6. 시각화 1: 샘플 이미지 (10개 카테고리)
# ============================================================
print("🎨 시각화 1: 샘플 이미지 생성 중...")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names_kr = ['티셔츠/탑', '바지', '풀오버', '드레스', '코트',
                  '샌들', '셔츠', '스니커즈', '가방', '앵클부츠']

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Fashion MNIST 샘플 이미지 (카테고리별)', fontsize=16, fontweight='bold')

for i in range(10):
    # 각 클래스의 첫 번째 이미지 찾기
    idx = np.where(train_labels == i)[0][0]
    ax = axes[i // 5, i % 5]
    ax.imshow(train_images[idx], cmap='gray')
    ax.set_title(f'{i}: {class_names_kr[i]}', fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig('images/샘플이미지.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 저장: images/샘플이미지.png")

# ============================================================
# 7. 시각화 2: 학습 곡선
# ============================================================
print("🎨 시각화 2: 학습 곡선 생성 중...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss 그래프
ax1.plot(range(1, num_epochs + 1), history['train_loss'], 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('학습 손실(Loss) 변화', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy 그래프
ax2.plot(range(1, num_epochs + 1), history['train_acc'], 'g-', linewidth=2, label='Training Accuracy')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('학습 정확도(Accuracy) 변화', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/학습곡선.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 저장: images/학습곡선.png")

# ============================================================
# 8. 시각화 3: 혼동 행렬
# ============================================================
print("🎨 시각화 3: 혼동 행렬 생성 중...")

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_kr,
            yticklabels=class_names_kr,
            cbar_kws={'label': '예측 개수'})
plt.xlabel('예측 라벨', fontsize=12)
plt.ylabel('실제 라벨', fontsize=12)
plt.title(f'혼동 행렬 (Confusion Matrix)\n테스트 정확도: {accuracy:.2f}%',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('images/혼동행렬.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 저장: images/혼동행렬.png")

# ============================================================
# 9. 추가 시각화: 예측 샘플
# ============================================================
print("🎨 추가 시각화: 예측 샘플 생성 중...")

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('모델 예측 결과 샘플 (정답 vs 예측)', fontsize=16, fontweight='bold')

# 테스트 세트에서 랜덤하게 15개 선택
np.random.seed(42)
sample_indices = np.random.choice(len(test_images), 15, replace=False)

for i, idx in enumerate(sample_indices):
    ax = axes[i // 5, i % 5]
    ax.imshow(test_images[idx], cmap='gray')

    true_label = test_labels[idx]
    pred_label = all_predictions[idx]

    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'실제: {class_names_kr[true_label]}\n예측: {class_names_kr[pred_label]}',
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('images/예측샘플.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 저장: images/예측샘플.png")

# ============================================================
# 10. 완료
# ============================================================
print("\n" + "="*60)
print("🎉 모든 시각화 이미지 생성 완료!")
print("="*60)
print("\n생성된 파일:")
print("  1. images/샘플이미지.png - 10개 카테고리별 샘플")
print("  2. images/학습곡선.png - Loss & Accuracy 변화")
print("  3. images/혼동행렬.png - Confusion Matrix")
print("  4. images/예측샘플.png - 모델 예측 결과 샘플")
print(f"\n최종 테스트 정확도: {accuracy:.2f}%")
print("\n다음 단계: README.md 작성")
