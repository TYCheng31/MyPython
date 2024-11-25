import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 設定隨機種子以確保可重現性
_exp_name = "optimized_sample"
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# 資料增強策略
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 定義自訂數據集
class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset, self).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample:", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        try:
            im = Image.open(fname)
            if im.mode == 'RGBA':
                im = im.convert('RGB')
            im = self.transform(im)
            try:
                label = int(os.path.basename(fname).split("_")[0])
            except ValueError:
                label = -1
        except OSError:
            print(f"Skip error image: {fname}")
            return None
        return im, label

# 定義 SeparableConv2d (深度可分離卷積)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定義改進後的模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            SeparableConv2d(in_channels=3, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SeparableConv2d(in_channels=32, out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SeparableConv2d(in_channels=64, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SeparableConv2d(in_channels=128, out_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SeparableConv2d(in_channels=256, out_channels=512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 使用全局平均池化代替全連接層前的展平
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # 展平輸出
        x = self.fc_layers(x)
        return x

# 設置設備
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
    print("GPU Use")
else:
    device = torch.device("cpu")  # 使用CPU
    print("CPU Use")
model = Classifier().to(device)

# 訓練參數
num_epochs = 50
batch_size = 256
best_accuracy = 0
patience = 5  # 提前停止的耐心值

# 數據加載器
_dataset_dir = r'C:\Users\littl\Downloads\Dataset'
train_dataset = FoodDataset(os.path.join(_dataset_dir, "training"), tfm=train_tfm)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定義損失函數（Label Smoothing）
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 定義優化器與學習率調度器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=num_epochs)

# 訓練過程
early_stop_counter = 0
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader):
        images, labels = batch

        if images is None or labels is None:
            continue

        images, labels = images.to(device), labels.to(device)

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 清除前一次的梯度
        optimizer.zero_grad()

        # 反向傳播
        loss.backward()

        # 更新參數
        optimizer.step()

        # 統計損失值和正確預測的樣本數
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 更新學習率
    scheduler.step()

    # 計算平均損失和準確度
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # 模型檢查點
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(_dataset_dir, f"{_exp_name}_best.ckpt"))

    # 提前停止機制
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 加載最佳模型
model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(os.path.join(_dataset_dir, f"{_exp_name}_best.ckpt")))
model_best.eval()

# 測試集預測
test_dataset = FoodDataset(os.path.join(_dataset_dir, "testing"), tfm=test_tfm)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

prediction = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        test_pred = model_best(data)
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

# 創建測試 CSV
def pad4(i):
    return "0" * (4 - len(str(i))) + str(i)

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(test_dataset) + 1)]
df["Category"] = prediction
df.to_csv(os.path.join(_dataset_dir, "submission.csv"), index=False)

print("CSV file created successfully.")
