import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import PIL.Image as Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from segmentation_models_pytorch import UnetPlusPlus

# 自定義資料集類別
class TrafficDataset(Dataset):
    def __init__(self, root1, has_labels=True, transform=None):
        self.imgs = self.make_dataset(root1, has_labels)
        self.has_labels = has_labels
        self.transform = transform

    def make_dataset(self, root1, has_labels=True):
        imgs = []
        if has_labels:
            for filename in os.listdir(root1):
                img = os.path.join(root1, filename)
                mask = os.path.join(root1.replace('trainingimages', 'traininglabels'), filename.replace('jpg', 'png'))
                imgs.append((img, mask))
        else:
            for filename in os.listdir(root1):
                img = os.path.join(root1, filename)
                imgs.append(img)
        return imgs

    def __getitem__(self, index):
        if self.has_labels:
            x_path, y_path = self.imgs[index]
            img = Image.open(x_path)
            mask = Image.open(y_path)
            img = np.array(img)
            mask = np.array(mask)
            mask = mask[:, :, 2]  # 提取第三個通道
            mask = np.where(mask >= 1, 1, 0).astype(float)
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            return img, mask
        else:
            x_path = self.imgs[index]
            img = Image.open(x_path)
            img = np.array(img)
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            return img

    def __len__(self):
        return len(self.imgs)

# Focal Tversky Loss 定義
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky

bce_loss = nn.BCEWithLogitsLoss()
focal_tversky_loss = FocalTverskyLoss()

# 損失函數結合
def criterion(outputs, targets):
    bce = bce_loss(outputs, targets)
    focal_tversky = focal_tversky_loss(outputs, targets)
    return 0.5 * bce + 0.5 * focal_tversky

# Early Stopping 類別
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'best_model.pth')
        self.val_loss_min = val_loss

# 模型驗證
def validate_model(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            x, y = data
            x, y = x.to(device).float(), y.to(device).float()
            outputs = model(x)
            loss = criterion(outputs, y.unsqueeze(1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

# 模型訓練與驗證
def train_model_with_validation(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=30, device='cpu'):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    scaler = GradScaler()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            x, y = data
            x, y = x.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # 驗證模型
        val_loss = validate_model(model, criterion, val_loader, device)
        scheduler.step()

        print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model

# 設置參數
batch_size = 16
num_epochs = 30

# 圖像增強技術：同時應用於圖像和 mask
transform = A.Compose([
    A.RandomResizedCrop(256, 256, scale=(0.7, 1.0)),
    A.RandomBrightnessContrast(p=0.3),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.6),
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=0.5),
    A.GaussianBlur(p=0.4),
    A.GridDistortion(p=0.5),
    A.ElasticTransform(p=0.4),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# 訓練集與驗證集路徑
train_img_path = 'C:\\Users\\littl\\Python\\IC\\HW2_FoodSegmentation\\food\\food\\trainingimages'

trainset = TrafficDataset(train_img_path, has_labels=True, transform=transform)
valsize = int(len(trainset) * 0.2)
trainsize = len(trainset) - valsize
train_set, val_set = torch.utils.data.random_split(trainset, [trainsize, valsize])

# 建立 dataloader
train_dataloaders = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloaders = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

# 初始化 U-Net++ 模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
in_channel, out_channel = 3, 1
model = UnetPlusPlus(encoder_name="efficientnet-b5", encoder_weights="imagenet", in_channels=in_channel, classes=out_channel).to(device)

# 設定損失函數與優化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# 訓練模型並加入驗證過程
model = train_model_with_validation(model, criterion, optimizer, scheduler, train_dataloaders, val_dataloaders, num_epochs=num_epochs, device=device)

# 測試集部分
test_img_path = 'C:\\Users\\littl\\Python\\IC\\HW2_FoodSegmentation\\food\\food\\testimages'

transform = A.Compose([
    A.Resize(256, 256),  # 調整大小
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

testset = TrafficDataset(test_img_path, has_labels=False, transform=transform)
bs = 64
test_dataloaders = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=0)

# 加載最佳模型權重
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

result = np.empty([0, 1, 256, 256])
pbar = tqdm(total=len(test_dataloaders) + 1, ncols=150)

for x in test_dataloaders:
    if x is None:
        continue  # 跳過無效數據
    inputs = x.to(device).float()
    outputs = model(inputs)
    outputs = torch.sigmoid(outputs)
    result = np.concatenate((result, outputs.cpu().detach().numpy()), axis=0)
    pbar.update(1)

pbar.close()

# 產生最終結果
threshold = 0.5
pred = np.where(result >= threshold, 1, 0)

# RLE 編碼與儲存
def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def convert_into_rle(mask, pred_root):
    encoding = []
    for i, m in enumerate(mask[:]):
        encoding.append(rle_encoding(m))

    with open(pred_root, 'w') as csvfile:
        csvfile.write("ImageId,EncodedPixels\n")
        for i, m in enumerate(encoding):
            conv = lambda l: ' '.join(map(str, l))
            text = '{},{}'.format(i, conv(encoding[i])) + '\n'
            csvfile.write(text)

convert_into_rle(pred, 'C:\\Users\\littl\\Python\\IC\\HW2_FoodSegmentation\\HW2pred.csv')
