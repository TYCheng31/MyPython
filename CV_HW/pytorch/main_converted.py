import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 設定隨機種子以確保結果可重現
np.random.seed(42)
torch.manual_seed(42)

# 1. 載入資料
class ImageDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, target_size=(128, 128)):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
        self.target_size = target_size

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_img = self.load_image(self.noisy_files[idx])
        clean_img = self.load_image(self.clean_files[idx])
        return torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0), torch.tensor(clean_img, dtype=torch.float32).unsqueeze(0)

    def load_image(self, filepath):
        img = Image.open(filepath).convert('L').resize(self.target_size)
        img_array = np.array(img) / 255.0
        return img_array

clean_images_folder = 'E:\\CV\\Grayscale'
noisy_images_folder = 'E:\\CV\\Noise'

dataset = ImageDataset(noisy_images_folder, clean_images_folder)
train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. 定義 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # 編碼器
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        
        # 解碼器
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = self.conv_block(1024, 512)
        
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec7 = self.conv_block(512, 256)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec8 = self.conv_block(256, 128)
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec9 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 編碼器
        x1 = self.enc1(x)
        x2 = self.enc2(nn.MaxPool2d(2)(x1))
        x3 = self.enc3(nn.MaxPool2d(2)(x2))
        x4 = self.enc4(nn.MaxPool2d(2)(x3))
        x5 = self.enc5(nn.MaxPool2d(2)(x4))
        
        # 解碼器
        x6 = self.up6(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.dec6(x6)
        
        x7 = self.up7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.dec7(x7)
        
        x8 = self.up8(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.dec8(x8)
        
        x9 = self.up9(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.dec9(x9)
        
        return self.final(x9)

# 初始化模型
model = UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 訓練模型
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 30

# 加入保存模型功能
best_loss = float('inf')
save_path = 'E:\\CV\\best_denoise_unet_model50.pth'

# 訓練過程的Loss記錄
train_losses = []

print("開始訓練模型...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print(f"模型改進，已保存到 {save_path}")

print("訓練完成！最佳模型已保存。")

# 繪製Loss下降的XY圖
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.savefig('E:\\CV\\training_loss_curve.png')  # 保存圖像
plt.show()

# 4. 評估模型
def evaluate_model(model, loader):
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    num_images = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy).cpu().numpy()
            clean = clean.cpu().numpy()
            for i in range(len(outputs)):
                psnr = peak_signal_noise_ratio(clean[i, 0], outputs[i, 0], data_range=1.0)
                ssim = structural_similarity(clean[i, 0], outputs[i, 0], data_range=1.0)
                psnr_total += psnr
                ssim_total += ssim
                num_images += 1
    avg_psnr = psnr_total / num_images
    avg_ssim = ssim_total / num_images
    print(f"測試集平均 PSNR: {avg_psnr:.2f} dB, 平均 SSIM: {avg_ssim:.4f}")

# 載入最佳模型進行評估
model.load_state_dict(torch.load(save_path))
print("已加載最佳模型！")
evaluate_model(model, test_loader)
