import cv2
import numpy as np
import torch
import torch.nn as nn

# 定義 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
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
        x1 = self.enc1(x)
        x2 = self.enc2(nn.MaxPool2d(2)(x1))
        x3 = self.enc3(nn.MaxPool2d(2)(x2))
        x4 = self.enc4(nn.MaxPool2d(2)(x3))
        x5 = self.enc5(nn.MaxPool2d(2)(x4))
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

# 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('E:\\model\\best_denoise_unet_model50.pth', map_location=device))
model.eval()
print("模型已成功載入！")

# 定義添加雜訊的函數
def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# 處理單幀影像
def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 轉灰階
    resized_frame = cv2.resize(gray_frame, (512, 512))  # 調整大小
    noisy_frame = add_noise(resized_frame)  # 添加雜訊

    # 用模型處理去雜訊
    img_tensor = torch.tensor(noisy_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        denoised_tensor = model(img_tensor).cpu().squeeze().numpy()
    denoised_frame = (np.clip(denoised_tensor, 0, 1) * 255).astype(np.uint8)

    return noisy_frame, denoised_frame

# 開啟攝影機
cap = cv2.VideoCapture(0)
print("按 'q' 鍵退出程式。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("無法捕捉影像，請檢查攝影機。")
        break

    noisy_frame, denoised_frame = process_frame(frame)  # 處理畫面

    # 合併影像顯示
    combined_frame = cv2.hconcat([noisy_frame, denoised_frame])

    # 顯示影像
    cv2.imshow('Noisy (Left) | Denoised (Right)', combined_frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
