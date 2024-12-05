import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 定義 U-Net 模型（與之前一致）
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

# 加載模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('E:\\model\\best_denoise_unet_model50.pth', map_location=device))
model.eval()

# 實時影像處理
def process_frame(frame):
    # 將影像轉換為灰階
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 調整大小
    resized_frame = cv2.resize(gray_frame, (128, 128)) / 255.0
    # 轉換為張量
    tensor_frame = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # 模型推理
    with torch.no_grad():
        output = model(tensor_frame).cpu().squeeze().numpy()
    # 將結果還原為影像
    output_image = (np.clip(output, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(output_image, (frame.shape[1], frame.shape[0]))  # 還原到原始大小

# 開啟攝像頭
cap = cv2.VideoCapture(0)

print("按 'q' 鍵退出程式。")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 處理影像
    denoised_frame = process_frame(frame)

    # 顯示原始影像與去雜訊影像
    combined_frame = np.hstack((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), denoised_frame))
    cv2.imshow('Original (Left) | Denoised (Right)', combined_frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
