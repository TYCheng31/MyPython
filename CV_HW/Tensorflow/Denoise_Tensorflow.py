import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# 自定義修正函數，用於移除不支持的參數
def fix_conv2d_transpose_config(cls, config):
    """
    修正 Conv2DTranspose 的配置，移除不支持的參數（例如 groups）。
    """
    config = config.copy()  # 確保不影響原始 config
    if "groups" in config:
        del config["groups"]  # 刪除不支持的參數
    return cls.from_config(config)

# 自定義處理器，處理 Conv2DTranspose
def custom_objects_handler():
    return {"Conv2DTranspose": lambda **kwargs: fix_conv2d_transpose_config(Conv2DTranspose, kwargs)}

# 嘗試載入模型
try:
    print("載入最佳去雜訊模型...")
    with tf.keras.utils.custom_object_scope(custom_objects_handler()):
        model = load_model('D:\\彰師大研究所\\Python\\CV\\FinalProject\\Denoise-Project-main\\best_denoising_unet_model.h5')
    print("模型已成功載入！")
except Exception as e:
    print(f"載入模型時發生錯誤：{e}")
    exit()

# 定義添加雜訊的函數
def add_noise(image):
    """
    向圖像添加高斯雜訊。
    """
    noise = np.random.normal(0, 25, image.shape).astype(np.int16)  # 使用 int16 防止溢出
    noisy_image = image.astype(np.int16) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # 確保數據範圍在 [0, 255]
    return noisy_image

# 定義影像預處理函數
def preprocess_image(image, target_size=(128, 128)):
    """
    將圖像轉換為適合模型的格式。
    """
    resized_image = cv2.resize(image, target_size)  # 調整大小
    normalized_image = resized_image / 255.0  # 正規化到 [0, 1]
    img_tensor = np.expand_dims(normalized_image, axis=(0, -1))  # 增加批次和通道維度
    return img_tensor

# 定義影像後處理函數
def postprocess_image(denoised_tensor, original_size):
    """
    將模型輸出轉換為顯示格式。
    """
    denoised_img = denoised_tensor.squeeze()  # 去掉多餘維度
    denoised_img = np.clip(denoised_img, 0.0, 1.0)  # 保持在 [0, 1]
    denoised_img = (denoised_img * 255).astype(np.uint8)  # 轉回 [0, 255]
    denoised_img = cv2.resize(denoised_img, original_size)  # 還原到原始大小
    return denoised_img

# 定義文件選擇與處理函數
def process_image():
    # 打開文件選擇對話框
    file_path = filedialog.askopenfilename(
        title="選擇一張圖像",
        filetypes=(("圖片檔案", "*.jpg *.png *.jpeg"), ("所有檔案", "*.*"))
    )
    
    if not file_path:
        print("未選擇任何圖像。")
        return
    
    # 加載圖像
    original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print("無法加載選擇的圖像，請檢查檔案。")
        return
    
    # 添加雜訊
    noisy_image = add_noise(original_image)
    
    # 預處理雜訊圖像
    input_tensor = preprocess_image(noisy_image)
    
    # 使用模型進行去雜訊
    denoised_tensor = model.predict(input_tensor)
    
    # 後處理影像
    denoised_image = postprocess_image(denoised_tensor, original_image.shape[::-1])
    
    # 顯示對比圖
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(noisy_image, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Denoised Image")
    plt.imshow(denoised_image, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# 使用 Tkinter 建立簡單的 GUI
if __name__ == "__main__":
    root = Tk()
    root.title("去雜訊模型測試")
    root.geometry("300x150")
    root.resizable(False, False)

    # 隱藏主窗口，只顯示文件選擇對話框
    root.withdraw()

    print("打開文件選擇器...")
    process_image()
