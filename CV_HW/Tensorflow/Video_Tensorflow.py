import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model

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
def preprocess_image(frame, target_size=(128, 128)):
    """
    將攝像頭影像轉換為適合模型的格式。
    """
    resized_frame = cv2.resize(frame, target_size)  # 調整大小
    normalized_frame = resized_frame / 255.0  # 正規化到 [0, 1]
    img_tensor = np.expand_dims(normalized_frame, axis=(0, -1))  # 增加批次和通道維度
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

# 開啟攝像頭
cap = cv2.VideoCapture(0)
print("開啟攝影機，按 'q' 鍵退出程式。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("無法捕捉影像，請檢查攝影機。")
        break

    # 取得原始大小並轉為灰階影像
    original_size = (frame.shape[1], frame.shape[0])  # (width, height)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 添加雜訊到影像
    noisy_frame = add_noise(gray_frame)

    # 預處理雜訊影像
    input_tensor = preprocess_image(noisy_frame)

    # 使用模型進行去雜訊
    denoised_tensor = model.predict(input_tensor)

    # 後處理影像
    denoised_frame = postprocess_image(denoised_tensor, original_size)

    # 合併影像以顯示對比效果
    combined_frame = cv2.hconcat([noisy_frame, denoised_frame])
    cv2.imshow('Noisy (Left) | Denoised (Right)', combined_frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉攝影機和視窗
cap.release()
cv2.destroyAllWindows()
print("程式已退出。")
