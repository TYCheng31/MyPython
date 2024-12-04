from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
from model import YourGANModel  # 假設你的 GAN 模型保存在 model.py 中

app = Flask(__name__)

# 初始化你的 GAN 模型
model = YourGANModel()
model.load_state_dict(torch.load('your_model.pth'))  # 請確保替換為你實際的模型文件路徑
model.eval()

# 影像處理函數
def denoise_image(image):
    # 將圖像轉換為 PyTorch 張量並進行預處理
    input_image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output_image = model(input_image)  # 使用模型進行去雜訊處理
    return output_image.squeeze().numpy()

@app.route('/')
def index():
    return render_template('index.html')  # 顯示主頁

@app.route('/process', methods=['POST'])
def process_image():
    # 從請求中獲取影像
    file = request.files['image']
    img_array = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    # 去雜訊
    denoised_img = denoise_image(img)
    
    # 返回處理後的圖像
    _, img_encoded = cv2.imencode('.png', denoised_img)
    return img_encoded.tobytes()

if __name__ == '__main__':
    app.run(debug=True)
