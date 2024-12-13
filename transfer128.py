from PIL import Image
import numpy as np

# 打开图片文件
image_path = 'E:\\Dataset\\Noise\\000001.png'  # 替换为你的图片路径
image = Image.open(image_path)

# 将图片调整为128x128
image_resized = image.resize((128, 128))

# 将图片转换为numpy数组
image_array = np.array(image_resized)

# 定义高斯噪声的参数
mean = 0        # 均值
sigma = 25      # 标准差
noise = np.random.normal(mean, sigma, image_array.shape)

# 将高斯噪声添加到图片中
noisy_image_array = np.clip(image_array + noise, 0, 255)  # 确保像素值在0到255之间

# 将噪声图片转换回PIL图像
noisy_image = Image.fromarray(noisy_image_array.astype(np.uint8))

# 保存添加噪声后的图片
noisy_image.save('noisy_image.jpg')  # 保存为新的文件

# 显示添加噪声后的图片（可选）
noisy_image.show()

