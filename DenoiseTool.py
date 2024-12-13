import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, filedialog, Button, Label, Frame, messagebox, font
from PIL import Image, ImageTk

# 直接嵌入訓練模型的結構
def multi_scale_conv_block(inputs, filters):
    conv_1x1 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(inputs)
    conv_3x3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv_5x5 = layers.Conv2D(filters, (5, 5), activation='relu', padding='same')(inputs)
    concat = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])
    return concat

def unet_generator(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)

    # 編碼器
    conv1 = multi_scale_conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = multi_scale_conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = multi_scale_conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = multi_scale_conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = multi_scale_conv_block(pool4, 1024)

    # 解碼器
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.Concatenate()([up6, conv4])
    conv6 = multi_scale_conv_block(up6, 512)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.Concatenate()([up7, conv3])
    conv7 = multi_scale_conv_block(up7, 256)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.Concatenate()([up8, conv2])
    conv8 = multi_scale_conv_block(up8, 128)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.Concatenate()([up9, conv1])
    conv9 = multi_scale_conv_block(up9, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 建立主視窗
root = Tk()
root.title("去噪小工具")
root.geometry('1920x1080')

# 定義粗體字體（在初始化主視窗之後）
bold_font = font.Font(weight='bold')

# 載入訓練模型的權重
generator = unet_generator()
try:
    # 載入 main 訓練的權重
    generator.load_weights('E:\\Denoise\\best_generator_model.h5')  # 載入訓練過的生成器權重
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model weights: {e}")

# 去噪處理
def denoise_image(filepath):
    target_size = (256, 256)
    img = load_img(filepath, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img) / 255.0  # 正規化
    img_input = np.expand_dims(img_array, axis=0)
    denoised_img = generator.predict(img_input)[0]  # 使用載入的生成器進行預測
    denoised_img_uint8 = (denoised_img * 255).astype(np.uint8)
    denoised_img_pil = tf.keras.preprocessing.image.array_to_img(denoised_img_uint8, scale=False)
    return denoised_img_pil

# 選擇圖片並進行去噪
def select_and_denoise():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not filepath:
        return
    try:
        original_img = Image.open(filepath).resize((512, 512))  # 修改為 512x512
        original_img_tk = ImageTk.PhotoImage(original_img)
        original_image_label.config(image=original_img_tk)
        original_image_label.image = original_img_tk

        denoised_image_label.config(text="照片處理中...", font=bold_font)
        root.update_idletasks()

        denoised_img = denoise_image(filepath)
        show_result(denoised_img)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to denoise image: {e}")
    finally:
        denoised_image_label.config(text="", font=bold_font)

# 顯示處理後的結果
def show_result(denoised_img):
    denoised_img = denoised_img.resize((512, 512))  # 修改為 512x512
    denoised_img_tk = ImageTk.PhotoImage(denoised_img)
    denoised_image_label.config(image=denoised_img_tk)
    denoised_image_label.image = denoised_img_tk

    global processed_image
    processed_image = denoised_img


# 下載處理後的圖片
def download_denoised_image():
    if processed_image is None:
        messagebox.showerror("Error", "No processed image to download.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if save_path:
        try:
            processed_image.save(save_path)
            messagebox.showinfo("Success", f"Denoised image saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

# 清除圖片
def clear_images():
    original_image_label.config(image='', font=bold_font)
    denoised_image_label.config(image='', font=bold_font)
    original_image_label.image = None
    denoised_image_label.image = None
    global processed_image
    processed_image = None

# 退出應用程式
def exit_application():
    root.quit()

# 處理資料夾中的圖片
def process_folder():
    input_folder = filedialog.askdirectory(title="選擇要處理的資料夾")
    if not input_folder:
        return

    output_folder = filedialog.askdirectory(title="選擇保存處理後圖片的資料夾")
    if not output_folder:
        return

    supported_formats = ('.png', '.jpg', '.jpeg')

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
    if not image_files:
        messagebox.showinfo("資訊", "選擇的資料夾中沒有支援的圖片檔案。")
        return

    success_count = 0
    fail_count = 0
    failed_files = []

    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)
        try:
            denoised_img = denoise_image(input_path)
            denoised_img.save(output_path)
            success_count += 1
        except Exception as e:
            fail_count += 1
            failed_files.append(img_file)

    result_message = f"處理完成！\n成功處理 {success_count} 張圖片。"
    if fail_count > 0:
        result_message += f"\n失敗 {fail_count} 張圖片。\n失敗的檔案:\n" + "\n".join(failed_files)
    
    messagebox.showinfo("處理結果", result_message)

# 建立主框架
main_frame = Frame(root)
main_frame.pack(pady=20, fill='both', expand=True)

content_frame = Frame(main_frame)
content_frame.pack(side='top', fill='both', expand=True)

button_frame = Frame(content_frame)
button_frame.grid(row=0, column=0, padx=20, pady=20, sticky='nw')

image_frame = Frame(content_frame)
image_frame.grid(row=0, column=1, padx=20, pady=20, sticky='nw')

# 建立按鈕並應用粗體字體
folder_button = Button(button_frame, text="上傳資料夾", command=process_folder, height=4, width=20, font=bold_font)
folder_button.pack(pady=5, anchor='w')

upload_button = Button(button_frame, text="上傳圖片", command=select_and_denoise, height=4, width=20, font=bold_font)
upload_button.pack(pady=5, anchor='w')

download_button = Button(button_frame, text="下載圖片", command=download_denoised_image, height=4, width=20, font=bold_font)
download_button.pack(pady=5, anchor='w')

clear_button = Button(button_frame, text="清除圖片", command=clear_images, height=4, width=20, font=bold_font)
clear_button.pack(pady=5, anchor='w')

exit_button = Button(button_frame, text="退出工具", command=exit_application, height=4, width=20, font=bold_font)
exit_button.pack(pady=5, anchor='w')

# 建立圖片標籤
original_image_label = Label(image_frame)
original_image_label.grid(row=0, column=0, padx=10, pady=10)

denoised_image_label = Label(image_frame, font=bold_font)
denoised_image_label.grid(row=0, column=1, padx=10, pady=10)

processed_image = None

# 開始主迴圈
root.mainloop()
