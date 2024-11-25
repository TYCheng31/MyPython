import os
import glob
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# 設定隨機種子
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

same_seeds(2024)

# 資料集處理
def get_dataset(root, img_size=(64, 64)):
    fnames = glob.glob(os.path.join(root, '*'))
    images = []
    for fname in fnames:
        img = load_img(fname, target_size=img_size)
        img = img_to_array(img)
        img = (img / 127.5) - 1  # Normalize to [-1, 1]
        images.append(img)
    return np.array(images)

# 生成器 (DCGAN Generator)
def build_generator(z_dim):
    model = tf.keras.Sequential([
        layers.Dense(4 * 4 * 512, input_dim=z_dim),
        layers.Reshape((4, 4, 512)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh"),
    ])
    return model

# 判別器 (DCGAN Discriminator)
def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(512, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model

# DCGAN 主類別
class DCGAN:
    def __init__(self, img_shape, z_dim, lr=0.0002):
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.generator = build_generator(z_dim)
        self.discriminator = build_discriminator(img_shape)

        self.generator_optimizer = Adam(lr, beta_1=0.5)
        self.discriminator_optimizer = Adam(lr, beta_1=0.5)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.z_dim))

        # 標籤設置
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # 訓練判別器
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            d_loss_real = self.loss_fn(real_labels, real_logits)
            d_loss_fake = self.loss_fn(fake_labels, fake_logits)
            d_loss = (d_loss_real + d_loss_fake) / 2

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # 訓練生成器
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(real_labels, fake_logits)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return d_loss, g_loss

    def train(self, dataset, batch_size, epochs, save_dir="output"):
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(dataset, desc=f"Epoch {epoch + 1}")

            for step, real_images in enumerate(progress_bar):
                d_loss, g_loss = self.train_step(real_images)
                progress_bar.set_postfix(d_loss=d_loss.numpy(), g_loss=g_loss.numpy())

            # 每 10 個 epoch 保存生成圖片
            if (epoch + 1) % 10 == 0:
                self.save_images(epoch, save_dir)

    def save_images(self, epoch, save_dir):
        random_latent_vectors = tf.random.normal(shape=(16, self.z_dim))
        fake_images = self.generator(random_latent_vectors)
        fake_images = (fake_images + 1) / 2.0

        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        idx = 0
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(fake_images[idx].numpy())
                axs[i, j].axis("off")
                idx += 1
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch + 1}.png"))
        plt.close()

# 主函數
if __name__ == '__main__':
    workspace_dir = "D:\\彰師大研究所\\Python\\IC\\HW3_AnimeFaceGeneration\\anim"
    batch_size = 64
    z_dim = 100
    img_shape = (64, 64, 3)

    if os.path.exists(workspace_dir):
        print(f"The directory {workspace_dir} exists.")
    else:
        print(f"The directory {workspace_dir} does not exist.")
        exit()

    data = get_dataset(workspace_dir)
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=1000).batch(batch_size)

    dcgan = DCGAN(img_shape=img_shape, z_dim=z_dim)
    dcgan.train(dataset, batch_size=batch_size, epochs=50)
