import os
import glob
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# 固定隨機種子
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

same_seeds(6666)

# 資料集處理
def get_dataset(root, image_size=(64, 64)):
    def preprocess_image(filepath):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = (img / 127.5) - 1  # Normalize to [-1, 1]
        return img

    dataset = tf.data.Dataset.list_files(os.path.join(root, "*"))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

workspace_dir = 'D:\\彰師大研究所\\Python\\IC\\HW3_AnimeFaceGeneration\\anim'

if os.path.exists(workspace_dir):
    print(f"The directory {workspace_dir} exists.")
else:
    print(f"The directory {workspace_dir} does not exist.")

# Generator 模型
def build_generator(z_dim):
    model = tf.keras.Sequential([
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(z_dim,)),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Reshape((4, 4, 512)),

        layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')
    ])
    return model

# Discriminator 模型
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3)),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(512, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(1, kernel_size=4, strides=1, padding='valid')
    ])
    return model

# WGAN-GP 的梯度懲罰計算
def gradient_penalty(discriminator, real_images, fake_images, batch_size):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        pred = discriminator(interpolated_images)
    grads = tape.gradient(pred, interpolated_images)
    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    return tf.reduce_mean((grads_norm - 1.0) ** 2)

# WGAN-GP 訓練器
class TrainerWGAN:
    def __init__(self, z_dim, batch_size, lr, n_critic, device):
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.generator = build_generator(z_dim)
        self.discriminator = build_discriminator()
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.device = device

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            progress_bar = tqdm(dataset.batch(self.batch_size), desc=f"Epoch {epoch + 1}/{epochs}")
            for real_images in progress_bar:
                real_images = tf.convert_to_tensor(real_images)
                batch_size = tf.shape(real_images)[0]

                # 訓練 Discriminator
                for _ in range(self.n_critic):
                    noise = tf.random.normal([batch_size, self.z_dim])
                    with tf.GradientTape() as disc_tape:
                        fake_images = self.generator(noise, training=True)
                        real_output = self.discriminator(real_images, training=True)
                        fake_output = self.discriminator(fake_images, training=True)
                        gp = gradient_penalty(self.discriminator, real_images, fake_images, batch_size)
                        disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + 10.0 * gp
                    disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                    self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

                # 訓練 Generator
                noise = tf.random.normal([batch_size, self.z_dim])
                with tf.GradientTape() as gen_tape:
                    fake_images = self.generator(noise, training=True)
                    fake_output = self.discriminator(fake_images, training=True)
                    gen_loss = -tf.reduce_mean(fake_output)
                gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

            progress_bar.set_postfix(gen_loss=gen_loss.numpy(), disc_loss=disc_loss.numpy())

# 設定參數
config = {
    "z_dim": 100,
    "batch_size": 64,
    "lr": 1e-4,
    "n_epoch": 70,
    "n_critic": 5,
}
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
trainer = TrainerWGAN(config["z_dim"], config["batch_size"], config["lr"], config["n_critic"], device)

# 資料集與訓練
dataset = get_dataset(workspace_dir)
trainer.train(dataset, epochs=config["n_epoch"])
