import os
from pathlib import Path
import random
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

IMG_WIDTH = 1080 // 3
IMG_HEIGHTS = 1920 // 3
IMG_CHANNELS = 3
LEARNING_RATE = 1e-4
NUM_DIFFUSION_STEPS = 16
BATCH_SIZE = 16

time_bar = 1 - np.linspace(0, 1.0, NUM_DIFFUSION_STEPS + 1) # linspace for timesteps

def forward_noise(x, t):
    a = time_bar[t]      # based on t
    b = time_bar[t + 1]  # image for t + 1
    
    noise = np.random.normal(size=x.shape)  # noise mask
    a = a.reshape((-1, 1, 1, 1))
    b = b.reshape((-1, 1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b

def generate_ts(num):
    return np.random.randint(0, NUM_DIFFUSION_STEPS, size=num)


def conv_block(x_img, x_ts):
    x_parameter = layers.Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_parameter = layers.Activation('relu')(x_parameter)

    time_parameter = layers.Dense(128)(x_ts)
    time_parameter = layers.Activation('relu')(time_parameter)
    time_parameter = layers.Reshape((1, 1, 128))(time_parameter)
    x_parameter = x_parameter * time_parameter
    
    x_out = layers.Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_out = x_out + x_parameter
    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation('relu')(x_out)
    
    return x_out

def make_diffusion_model():
    x = x_input = layers.Input(shape=(IMG_HEIGHTS, IMG_WIDTH, IMG_CHANNELS), name='x_input')
    
    x_ts = x_ts_input = layers.Input(shape=(1,), name='x_ts_input')
    x_ts = layers.Dense(192)(x_ts)
    x_ts = layers.LayerNormalization()(x_ts)
    x_ts = layers.Activation('relu')(x_ts)
    
    # ----- left ( down ) -----
    x = x32 = conv_block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x16 = conv_block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x8 = conv_block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x4 = conv_block(x, x_ts)

    # # ----- MLP -----
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, x_ts])
    x = layers.Dense(128)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(80 * 45 * 128)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape((80, 45, 128))(x) #(4,4,32)
    
    # # ----- right ( up ) -----
    x = layers.Concatenate()([x, x4])
    x = conv_block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x8])
    x = conv_block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x16])
    x = conv_block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x32])
    x = conv_block(x, x_ts)
    
    # # ----- output -----
    x = layers.Conv2D(3, kernel_size=1, padding='same')(x)
    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model


def make_classifier():
    input_image = layers.Input(shape=(IMG_HEIGHTS, IMG_WIDTH, IMG_CHANNELS), name='discriminator_input')
    
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(input_image)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # Real/Fake classification
    real_fake_output = layers.Dense(1, activation='sigmoid', name='real_fake_output')(x)
    
    # Image label classification
    label_output = layers.Dense(4, activation='softmax', name='label_output')(x)  # Assuming 10 classes (e.g., CIFAR-10)
    
    model = tf.keras.models.Model(input_image, [real_fake_output,label_output])
    return model

def generator_loss(real_image, generated_image, discriminator_label, true_label, image_to_be):
    # Mean Squared Error (MSE) between real and generated images
    mse_loss = tf.reduce_mean(tf.square(image_to_be - generated_image))
    
    # Cross-entropy loss for classification (label output from discriminator)
    classification_loss = tf.keras.losses.categorical_crossentropy(true_label, discriminator_label)
    
    total_loss = mse_loss + tf.reduce_mean(classification_loss)
    return total_loss

def discriminator_loss(real_output, fake_output, real_labels, generated_labels):
    
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    
    # Cross-entropy loss for label classification (real image labels)
    label_loss = tf.keras.losses.categorical_crossentropy(real_labels, generated_labels)
    
    total_loss = tf.reduce_mean(real_loss + fake_loss + label_loss)
    return total_loss

# Optimizer
optimizer_diff = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer_disc = tf.keras.optimizers.Adam(learning_rate=1e-4)


def train_step(generator, discriminator, real_images, real_labels, timesteps):
    # Generate noisy images
    ts = generate_ts(len(real_images))
    noisy_images_out, image_to_be = forward_noise(real_images, ts)
    
    # Gradient calculation and update for discriminator
    with tf.GradientTape() as disc_tape:
        generated_images = generator([noisy_images_out, ts], training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        disc_loss = discriminator_loss(real_output[0], fake_output[0], real_labels, fake_output[1])
    
    # print("Calculated Discriminator loss")
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    # Gradient calculation and update for generator
    with tf.GradientTape() as gen_tape:
        generated_images = generator([noisy_images_out, ts], training=True)
        disc_class_output = discriminator(generated_images, training=True)[1]
        
        gen_loss = generator_loss(real_images, generated_images, disc_class_output, real_labels, image_to_be)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    optimizer_diff.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    return gen_loss, disc_loss

# Function to train the models on dataset
def train(generator, discriminator, dataset, epochs, batch_size, timesteps):
    for epoch in range(epochs):
        for real_images, real_labels in dataset:
            gen_loss, disc_loss = train_step(generator, discriminator, real_images, real_labels, timesteps)
        
        print(f'Epoch {epoch+1}/{epochs}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

# Save random images generated by the generator for each class
def save_generated_images(data_path, generator, epoch, num_classes=4, num_images=1, output_dir='generated_images'):
    os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
    
    # Generate random images for each class
    data_path = Path(data_path)
    device_types = np.array([item.name for item in data_path.glob('*') if not item.name.startswith('.')])

    num_images = len(device_types)
    images = np.empty((BATCH_SIZE, 1920//3, 1080//3, 3), dtype=np.float32)

    for i in range(num_images):
        device_path = Path(os.path.join(data_path, device_types[i]))
        image_paths = list((device_path.glob(f'Validation/**/*.jpg')))
        
        for d in range(BATCH_SIZE):
            image = get_image(image_paths[d])
            images[i, ...] = image
        # img = random.choice(image_paths)

        ts = generate_ts(len(images))
        noisy_img, _ = forward_noise(images,ts)
        
        generated_images = generator([noisy_img, ts], training=False)
        
        # Save generated images
        for c in range(num_images):
        # for c in range(len(generated_images)):
            img = generated_images[c]
            img_norm = (img.numpy() * 255).astype(np.uint8)
            # plt.imsave(os.path.join(output_dir, f'epoch_{epoch}_class_{device_types[i]}_original.png'), img_norm)
            plt.imsave(os.path.join(output_dir, f'im_number_{c}_class_{device_types[i]}_original.png'), img_norm)
            
            plt.close()


def get_image(image_path):
    img = cv2.imread(image_path)
    if img.shape[0] == 1080 and img.shape[1] == 1920:
            img = np.transpose(img, (1, 0, 2))
    height, width = img.shape[:2]
    img = cv2.resize(img, (width//3, height//3))
    
    # img = apply_cfa(img)
    # Normalize the pixel values
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img