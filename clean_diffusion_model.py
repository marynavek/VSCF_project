import os
import random
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

def add_noise(image, noise_level):
    """Add Gaussian noise to the image."""
    noise = tf.random.normal(shape=tf.shape(image))
    noisy_image = image + noise_level * noise
    return noisy_image

def forward_noise(x, t):
    a = time_bar[t]      # base on t
    b = time_bar[t + 1]  # image for t + 1
    
    noise = np.random.normal(size=x.shape)  # noise mask
    a = a.reshape((-1, 1, 1, 1))
    b = b.reshape((-1, 1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b

def generate_ts(num):
    return np.random.randint(0, NUM_DIFFUSION_STEPS, size=num)

def conv_block(input_image, x_ts):
    x_parameter = layers.Conv2D(128, kernel_size=3, padding='same')(input_image)
    x_parameter = layers.Activation('relu')(x_parameter)

    time_parameter = layers.Dense(128)(x_ts)
    time_parameter = layers.Activation('relu')(time_parameter)
    time_parameter = layers.Reshape((1, 1, 128))(time_parameter)
    x_parameter = x_parameter * time_parameter
    
    # -----
    x_out = layers.Conv2D(128, kernel_size=3, padding='same')(input_image)
    x_out = x_out + x_parameter
    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation('relu')(x_out)
    
    return x_out

def diffusion_model_build():
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


    

def linear_beta_scheduler(timesteps, start=1e-4, end=0.02):
    return np.linspace(start, end, timesteps)

def get_mse_loss(noisy_images, clean_images, model_output):
    return tf.reduce_mean(tf.square(model_output-clean_images))

def build_diffusion_trainable():
    inputs = layers.Input(shape=(IMG_HEIGHTS,IMG_WIDTH,IMG_CHANNELS))

    #build encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # Output: (320, 180)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # Output: (160, 90)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)  # Output: (80, 45)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)  # Output: (40, 22)

    # Decoder
    up1 = layers.UpSampling2D(size=(2, 2))(pool4)  # Output: (80, 44)
    up1 = layers.Conv2D(512, (2, 2), padding='same')(up1)
    reshaped_up1 = layers.Resizing(80,45)(up1)
    concat1 = layers.Concatenate()([reshaped_up1, conv4])  # Shape matches
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat1)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up2 = layers.UpSampling2D(size=(2, 2))(conv5)  # Output: (160, 88)
    up2 = layers.Conv2D(256, (2, 2), padding='same')(up2)
    concat2 = layers.Concatenate()([up2, conv3])  # Shape matches
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat2)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up3 = layers.UpSampling2D(size=(2, 2))(conv6)  # Output: (320, 176)
    up3 = layers.Conv2D(128, (2, 2), padding='same')(up3)
    concat3 = layers.Concatenate()([up3, conv2])  # Shape matches
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up4 = layers.UpSampling2D(size=(2, 2))(conv7)  # Output: (640, 352)
    up4 = layers.Conv2D(64, (2, 2), padding='same')(up4)
    concat4 = layers.Concatenate()([up4, conv1])  # Shape matches
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    outputs = layers.Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(conv8)

    return tf.keras.Model(inputs, outputs)

class DiffusionModel(tf.keras.Model):
    def __init__(self, unet, timesteps=NUM_DIFFUSION_STEPS):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.beta = linear_beta_scheduler(timesteps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = np.cumprod(self.alpha)

    def call(self, inputs, training=None):
        noisy_images, t = inputs
        return self.unet(noisy_images)

    def diffusion_step(self, images, t):
        """Add noise to images based on the diffusion schedule."""
        noise_level = np.sqrt(self.beta[t])
        noisy_images = add_noise(images, noise_level)
        return noisy_images
    

def combined_loss(original_image, noisy_images, reconstructed_image, camera_label, classifier):
    mse_loss = get_mse_loss(noisy_images, original_image, reconstructed_image)

    pred_camera_model = classifier(reconstructed_image)
    classification_loss = tf.keras.losses.categorical_crossentropy(camera_label, pred_camera_model)
    
    print(f"Classification Loss: ", classification_loss)
    # Total loss is a weighted combination of reconstruction and classification loss
    total_loss = mse_loss + tf.reduce_mean(classification_loss)
    return total_loss


def train_step(diffusion_model, classifier, input_image, camera_label):
    # clean_images = data[0]
    # batch_size = tf.shape(clean_images)[0]

    # Randomly sample time steps
    t = tf.random.uniform([BATCH_SIZE], 0, NUM_DIFFUSION_STEPS, dtype=tf.int32)
    print(t.shape)
    print(input_image.shape)
    # Add noise to the clean images
    noisy_images = diffusion_model.diffusion_step(input_image, t)

    with tf.GradientTape() as tape:
        # Predict clean images from noisy images
        predicted_images = diffusion_model((noisy_images, t), training=True)

        # Compute the loss
        loss = combined_loss(original_image=input_image, noisy_images=noisy_images, reconstructed_image=predicted_images, camera_label=camera_label, classifier=classifier)

    # Apply gradients
    gradients = tape.gradient(loss, diffusion_model.trainable_variables)
    diffusion_model.optimizer.apply_gradients(zip(gradients, diffusion_model.trainable_variables))

    return {"loss": loss}


def train_step_new(diffusion_model, classifier, input_images, camera_label):
    # clean_images = data[0]
    # batch_size = tf.shape(clean_images)[0]

    # Randomly sample time steps
    x_ts = generate_ts(len(input_images))
    x_a, x_b = forward_noise(input_images, x_ts)
    # print(input_image.shape)
    # Add noise to the clean images
    loss = diffusion_model.train_on_batch([x_a, x_ts], x_b)
    predicted_images()
    # noisy_images = diffusion_model.diffusion_step(input_image, t)

    with tf.GradientTape() as tape:
        # Predict clean images from noisy images
        predicted_images = diffusion_model((noisy_images, t), training=True)

        # Compute the loss
        loss = combined_loss(original_image=input_image, noisy_images=noisy_images, reconstructed_image=predicted_images, camera_label=camera_label, classifier=classifier)

    # Apply gradients
    gradients = tape.gradient(loss, diffusion_model.trainable_variables)
    diffusion_model.optimizer.apply_gradients(zip(gradients, diffusion_model.trainable_variables))

    return {"loss": loss}

def combined_loss_new(y_true, y_pred, classifier):
    mse_loss =  tf.reduce_mean(tf.square(y_true-y_pred))

    pred_camera_model = classifier(y_pred)
    classification_loss = tf.keras.losses.categorical_crossentropy(camera_label, pred_camera_model)
    
    print(f"Classification Loss: ", classification_loss)
    # Total loss is a weighted combination of reconstruction and classification loss
    total_loss = mse_loss + tf.reduce_mean(classification_loss)
    return total_loss