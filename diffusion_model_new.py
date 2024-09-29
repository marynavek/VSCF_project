import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# --- Hyperparameters ---
IMG_SIZE = 64
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_DIFFUSION_STEPS = 1000

# --- Helpers ---

def add_noise(images, noise_level):
    """Add Gaussian noise to the image."""
    noise = tf.random.normal(shape=tf.shape(images))
    noisy_images = images + noise_level * noise
    return noisy_images

def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    """Linear schedule for beta values in diffusion process."""
    return np.linspace(start, end, timesteps)

def get_loss(noisy_images, clean_images, model_output):
    """Mean squared error loss between the predicted and true image."""
    return tf.reduce_mean(tf.square(model_output - clean_images))

# --- U-Net Architecture for diffusion model ---

def build_unet():
    """Builds a simple U-Net for image-to-image operation."""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up1 = layers.UpSampling2D(size=(2, 2))(pool2)
    concat1 = layers.Concatenate()([up1, conv2])
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up2 = layers.UpSampling2D(size=(2, 2))(conv3)
    concat2 = layers.Concatenate()([up2, conv1])
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    outputs = layers.Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(conv4)

    return tf.keras.Model(inputs, outputs)

# --- Diffusion Model ---
class DiffusionModel(tf.keras.Model):
    def __init__(self, unet, timesteps=NUM_DIFFUSION_STEPS):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.beta = linear_beta_schedule(timesteps)
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

    def train_step(self, data):
        clean_images = data[0]
        batch_size = tf.shape(clean_images)[0]

        # Randomly sample time steps
        
        t = tf.random.uniform([batch_size], 0, self.timesteps, dtype=tf.int32)
        print(t.shape)
        print(clean_images.shape)
        # Add noise to the clean images
        noisy_images = self.diffusion_step(clean_images, t)

        with tf.GradientTape() as tape:
            # Predict clean images from noisy images
            predicted_images = self((noisy_images, t), training=True)

            # Compute the loss
            loss = get_loss(noisy_images, clean_images, predicted_images)

        # Apply gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

# --- Training Setup ---

def get_dataset():
    """Create a simple dataset (e.g., CIFAR-10) for image-to-image tasks."""
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# Instantiate model and optimizer
unet = build_unet()
diffusion_model = DiffusionModel(unet)
diffusion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

# Load dataset
dataset = get_dataset()

# Train the model
diffusion_model.fit(dataset, epochs=EPOCHS)
