import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
image_height = 360
image_width = 640
channels = 3
timesteps = 1000  # Number of diffusion steps
num_camera_models = 10  # 10 possible camera models

# Dummy external classifier (replace with actual classifier)
class ExternalClassifier(tf.keras.Model):
    def __init__(self):
        super(ExternalClassifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_camera_models, activation='softmax')

    def call(self, x):
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        return self.dense2(x)

# Helper function to add Gaussian noise
def add_noise(image, noise_level):
    noise = tf.random.normal(shape=image.shape, mean=0.0, stddev=noise_level)
    noisy_image = image + noise
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    return noisy_image

# Forward diffusion process
def forward_diffusion_process(image, noise_level, timesteps):
    for t in range(timesteps):
        image = add_noise(image, noise_level)
    return image

# Reverse diffusion process (reconstruction)
def reverse_diffusion_process(noisy_image, noise_level, timesteps):
    for t in range(timesteps):
        denoised_image = noisy_image - tf.random.normal(shape=noisy_image.shape, mean=0.0, stddev=noise_level)
        denoised_image = tf.clip_by_value(denoised_image, 0.0, 1.0)
    return denoised_image

# Diffusion Model (Generator)
class DiffusionModel(tf.keras.Model):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.noise_level = 0.1

    def call(self, x):
        noisy_image = forward_diffusion_process(x, self.noise_level, timesteps)
        reconstructed_image = reverse_diffusion_process(noisy_image, self.noise_level, timesteps)
        return reconstructed_image

# Combined loss function
def combined_loss(original_image, reconstructed_image, camera_label, classifier):
    # MSE Loss for image reconstruction
    mse_loss = tf.reduce_mean(tf.square(original_image - reconstructed_image))
    
    # Camera classification loss (cross-entropy)
    pred_camera_model = classifier(reconstructed_image)
    classification_loss = tf.keras.losses.sparse_categorical_crossentropy(camera_label, pred_camera_model)
    
    # Total loss is a weighted combination of reconstruction and classification loss
    total_loss = mse_loss + tf.reduce_mean(classification_loss)
    return total_loss

# Training function
def train_step(model, classifier, optimizer, input_image, camera_label):
    with tf.GradientTape() as tape:
        reconstructed_image = model(input_image)
        loss = combined_loss(input_image, reconstructed_image, camera_label, classifier)

    # Backpropagation
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Example usage
def main():
    # Initialize models
    diffusion_model = DiffusionModel()
    external_classifier = ExternalClassifier()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Dummy data: random images of shape 640x360x3, and camera labels (0 to 9)
    input_images = tf.random.uniform(shape=[32, image_height, image_width, channels], minval=0.0, maxval=1.0)
    camera_labels = tf.random.uniform(shape=[32], minval=0, maxval=num_camera_models, dtype=tf.int32)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        loss = train_step(diffusion_model, external_classifier, optimizer, input_images, camera_labels)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}')

    # Display the input, noisy, and reconstructed images
    reconstructed_images = diffusion_model(input_images)
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(input_images[i])
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Reconstructed Image")
        plt.imshow(reconstructed_images[i])
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
