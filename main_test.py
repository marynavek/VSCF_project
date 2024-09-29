import tensorflow as tf
from diffusion_model_updated import make_diffusion_model, make_classifier, save_generated_images, train_step
import numpy as np

diffusion_model_path = "/Users/marynavek/Projects/VSCF_project/models_diff/diffusion_model_epoch_1.keras"


if __name__ == "__main__":
    BATCH_SIZE = 16
    dataset_path = "/Users/marynavek/Downloads/frames/Test_data"
    image_path =  '/Users/marynavek/Projects/VSCF_project/images_diff'

    trained_diffuser = tf.keras.models.load_model(diffusion_model_path)


    save_generated_images(dataset_path, trained_diffuser, 0)