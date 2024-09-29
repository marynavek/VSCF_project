import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import TensorBoard

import argparse, os

from clean_diffusion_model import build_diffusion_trainable, DiffusionModel, diffusion_model_build, train_step
from dcgan import DCGAN
from diffusion_model import ExternalClassifier, save_random_image_for_each_class
from utils.datagenGAN import DataSetGeneratorGAN
from utils.datagenGAN import DataGeneratorGAN
# from models import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train a video camera source antiforensic model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data folder")
parser.add_argument("--classifier_path", type=str, required=True, help="Path to the external classifier")
parser.add_argument("--use_cpu", type=bool, default=False, help="Use CPU for training")
parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder")
parser.add_argument("--model_type", type=str, required=True, choices=['wgan', 'dcgan'], help="Type of model to train")

if __name__ == "__main__":

    EPOCHS = 20
    BATCH_SIZE = 16

    # parse arguments
    # args = parser.parse_args()
    # dataset_path = args.data_path
    # image_path = args.output_path + "/images"
    # model_path = args.output_path + "/models"
    # tensor_board_path = args.output_path + "/logs"
    # classifier_path = args.classifier_path
    # model_type = args.model_type
    # use_cpu = args.use_cpu
    dataset_path = "/Users/marynavek/Downloads/frames/Test_data"
    image_path =  '/Users/marynavek/Projects/VSCF_project/images_diff'
    diff_model_path =  "/Users/marynavek/Projects/VSCF_project/models_diff"
    use_cpu = True
    

    # # set CPU
    # if use_cpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # else:
    #     physical_devices = tf.config.list_physical_devices('GPU')
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create dataset
    dataset_maker = DataSetGeneratorGAN(input_dir_patchs=dataset_path)

    train_set = dataset_maker.create_dataset()
    print(f"Train dataset contains {len(train_set)} samples")

    num_classes = len(dataset_maker.get_class_names())

    print(num_classes)
    print(len(train_set))

    shape = (1920 // 3, 1080 // 3, 3)
    print(shape)
   

    diffusion_model = diffusion_model_build()
    print(diffusion_model.summary())
    # diffusion_model = DiffusionModel(diffusion_modules)
    classifier_model = ExternalClassifier()
    # print(classifier_model.summary())

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Create data generator
    # batch_size = 16

    data_generator = DataGeneratorGAN(train_set, num_classes, BATCH_SIZE)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        progress_bar = tqdm(total=len(data_generator), desc="Training", unit="batch")
        for input_images, camera_labels in data_generator:
            loss = train_step(diffusion_model, classifier_model, input_images, camera_labels)

            progress_bar.set_postfix(loss=loss.numpy())
            progress_bar.update(1)
        
        progress_bar.close()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}')

        # Save model after each epoch
        diffusion_model.save(os.path.join(diff_model_path, f'diffusion_model_epoch_{epoch+1}.h5'))

        # Save a random image per class after each epoch
        save_random_image_for_each_class(diffusion_model, data_generator, epoch+1, image_path)

    

