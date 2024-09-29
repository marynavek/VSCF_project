import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import TensorBoard

from pathlib import Path
import argparse, os

from dcgan import DCGAN
from discriminator import Discriminator
from generator import Generator
from utils.datagenGAN import DataSetGeneratorGAN
from utils.datagenGAN import DataGeneratorGAN
from utils.callbacks import GANMonitor, ModelSaveCallback
from wgan import WGANGP
# from models import *


parser = argparse.ArgumentParser(description="Train a video camera source antiforensic model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data folder")
parser.add_argument("--classifier_path", type=str, required=True, help="Path to the external classifier")
parser.add_argument("--use_cpu", type=bool, default=False, help="Use CPU for training")
parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder")
parser.add_argument("--model_type", type=str, required=True, choices=['wgan', 'dcgan'], help="Type of model to train")

if __name__ == "__main__":

    EPOCHS = 20
    BATCH_SIZE = 12

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
    image_path =  '/Users/marynavek/Projects/VSCF_project/images'
    model_path =  "/Users/marynavek/Projects/VSCF_project/models"
    tensorboard_path = "/Users/marynavek/Projects/VSCF_project/logs"
    classifier_path = ""
    model_type = "wgan"
    use_cpu = True
    

    # set CPU
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create dataset
    dataset_maker = DataSetGeneratorGAN(input_dir_patchs=dataset_path)

    train_set = dataset_maker.create_dataset()
    print(f"Train dataset contains {len(train_set)} samples")

    num_classes = len(dataset_maker.get_class_names())

    print(num_classes)
    print(len(train_set))

    shape = (1920 // 3, 1080 // 3, 3)
    print(shape)
    # shape = (1080 // 2, 1920 // 2,3)
    # shape = (1920 // 2, 1080 // 2,3)

    # define models
    gen = Generator(shape, num_classes)
    gen.create_model()

    disc = Discriminator(shape)
    disc.create_model()

    # load pre-trained classifier
    print('Classifier Exists: ', os.path.exists(classifier_path))
    # classifier = keras.models.load_model(classifier_path)

    # create callbacks
    image_callback = GANMonitor(data_path=dataset_path, save_path=image_path)
    model_callback = ModelSaveCallback(gen.model, disc.model, model_path)
    tensorboard_callback = TensorBoard(log_dir=tensorboard_path)

    # create optimizers
    generator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    if model_type == 'wgan':
        total_steps = len(train_set) // BATCH_SIZE * EPOCHS
        # model = WGANGP(discriminator=disc.model, generator=gen.model, classifier=classifier, input_shape=shape, total_steps=total_steps)
        model = WGANGP(discriminator=disc.model, generator=gen.model, input_shape=shape, total_steps=total_steps)
        print(f"Total steps: {total_steps}")

    # elif model_type == 'dcgan':
    #     model = DCGAN(discriminator=disc.model, generator=gen.model, classifier=classifier, input_shape=shape, num_classes=num_classes, embedding_dim=50)
    # else:
    #     raise ValueError("Invalid model type")

    # compile and train

    model.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )

    

    model.fit(
        DataGeneratorGAN(train_set, num_classes, BATCH_SIZE),
        epochs=EPOCHS,
        initial_epoch=0,
        callbacks=[model_callback, tensorboard_callback, image_callback],
    )

    gen.model.save(model_path + "/final_gen.keras")
    disc.model.save(model_path + "/final_disc.keras")

    print("Training complete")

