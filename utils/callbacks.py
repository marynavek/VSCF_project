from glob import glob
from pathlib import Path
from keras.callbacks import Callback
import os, random
from keras.models import Model
from utils.helpers import display_samples
import os


class GANMonitor(Callback):
    """
    Callback for monitoring and saving generated images during training.
    Shows the progression of the same generated image after each epoch.
    """

    def __init__(self, data_path: str, save_path: str, num_img:int =None, folder: str = "Validation"):
        # device_paths = sorted(list(glob(f"{data_path}*")))
        data_path=Path(data_path)
        device_paths = sorted(list(data_path.glob("*")))
        print(f"data path: {data_path}")
        print(f"Device paths: {device_paths}")
        if len(device_paths) == 0:
            raise ValueError(f"No devices found in {data_path}")
        
        self.num_img = num_img if num_img else len(device_paths)
        # Create save path if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.data_path = data_path
        imgs = []

        print(f"Number of images: {self.num_img}")
        for i in range(self.num_img):
            print(f"Number of device paths: {len(device_paths)}")
            if i < len(device_paths):
                print("Hello")
                device_path = device_paths[i]
                print(device_path)
                # image_paths = list(glob(f"{device_path}/{folder}/**/*.jpg"))
                image_paths= list(device_path.glob(f"**/{folder}/**/*.jpg"))

            # else:
            #     image_paths = list(glob(f"{device_path}/{folder}/**/*.jpg"))

            img = random.choice(image_paths)
            imgs.append(img)
        self.images = imgs
        print(f"Using images: {[os.path.basename(img) for img in self.images]}")


    def on_epoch_end(self, epoch: int, logs=None):
        """
        Callback function called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary containing the training metrics for the current epoch. Defaults to None.
        """
        for i in range(self.num_img):
            display_samples(
                model_path=self.model.generator,
                data_path=self.data_path,
                save_path=f"{self.save_path}/epoch_{epoch}_img{i}.png",
                image_path=self.images[i],
                show=False,
            )

class ModelSaveCallback(Callback): 
    """
    Callback to save the generator and discriminator models at the end of each epoch.

    Args:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        save_path (str): The directory path to save the models.

    Methods:
        on_epoch_end(epoch, logs=None):
            Saves the generator and discriminator models at the end of each epoch.

    Example:
        generator = create_generator_model()
        discriminator = create_discriminator_model()
        save_path = "/path/to/save/models"
        callback = ModelSaveCallback(generator, discriminator, save_path)
        model.fit(x_train, y_train, callbacks=[callback])
    """

    def __init__(self, generator: Model, discriminator: Model, save_path: str):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        # Create save path if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        """
        Saves the generator and discriminator models at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training metrics for the current epoch.

        Returns:
            None
        """
        if self.generator is not None:
            self.generator.save(f"{self.save_path}/generator_epoch_{epoch}.keras")
        if self.discriminator is not None:
            self.discriminator.save(f"{self.save_path}/discriminator_epoch_{epoch}.keras")