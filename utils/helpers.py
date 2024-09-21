import cv2
import numpy as np
from keras.models import load_model  # type: ignore
from glob import glob
import random
from matplotlib import pyplot as plt
from pathlib import Path


def apply_cfa(image: np.ndarray) -> np.ndarray:
    """Applies a Bayer filter to an image. (Currently Not Used)

    Args:
        image (np.ndarray): Takes in an OpenCV image in BGR format

    Returns:
        _type_: OpenCV Image of the same dimensions also in BGR format with the Bayer filter applied
    """
    # resize before interpolation to keep dimensions the same
    image = cv2.resize(image, None, fx=3, fy=3)
    # interpolate to get the 'cubey' effect with bayer filter
    image = cv2.resize(image, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_NEAREST)
    # G R B G bayer pattern
    mask_r = np.zeros_like(image)
    mask_g = np.zeros_like(image)
    mask_b = np.zeros_like(image)
    mask_r[::2, 1::2, 0] = 1  # Red channel mask
    mask_g[::2, ::2, 1] = 1  # Green channel mask (even rows, even columns)
    mask_g[1::2, 1::2, 1] = 1  # Green channel mask (odd rows, odd columns)
    mask_b[1::2, ::2, 2] = 1  # Blue channel mask

    # Apply masks to original image
    cfa_image = np.zeros_like(image)
    cfa_image += mask_r * image
    cfa_image += mask_g * image
    cfa_image += mask_b * image

    return cfa_image

def display_samples(
    model_path,
    data_path,
    folder: str = "Validation",
    image_path: str = None,
    save_path: str = None,
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Display original and generated images side by side as a pyplot figure.

    Args:
        model_path (str or keras.Model): Path to the trained model or the model object itself.
        data_path (str): Path to the dataset main folder for grabbing a random image.
        folder (str, optional): The folder to use for the image. Defaults to "Testing". Options are "Training", "Validation", and "Testing".
        image_path (str, optional): Path to the input image. If not provided, a random image from the validation dataset will be used.
        save_path (str, optional): Path to save the generated plot. If not provided, the plot will not be saved.
        show (bool, optional): Whether to display the images using matplotlib. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the original image and the generated image as numpy arrays.
    """

    # Load the model
    if image_path is None:
        image_paths = glob(f"{data_path}**/{folder}/**/*.jpg")
        random.shuffle(image_paths)
        image_path = random.choice(image_paths)
        print(f"Using random image: {image_path}")
    if type(model_path) == str:
        model = load_model(model_path)
    else:
        model = model_path

    # Load and prepare the image
    img = cv2.imread(image_path)
    orig = img.copy()
    flip = False
    if img.shape[0] == 1080 and img.shape[1] == 1920:
        img = np.transpose(img, (1, 0, 2))
        # flip = True
    height, width = img.shape[:2]
    img = cv2.resize(img, (width//3, height//3))

    # preprocess the image
    img = (img - 127.5) / 127.5
    img = np.expand_dims(img, axis=0)

    label = _get_label(data_path, image_path)
    output = model.predict(img, verbose=0)
    output = (output * 127.5) + 127.5
    output = output[0]  # remove batch dimension
    if flip:
        output = np.transpose(output, (1, 0, 2))
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # display original and output images side by side using matplotlib
    if show or save_path:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(orig)
        class_names = _get_class_names(data_path)
        class_index = np.argmax(label)
        class_name = class_names[class_index]
        axes[0].set_title(f"Original Image: {class_name}")
        axes[0].axis("off")
        axes[1].imshow(output)
        axes[1].set_title("Generated Image")
        axes[1].axis("off")
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()
    return orig, output, label

def _get_label(data_path, image_path: str) -> np.ndarray:
    """
    Get the label for an image based on its path.

    Args:
        image_path (str): The path of the image.

    Returns:
        np.ndarray: An array representing the one-hot encoded label of the image.
    """
    devices = _get_class_names(data_path)
    label = np.zeros(len(devices), dtype=int)
    for i, device in enumerate(devices):
        if device in str(image_path):
            label[i] = 1
    return np.array([label])

def _get_class_names(data_path) -> list[str]:
    """
    Get the class names from the dataset.

    Args:
        data_path (str): The path to the dataset.

    Returns:
        list[str]: A list of class names.
    """
    devices = sorted(np.array(glob(f"{data_path}*")))
    # get basename
    devices = [Path(device).name for device in devices]
    return devices
