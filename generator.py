import tensorflow as tf

from keras.callbacks import TensorBoard # type: ignore
from keras import layers  # type: ignore
from keras.models import Model  # type: ignore

class Generator:
    def __init__(self, input_shape: tuple, num_classes: int):
        self.model = None
        self.model_name = None
        self.input_width, self.input_height, self.input_channels = input_shape
        self.num_classes = num_classes

    def __generate_model_name(self) -> str:
            """
            Generates a model name for the generator.

            Returns:
                str: The generated model name.
            """
            model_name = f"Generator"

            return model_name

    def create_model(self) -> Model:
        """
        Creates and returns a generator model. 
        
        - The model is a simple generator model with two downsample blocks and two upsample blocks.
        - The bias is turned off for all Conv2D layers since batch normalization is used.
        - At the end there is a feature map reduction to the desired number of channels.

        Returns:
            Model: The generative model.
        """
        shape = (self.input_width, self.input_height, self.input_channels)
        gen_input = layers.Input(shape=shape)

        x = gen_input

        # Two Downsample Blocks
        x = layers.Conv2D(64, (2, 2), strides=2, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(128, (2, 2), strides=2, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # Two Upsample Blocks
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(128, (2, 2), strides=1, padding="same")(x)
        x= layers.BatchNormalization()(x)
        x= layers.LeakyReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, (2, 2), strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # Feature Map Reduction. No batch norm for the output layer of the generator
        x = layers.Conv2D(
            self.input_channels, kernel_size=(2, 2), strides=1, padding="same"
        )(x)

        # Output Layers. The output is in the range [-1, 1]
        x = layers.Activation("tanh")(x)
        x = layers.Reshape(shape)(x)

        generator_output = x

        self.model_name = self.__generate_model_name()

        model = Model(gen_input, generator_output, name=self.model_name)

        self.model = model

        return model

    def print_model_summary(self) -> None:
            """
            Prints the summary of the model.

            If the model is None, it prints an error message.

            Returns:
                None
            """
            if self.model is None:
                print("Can't print model summary, self.model is None!")
            else:
                print(f"\nSummary of model:\n{self.model.summary()}")
