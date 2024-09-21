from keras.callbacks import TensorBoard  # type: ignore
from keras import layers  # type: ignore
from keras.models import Model  # type: ignore


class Discriminator:
    def __init__(self, input_shape):
        self.model = None
        self.model_name = None
        self.input_width, self.input_height, self.input_channels = input_shape

    def __generate_model_name(self) -> str:
            """
            Generates a model name for the discriminator.

            Returns:
                str: The generated model name.
            """
            model_name = f"Discriminator"

            return model_name


    def create_model(self, model_name=None):
        """
        Creates and returns a discriminator model. 
        
        - The model is a simple discriminator model
        - no batch normalization is used
        - output is flattened followed by sigmoid activation to get a prediction

        Returns:
            Model: The discriminator model.
        """
        # Define the shape of the input
        shape = (self.input_width, self.input_height, self.input_channels)
        disc_input = layers.Input(shape=shape)

        # First convolutional layer
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(disc_input)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.25)(x)

        # Second convolutional layer
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.25)(x)

        # Final convolutional layer
        x = layers.Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)

        # Output layer
        x = layers.Dense(1)(x)
        x = layers.Activation("sigmoid")(x)

        output_layer = x

        # Generate the model name
        self.model_name = self.__generate_model_name()

        # Create the model
        model = Model(disc_input, output_layer, name=self.model_name)

        # Set the model
        self.model = model

        # Return the model
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
