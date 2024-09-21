import tensorflow as tf
from keras.models import Model
from keras import metrics, layers, losses
import numpy as np


class DCGAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        classifier,
        input_shape,
        num_classes,
        embedding_dim=50,
        discriminator_extra_steps=3,
        cls_weight=1.0,  # TODO: adjust weights
        adv_weight=1.0,
        perceptual_weight=1.0,
    ):
        super().__init__()
        self.model_input_shape = input_shape

        self.discriminator = discriminator
        self.discriminator.summary()
        self.generator = generator
        self.generator.summary()
        self.classifier = classifier

        self.d_steps = discriminator_extra_steps

        # weights for the losses (generator)
        self.cls_weight = cls_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = metrics.Mean(name="d_loss")
        self.g_loss = metrics.Mean(name="g_loss")
        self.cls_loss = metrics.Mean(name="cls_loss")
        self.adv_loss = metrics.Mean(name="adv_loss")
        self.p_loss = metrics.Mean(name="perceptual_loss")

    @property
    def metrics(self):
        return [
            self.d_loss,
            self.g_loss,
            self.cls_loss,
            self.adv_loss,
            self.p_loss,
        ]

    def classifier_loss(self, generated_images, real_labels):
        cls_predictions = self.classifier(generated_images, training=False)

        cls_loss_fn = losses.CategoricalCrossentropy(label_smoothing=0.1)
        
        # Use softmax cross-entropy between generated image class and unaltered class
        cls_loss = cls_loss_fn(real_labels, cls_predictions)

        return cls_loss

    def perceptual_loss(self, img1, img2):
        return tf.reduce_mean(tf.abs(img1 - img2))

    @tf.function  # if training slow, turn this one
    def train_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            real_images, real_labels = data
        else:
            raise ValueError("Expected data format: (images, labels)")
        # get batch size
        batch_size = tf.shape(real_images)[0]
        # shift labels for untargeted attack
        real_labels = tf.roll(real_labels, shift=1, axis=1)

        # train discriminator
        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                # generate fake images
                fake_images = self.generator(real_images, training=True)
                # get discriminator output for real and fake images
                fake_predictions = self.discriminator(fake_images, training=True)
                real_predictions = self.discriminator(real_images, training=True)

                # calculate discriminator loss
                d_loss_real = tf.keras.losses.binary_crossentropy(tf.ones_like(real_predictions), real_predictions)
                d_loss_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_predictions), fake_predictions)
                d_loss = d_loss_real + d_loss_fake

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

        # train generator
        with tf.GradientTape() as tape:
            # calculate adversarial loss
            generated_images = self.generator(real_images, training=True)
            gen_predictions = self.discriminator(generated_images, training=True)

            adv_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(gen_predictions), gen_predictions)

            # calculate classification loss
            cls_loss = self.classifier_loss(generated_images, real_labels)

            # calculate perceptual loss
            perceptual_loss = self.perceptual_loss(real_images, generated_images)

            # add other losses to the generator loss
            g_loss = (
                (cls_loss * self.cls_weight)
                + (adv_loss * self.adv_weight)
                + (perceptual_loss * self.perceptual_weight)
            )

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        self.d_loss.update_state(d_loss)
        self.g_loss.update_state(g_loss)
        self.cls_loss.update_state(cls_loss)
        self.adv_loss.update_state(adv_loss)
        self.p_loss.update_state(perceptual_loss)

        return {m.name: m.result() for m in self.metrics}
