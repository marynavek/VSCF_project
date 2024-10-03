import tensorflow as tf
from keras.models import Model
from keras import metrics
from keras.losses import categorical_crossentropy

class WGANGP(Model):
    def __init__(
        self,
        discriminator,
        generator,
        classifier,
        input_shape,
        total_steps,
        discriminator_extra_steps=3,
        gp_weight=10.0,
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
        self.classifier.summary()

        self.d_steps = discriminator_extra_steps

        self.gp_weight = gp_weight  # gradient penalty weight (discriminator)

        # weights for the losses (generator)
        self.cls_weight = cls_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight

        self.current_step = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False)
        self.total_steps = total_steps

    def compile(self, d_optimizer, g_optimizer, c_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.d_wass_loss = metrics.Mean(name="d_wasserstein_loss")
        self.d_gp = metrics.Mean(name="d_gradient_penalty")
        self.g_loss = metrics.Mean(name="g_loss")
        self.d_loss = metrics.Mean(name="d_loss")
        self.cls_loss = metrics.Mean(name="cls_loss")
        self.adv_loss = metrics.Mean(name="adv_loss")
        self.p_loss = metrics.Mean(name="perceptual_loss")

    @property
    def metrics(self):
        return [
            self.d_loss,
            self.d_wass_loss,
            self.d_gp,
            self.g_loss,
            self.cls_loss,
            self.adv_loss,
            self.p_loss,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # get discriminator output for the interpolated image
            pred = self.discriminator(interpolated, training=True)

        # calculate gradients with respect to the interpolated image
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # compute the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def classifier_loss(self, generated_images, labels):

        # shifted_labels = tf.roll(labels, shift=1, axis=1)
        cls_predictions = self.classifier(generated_images, training=False)
        cls_loss = categorical_crossentropy(labels, cls_predictions)

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

        # train discriminator
        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                # generate fake images
                fake_images = self.generator(real_images, training=True)
                # get discriminator output for real and fake images
                fake_predictions = self.discriminator(fake_images, training=True)
                real_Predictions = self.discriminator(real_images, training=True)

                # calculate wasserstein loss
                d_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                    real_Predictions
                )
                d_gp = self.gradient_penalty(batch_size, real_images, fake_images)

                d_loss = d_wass_loss + d_gp * self.gp_weight

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

            with tf.GradientTape() as tape:
                fake_images = self.generator(real_images, training=True)
                cls_loss_real = self.classifier_loss(real_images, real_labels)
                cls_loss_fake = self.classifier_loss(fake_images, real_labels)
                cls_loss = tf.reduce_mean(cls_loss_real + cls_loss_fake)

            c_gradients = tape.gradient(cls_loss, self.classifier.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradients, self.classifier.trainable_variables)
            )

        # train generator
        with tf.GradientTape() as tape:
            # calculate adversarial loss
            generated_images = self.generator(real_images, training=True)
            gen_predictions = self.discriminator(generated_images, training=True)

            adv_loss = -tf.reduce_mean(gen_predictions)

            # calculate classification loss
            # if self.current_step >= self.total_steps // 2:
            cls_loss = self.classifier_loss(generated_images, real_labels)
            # else:
                # cls_loss = 0.0

            # calculate perceptual loss
            perceptual_loss = self.perceptual_loss(real_images, generated_images)

            # add other losses to the generator loss
            g_loss = (
                (cls_loss * self.cls_weight) +
                (adv_loss * self.adv_weight)
                + (perceptual_loss * self.perceptual_weight)
            )

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        self.d_loss.update_state(d_loss)
        self.d_wass_loss.update_state(d_wass_loss)
        self.d_gp.update_state(d_gp)
        self.g_loss.update_state(g_loss)
        self.cls_loss.update_state(cls_loss)
        self.adv_loss.update_state(adv_loss)
        self.p_loss.update_state(perceptual_loss)

        self.current_step.assign_add(1)
        return {m.name: m.result() for m in self.metrics}