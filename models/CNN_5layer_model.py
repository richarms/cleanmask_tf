from base.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # network architecture
        xp = tf.layers.conv2d(self.x, filters=16, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(xp, filters=16, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=16, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.relu)
        x = x + xp

        x = tf.layers.batch_normalization(x)
        x = tf.layers.conv2d(x, filters=16, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.relu)
        x = tf.layers.dropout(x, self.drop_out)

        x_out = tf.layers.conv2d(x, filters=1, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.relu)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=x_out))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(x_out, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
