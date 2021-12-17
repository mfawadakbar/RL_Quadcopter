from neural_networks import build_nn

import tensorflow as tf
import numpy as np


class ActorCritic(object):
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.shape[0]

        # Learning updates
        self.learning_rate = 3e-4
        self.discount = 0.995

        # Custom neural network
        self.model = build_nn(self.observation_space)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        # finfo is used for numerical stability, so you dont get a zero, cuz log(0) = -inf
        self.eps = np.finfo(np.float32).eps.item()  # epsilon used for log(0+epsilon)

    def save_model(self, file_path):
        tf.keras.models.save_model(self.model, file_path)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    def test_model(self):
        pass
