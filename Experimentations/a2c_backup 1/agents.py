from neural_networks import build_nn

import tensorflow as tf
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


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
        self.eps = np.finfo(np.float32).eps.item()

    def get_expected_returns(self, rewards, standardize=True):
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape

        for i in range(n):
            reward = rewards[i]
            discounted_sum = reward + self.discount * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = (returns - tf.reduce_mean(returns)) / (
                tf.math.reduce_std(returns) + self.eps
            )

        return returns

    def compute_loss(self, action_probs, returns, values):
        advantage = returns - values
        critic_loss = self.huber_loss(returns, values)

        action0_probs = action_probs[0]
        action1_probs = action_probs[1]
        action2_probs = action_probs[2]
        action3_probs = action_probs[3]
        action4_probs = action_probs[4]
        action5_probs = action_probs[5]
        action6_probs = action_probs[6]
        action7_probs = action_probs[7]

        action0_log_probs = tf.math.log(action0_probs)
        action1_log_probs = tf.math.log(action1_probs)
        action2_log_probs = tf.math.log(action2_probs)
        action3_log_probs = tf.math.log(action3_probs)
        action4_log_probs = tf.math.log(action4_probs)
        action5_log_probs = tf.math.log(action5_probs)
        action6_log_probs = tf.math.log(action6_probs)
        action7_log_probs = tf.math.log(action7_probs)

        actor0_loss = -tf.math.reduce_mean(action0_log_probs * advantage)
        actor1_loss = -tf.math.reduce_mean(action1_log_probs * advantage)
        actor2_loss = -tf.math.reduce_mean(action2_log_probs * advantage)
        actor3_loss = -tf.math.reduce_mean(action3_log_probs * advantage)
        actor4_loss = -tf.math.reduce_mean(action4_log_probs * advantage)
        actor5_loss = -tf.math.reduce_mean(action5_log_probs * advantage)
        actor6_loss = -tf.math.reduce_mean(action6_log_probs * advantage)
        actor7_loss = -tf.math.reduce_mean(action7_log_probs * advantage)

        actor_loss = tf.reduce_sum(
            [
                actor0_loss,
                actor1_loss,
                actor2_loss,
                actor3_loss,
                actor4_loss,
                actor5_loss,
                actor6_loss,
                actor7_loss,
            ]
        )

        return critic_loss + actor_loss

    def save_model(self, file_path):
        tf.keras.models.save_model(self.model, file_path)

    def test_model(self):
        pass
