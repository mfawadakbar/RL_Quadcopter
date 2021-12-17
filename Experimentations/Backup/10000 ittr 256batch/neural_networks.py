import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def build_nn(observation_space, learning_rate):
    inputs = Input(shape=(observation_space))

    # "float16" provides better performance on Turing GPUs
    x = tf.cast(inputs, dtype=tf.float16)

    x = Dense(
        units=observation_space * observation_space,
        activation="relu",
        kernel_regularizer="l2",
    )(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=1024, activation="relu", kernel_regularizer="l2")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=1024, activation="relu", kernel_regularizer="l2")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=256, activation="relu", kernel_regularizer="l2")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=256, activation="relu", kernel_regularizer="l2")(x)

    policy1 = Dense(units=6, activation="softmax")(x)
    policy2 = Dense(units=6, activation="softmax")(x)
    policy3 = Dense(units=6, activation="softmax")(x)
    policy4 = Dense(units=6, activation="softmax")(x)
    policy5 = Dense(units=11, activation="softmax")(x)
    policy6 = Dense(units=11, activation="softmax")(x)
    policy7 = Dense(units=11, activation="softmax")(x)
    policy8 = Dense(units=11, activation="softmax")(x)
    value = Dense(units=1, activation="linear")(x)

    model = Model(
        inputs=inputs,
        outputs=[
            policy1,
            policy2,
            policy3,
            policy4,
            policy5,
            policy6,
            policy7,
            policy8,
            value,
        ],
    )

    return model


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor):
    """Computes the combined actor-critic loss."""

    advantages = returns - values

    actor_loss = []
    critic_loss = []

    for index in range(len(action_probs)):
        action_prob = action_probs[index]
        advantage = advantages[index]

        actor_loss.append(tf.reduce_sum(-action_prob * advantage))

    actor_loss = tf.reduce_mean(actor_loss)
    critic_loss = huber_loss(values, returns)
    loss = actor_loss + critic_loss

    return loss
