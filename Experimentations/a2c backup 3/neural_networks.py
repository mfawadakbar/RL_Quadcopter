import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model


def build_nn(observation_space):
    inputs = Input(shape=(observation_space))

    # "float16" provides better performance on Turing GPUs
    x = tf.cast(inputs, dtype=tf.float16)

    x = Dense(
        units=observation_space * observation_space,
        activation="relu",
    )(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dense(units=256, activation="relu")(x)

    motors = Dense(units=128, activation="relu")(x)
    tilt = Dense(units=128, activation="relu")(x)

    policy1 = Dense(units=11, activation="softmax")(motors)
    policy2 = Dense(units=11, activation="softmax")(motors)
    policy3 = Dense(units=11, activation="softmax")(motors)
    policy4 = Dense(units=11, activation="softmax")(motors)
    policy5 = Dense(units=11, activation="softmax")(tilt)
    policy6 = Dense(units=11, activation="softmax")(tilt)
    policy7 = Dense(units=11, activation="softmax")(tilt)
    policy8 = Dense(units=11, activation="softmax")(tilt)
    value = Dense(units=1)(x)

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
