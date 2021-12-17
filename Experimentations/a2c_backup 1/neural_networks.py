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
    x = Dropout(rate=0.2)(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dense(units=256, activation="relu")(x)

    motors = Dense(units=128, activation="relu")(x)
    tilt = Dense(units=128, activation="relu")(x)

    policy1 = Dense(units=6)(motors)
    policy2 = Dense(units=6)(motors)
    policy3 = Dense(units=6)(motors)
    policy4 = Dense(units=6)(motors)
    policy5 = Dense(units=11)(tilt)
    policy6 = Dense(units=11)(tilt)
    policy7 = Dense(units=11)(tilt)
    policy8 = Dense(units=11)(tilt)
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
