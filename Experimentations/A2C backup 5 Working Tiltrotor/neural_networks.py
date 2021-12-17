import tensorflow as tf
from tensorflow.keras.layers import (
    AlphaDropout,
    Dense,
    Input,
)
from tensorflow.keras.models import Model


def build_nn(observation_space):
    position_input = Input(shape=(18,))
    tilt_input = Input(shape=(4,))

    a1 = Dense(
        units=64,
        activation="tanh",
    )(position_input)
    motors = Dense(
        units=64,
        activation="tanh",
    )(a1)

    a3 = Dense(units=16, activation="tanh")(tilt_input)
    tilt = Dense(units=16, activation="tanh")(tf.keras.layers.concatenate([a1, a3]))
    motor_tilt = tf.keras.layers.concatenate([tilt, motors])

    policy1 = Dense(units=5, activation="softmax")(motors)
    policy2 = Dense(units=5, activation="softmax")(motors)
    policy3 = Dense(units=5, activation="softmax")(motors)
    policy4 = Dense(units=5, activation="softmax")(motors)
    policy5 = Dense(units=9, activation="softmax")(motor_tilt)
    policy6 = Dense(units=9, activation="softmax")(motor_tilt)
    policy7 = Dense(units=9, activation="softmax")(motor_tilt)
    policy8 = Dense(units=9, activation="softmax")(motor_tilt)

    value = tf.keras.layers.concatenate([position_input, tilt_input])
    value = Dense(units=64, activation="tanh")(value)
    value = Dense(units=1)(value)

    model = Model(
        inputs=[position_input, tilt_input],
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
