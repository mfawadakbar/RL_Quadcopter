import numpy as np
import gym
import matplotlib.pyplot as plt
import time

from scores.score_logger import ScoreLogger
from agents import ActorCritic

import gym
import gym_multirotor  # must keep this!
from gym import wrappers

import tensorflow as tf
from tqdm import tqdm


ENV_NAME = "TiltrotorPlus8DofHoverEnv-v0"
environment = gym.make(ENV_NAME)
# environment.unwrapped.desired_position = np.array([0.0, 0.0, 2.0])
environment = wrappers.Monitor(
    environment,  # environment to watch
    f"./A2C/videos/",  # where to save videos
    force=True,  # clear old videos
    video_callable=lambda episode_id: episode_id % 1000 == 0,
)

# Config
state_dim = environment.observation_space.shape[0]
n_actions = environment.action_space.shape[0]
agent = ActorCritic(environment)

episode_rewards = []
score_logger = ScoreLogger(ENV_NAME)
start = time.time()


def clear_line():
    print("\033[A                             \033[A")


def plot(rewards):
    plt.plot(rewards)
    plt.ylabel("Moving average ep reward")
    plt.xlabel("Step")
    plt.savefig(f"./A2C/images/a2c_quadcopter_learning_{time.time()}.png")


def run_episode(environment, agent, epoch):
    state = environment.reset()
    done = False
    total_reward = 0
    t = 0

    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    action0_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action1_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action2_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action3_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action4_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action5_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action6_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action7_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    while not done:
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, axis=0)

        # Run the model and to get action probabilities and critic value
        *action_logits_t, value = agent.model(state)

        # Sample next action from the action probability distribution
        action0 = tf.random.categorical(action_logits_t[0], 1)[0, 0]
        action0_probs_t = tf.nn.softmax(action_logits_t[0])
        action1 = tf.random.categorical(action_logits_t[1], 1)[0, 0]
        action1_probs_t = tf.nn.softmax(action_logits_t[1])
        action2 = tf.random.categorical(action_logits_t[2], 1)[0, 0]
        action2_probs_t = tf.nn.softmax(action_logits_t[2])
        action3 = tf.random.categorical(action_logits_t[3], 1)[0, 0]
        action3_probs_t = tf.nn.softmax(action_logits_t[3])
        action4 = tf.random.categorical(action_logits_t[4], 1)[0, 0]
        action4_probs_t = tf.nn.softmax(action_logits_t[4])
        action5 = tf.random.categorical(action_logits_t[5], 1)[0, 0]
        action5_probs_t = tf.nn.softmax(action_logits_t[5])
        action6 = tf.random.categorical(action_logits_t[6], 1)[0, 0]
        action6_probs_t = tf.nn.softmax(action_logits_t[6])
        action7 = tf.random.categorical(action_logits_t[7], 1)[0, 0]
        action7_probs_t = tf.nn.softmax(action_logits_t[7])

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probabilities of actions
        action0_probs = action0_probs.write(t, action0_probs_t[0, action0])
        action1_probs = action1_probs.write(t, action1_probs_t[0, action1])
        action2_probs = action2_probs.write(t, action2_probs_t[0, action2])
        action3_probs = action3_probs.write(t, action3_probs_t[0, action3])
        action4_probs = action4_probs.write(t, action4_probs_t[0, action4])
        action5_probs = action5_probs.write(t, action5_probs_t[0, action5])
        action6_probs = action6_probs.write(t, action6_probs_t[0, action6])
        action7_probs = action7_probs.write(t, action7_probs_t[0, action7])

        # Apply action to the environment to get next state and reward
        state, reward, done, _ = environment.step(
            [action0, action1, action2, action3, action4, action5, action6, action7]
        )

        # Store reward
        rewards = rewards.write(t, reward)
        total_reward += reward
        t += 1

        # Clear quad-copter line
        clear_line()

        # Break if done
        if tf.cast(done, tf.bool):
            break

    # Stack tensors
    rewards = rewards.stack()
    values = values.stack()
    action0_probs = action0_probs.stack()
    action1_probs = action1_probs.stack()
    action2_probs = action2_probs.stack()
    action3_probs = action3_probs.stack()
    action4_probs = action4_probs.stack()
    action5_probs = action5_probs.stack()
    action6_probs = action6_probs.stack()
    action7_probs = action7_probs.stack()

    return (
        total_reward,
        rewards,
        tf.expand_dims(values, 1),
        tf.expand_dims(action0_probs, 1),
        tf.expand_dims(action1_probs, 1),
        tf.expand_dims(action2_probs, 1),
        tf.expand_dims(action3_probs, 1),
        tf.expand_dims(action4_probs, 1),
        tf.expand_dims(action5_probs, 1),
        tf.expand_dims(action6_probs, 1),
        tf.expand_dims(action7_probs, 1),
    )


def train_step(epoch):
    with tf.GradientTape() as tape:

        total_reward, rewards, values, *actions_probs = run_episode(
            environment, agent, epoch
        )

        returns = agent.get_expected_returns(rewards, standardize=False)
        returns = tf.expand_dims(returns, 1)

        loss = agent.compute_loss(actions_probs, returns, values)

    gradients = tape.gradient(loss, agent.model.trainable_variables)
    agent.optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

    score_logger.add_score(int(total_reward), epoch)
    episode_rewards.append(total_reward)

    with open("./A2C/learning.txt", "a+") as f:
        f.write(
            f"Run: {epoch}, exploration: {0}, score: {total_reward}, time: {time.time() - start}\n"
        )

    return total_reward, loss


def main():
    for epoch in range(2_500_000):
        reward, loss = train_step(epoch)

        print(f"Epoch: {epoch}, reward: {reward:.3f}, loss: {loss:.3f}")

    if epoch % 10_000 == 0:
        agent.save_model("./A2C/models")

    plot(episode_rewards)
    environment.close()


if __name__ == "__main__":
    main()
