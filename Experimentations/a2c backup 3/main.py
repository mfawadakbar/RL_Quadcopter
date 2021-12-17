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


def train_step(epoch):
    huber = tf.keras.losses.Huber()
    state = environment.reset()
    done = False
    episode_reward = 0
    critic_value_history = []
    rewards_history = []

    action0_probs_history = []
    action1_probs_history = []
    action2_probs_history = []
    action3_probs_history = []
    action4_probs_history = []
    action5_probs_history = []
    action6_probs_history = []
    action7_probs_history = []

    with tf.GradientTape() as tape:

        while not done:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, axis=0)

            *action_probs, value = agent.model(state)
            critic_value_history.append(value[0, 0])

            action = act(
                action0_probs_history,
                action1_probs_history,
                action2_probs_history,
                action3_probs_history,
                action4_probs_history,
                action5_probs_history,
                action6_probs_history,
                action7_probs_history,
                action_probs,
            )

            state, reward, done, _ = environment.step(action)
            rewards_history.append(reward)
            episode_reward += reward
            clear_line()

            if done:
                break

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + agent.discount * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + agent.eps)
        returns = returns.tolist()

        history = zip(
            action0_probs_history,
            action1_probs_history,
            action2_probs_history,
            action3_probs_history,
            action4_probs_history,
            action5_probs_history,
            action6_probs_history,
            action7_probs_history,
            critic_value_history,
            returns,
        )
        (
            actor0_losses,
            actor1_losses,
            actor2_losses,
            actor3_losses,
            actor4_losses,
            actor5_losses,
            actor6_losses,
            actor7_losses,
        ) = ([], [], [], [], [], [], [], [])
        critic_losses = []

        for (
            log_prob1,
            log_prob2,
            log_prob3,
            log_prob4,
            log_prob5,
            log_prob6,
            log_prob7,
            log_prob8,
            value,
            G,
        ) in history:
            advantage = G - value
            actor0_losses.append(-log_prob1 * advantage)
            actor1_losses.append(-log_prob2 * advantage)
            actor2_losses.append(-log_prob3 * advantage)
            actor3_losses.append(-log_prob4 * advantage)
            actor4_losses.append(-log_prob5 * advantage)
            actor5_losses.append(-log_prob6 * advantage)
            actor6_losses.append(-log_prob7 * advantage)
            actor7_losses.append(-log_prob8 * advantage)
            critic_losses.append(huber(tf.expand_dims(value, 0), tf.expand_dims(G, 0)))

        loss_value = (
            sum(actor0_losses)
            + sum(actor1_losses)
            + sum(actor2_losses)
            + sum(actor3_losses)
            + sum(actor4_losses)
            + sum(actor5_losses)
            + sum(actor6_losses)
            + sum(actor7_losses)
            + sum(critic_losses)
        )
        gradients = tape.gradient(loss_value, agent.model.trainable_variables)
        agent.optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

    return episode_reward, loss_value


def act(
    action0_probs_history,
    action1_probs_history,
    action2_probs_history,
    action3_probs_history,
    action4_probs_history,
    action5_probs_history,
    action6_probs_history,
    action7_probs_history,
    action_probs,
):
    action0 = np.random.choice(11, p=np.squeeze(action_probs[0]))
    action1 = np.random.choice(11, p=np.squeeze(action_probs[1]))
    action2 = np.random.choice(11, p=np.squeeze(action_probs[2]))
    action3 = np.random.choice(11, p=np.squeeze(action_probs[3]))
    action4 = np.random.choice(11, p=np.squeeze(action_probs[4]))
    action5 = np.random.choice(11, p=np.squeeze(action_probs[5]))
    action6 = np.random.choice(11, p=np.squeeze(action_probs[6]))
    action7 = np.random.choice(11, p=np.squeeze(action_probs[7]))

    action0_probs_history.append(tf.math.log(action_probs[0][0, action0]))
    action1_probs_history.append(tf.math.log(action_probs[1][0, action1]))
    action2_probs_history.append(tf.math.log(action_probs[2][0, action2]))
    action3_probs_history.append(tf.math.log(action_probs[3][0, action3]))
    action4_probs_history.append(tf.math.log(action_probs[4][0, action4]))
    action5_probs_history.append(tf.math.log(action_probs[5][0, action5]))
    action6_probs_history.append(tf.math.log(action_probs[6][0, action6]))
    action7_probs_history.append(tf.math.log(action_probs[7][0, action7]))

    action = [
        action0 * 0.2 - 1.2,
        action1 * 0.2 - 1.2,
        action2 * 0.2 - 1.2,
        action3 * 0.2 - 1.2,
        action4 * 0.2 - 1.2,
        action5 * 0.2 - 1.2,
        action6 * 0.2 - 1.2,
        action7 * 0.2 - 1.2,
    ]

    return action


def main():
    for epoch in range(2_500_000):
        reward, loss = train_step(epoch)

        print(f"Epoch: {epoch}, reward: {reward:.3f}, loss: {loss:.3f}")
        score_logger.add_score(int(reward), epoch)

    if epoch % 10_000 == 0:
        agent.save_model("./A2C/models")

    plot(episode_rewards)
    environment.close()


if __name__ == "__main__":
    main()
