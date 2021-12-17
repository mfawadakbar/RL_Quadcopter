"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import gym
import gym_multirotor  # must keep this!
from gym import wrappers

import time
import matplotlib.pyplot as plt

from shared_adam import SharedAdam
from utils import v_wrap, set_init, push_and_pull, record

# os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITERATION = 5
GAMMA = 0.9
MAX_EPISODE = 1000

environment = gym.make("QuadrotorPlusHoverEnv-v0")
observation_space = environment.observation_space.shape[0]
action_space = environment.action_space.shape[0]

environment = wrappers.Monitor(
    environment,  # environment to watch
    f"./videos/",  # where to save videos
    force=True,  # clear old videos
    video_callable=lambda episode_id: episode_id % 2 == 0,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.actor_input = nn.Linear(observation_space, 128)
        self.actor_output = nn.Linear(128, action_space)

        self.critic_input = nn.Linear(observation_space, 128)
        self.critic_output = nn.Linear(128, 1)

        set_init(
            [self.actor_input, self.actor_output, self.critic_input, self.critic_output]
        )
        self.distribution = torch.distributions.Categorical

    def forward(self, inputs):
        hidden_state = torch.tanh(self.actor_input(inputs))
        policy = self.actor_output(hidden_state)

        hidden_state = torch.tanh(self.critic_input(inputs))
        values = self.critic_output(hidden_state)

        return policy, values

    def choose_action(self, state):
        self.train(False)
        logits, _ = self.forward(state)
        probabilities = F.softmax(logits, dim=1).data
        distribution_probabilities = self.distribution(probabilities)
        distribution_sample = distribution_probabilities.sample()

        # torch tensor -> numpy array
        return distribution_sample.numpy()[0]

    # Compute advantage
    def loss_func(self, state, action, values_target):
        self.train(True)
        logits, values = self.forward(state)
        error = values_target - values
        value_error = error.pow(2)

        probabilities = F.softmax(logits, dim=1)
        distribution_probabilities = self.distribution(probabilities)
        expected_value = (
            distribution_probabilities.log_prob(action) * error.detach().squeeze()
        )
        policy_error = -expected_value

        return (value_error + policy_error).mean()


class Worker(mp.Process):
    def __init__(
        self,
        global_network,
        global_optimizer,
        global_episode,
        global_episode_reward,
        response_queue,
        name,
        env,
    ):
        super(Worker, self).__init__()
        self.name = f"W{name:02}"
        self.global_episode, self.global_episode_reward, self.response_queue = (
            global_episode,
            global_episode_reward,
            response_queue,
        )
        self.global_network, self.global_optimizer = global_network, global_optimizer
        self.local_network = Net()
        self.env = env

    def run(self):
        total_step = 1
        while self.global_episode.value < MAX_EPISODE:
            state = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.0

            while True:
                action = self.local_network.choose_action(v_wrap(state[None, :]))
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                episode_reward += reward

                buffer_action.append(action)
                buffer_state.append(state)
                buffer_reward.append(reward)

                if (
                    total_step % UPDATE_GLOBAL_ITERATION == 0 or done
                ):  # update global and assign to local net
                    # sync
                    push_and_pull(
                        self.global_optimizer,
                        self.local_network,
                        self.global_network,
                        done,
                        next_state,
                        buffer_state,
                        buffer_action,
                        buffer_reward,
                        GAMMA,
                    )
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:  # done and print information
                        record(
                            self.global_episode,
                            self.global_episode_reward,
                            episode_reward,
                            self.response_queue,
                            self.name,
                        )
                        break
                state = next_state
                total_step += 1
        self.response_queue.put(None)


def parallel_training(global_ep, global_ep_r, res_queue):
    workers = [
        Worker(
            global_network,
            global_optimizer,
            global_ep,
            global_ep_r,
            res_queue,
            i,
            environment,
        )
        for i in range(mp.cpu_count())
    ]
    [w.start() for w in workers]
    rewards = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            rewards.append(r)
        else:
            break
    [w.join() for w in workers]

    return rewards


def plot(rewards):
    plt.plot(rewards)
    plt.ylabel("Moving average ep reward")
    plt.xlabel("Step")
    plt.savefig(f"./images/a3c_quadcopter_learning_{time.time()}.png")


def test(gnet):
    gnet.load_state_dict(torch.load("./models/a3c_quadcopter_model.pkl"))
    gnet.eval()

    for _ in range(5):
        s = environment.reset()
        while True:
            environment.render()
            a = gnet.choose_action(v_wrap(s[None, :]))
            s_, r, done, _ = environment.step(a)
            s = s_
            if done:
                break


if __name__ == "__main__":
    global_network = Net()
    global_network.share_memory()
    global_optimizer = SharedAdam(
        global_network.parameters(), lr=1e-4, betas=(0.92, 0.999)
    )
    global_episode, global_episode_reward, response_queue = (
        mp.Value("i", 0),
        mp.Value("d", 0.0),
        mp.Queue(),
    )

    rewards = parallel_training(global_episode, global_episode_reward, response_queue)

    torch.save(global_network.state_dict(), "./models/a3c_quadcopter_model.pkl")

    plot(rewards)
    test(global_network)

    environment.close()
"""
# -*- coding: utf-8 -*-
import random

import random
import gym
import gym_multirotor  # must keep this!
from gym import wrappers

import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf
from scores.score_logger import ScoreLogger

EPISODES = 20
ENV_NAME = "QuadrotorPlusHoverEnv-v0"


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (
            K.abs(error) - clip_delta
        )

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make("QuadrotorPlusHoverEnv-v0")
    score_logger = ScoreLogger(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32
    print("Double DQN Training")
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            print("next_state", next_state)
            # reward = reward if not done else -10

            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (
                env.theta_threshold_radians - abs(theta)
            ) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(
                    "episode: {}/{}, score: {}, e: {:.2}".format(
                        e, EPISODES, time, agent.epsilon
                    )
                )
                score_logger.add_score(step, run)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
