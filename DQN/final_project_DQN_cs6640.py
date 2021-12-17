from decimal import HAVE_THREADS
import random
from sys import hexversion
import gym
import gym_multirotor  # must keep this!
from gym import wrappers

import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import time
import matplotlib.pyplot as plt

from scores.score_logger import ScoreLogger

ENV_NAME = "QuadrotorPlusHoverEnv-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 5000
BATCH_SIZE = 64

EXPLORATION_MAX = 0.75
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.995

# gym_multirotor.UAVBaseEnv.desired_position = np.array([10.0, 10.0, 10.0])


class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = 5
        self.actions = [0.0, 0.2, 0.4, 0.6, 0.8]
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Input(shape=observation_space))
        self.model.add(Dense(32, activation="tanh"))
        self.model.add(Dense(32, activation="tanh"))
        self.model.add(Dense(5, activation="linear"))
        self.model.compile(loss="huber", optimizer=Adam(learning_rate=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_ = np.copy(state)

        if np.random.rand() < self.exploration_rate:
            index = random.randrange(self.action_space)
        else:
            q_values = self.model.predict(state_)[0]
            index = np.argmax(q_values)

        return index

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = reward + GAMMA * np.amax(self.model.predict(state_next)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)

    def end_epoch(self):
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def test(dqn_solver, env):
    dqn_solver.model = tf.keras.models.load_model("./DQN/model")

    for _ in range(5):
        s = env.reset()
        s = np.reshape(s, [1, env.observation_space.shape[0]])

        while True:
            env.render()
            a = dqn_solver.act(s)
            s_, _, done, _ = env.step(a)
            s_ = np.reshape(s_, [1, env.observation_space.shape[0]])
            s = s_

            if done:
                break


def main():
    env = gym.make(ENV_NAME)

    env = wrappers.Monitor(
        env,  # environment to watch
        f"./DQN/videos",  # where to save videos
        force=True,  # clear old videos
        video_callable=lambda episode_id: episode_id % 100 == 0,
    )
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    start = time.time()

    while run <= 200:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        total_reward = 0.0

        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            shaped_action = dqn_solver.actions[action]
            state_next, reward, terminal, info = env.step(shaped_action)
            # print("Next state: ", state_next)
            total_reward += reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next

            if terminal:
                with open("./DQN/learning.txt", "a+") as f:
                    f.write(
                        f"Run: {run}, exploration: {dqn_solver.exploration_rate}, score: {total_reward}, time: {time.time() - start}\n"
                    )
                score_logger.add_score(step, run)
                dqn_solver.experience_replay()
                dqn_solver.end_epoch()
                break

        print(
            f"Run: {run}, exploration: {dqn_solver.exploration_rate}, score: {total_reward}, time: {time.time() - start}"
        )

    dqn_solver.model.save("./DQN/model")
    test(dqn_solver, env)


if __name__ == "__main__":
    main()
