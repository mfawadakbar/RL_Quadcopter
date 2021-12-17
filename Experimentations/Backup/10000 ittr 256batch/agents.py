from neural_networks import build_nn, compute_loss

import tensorflow as tf
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

eps = np.finfo(np.float32).eps.item()


class ActorCritic(object):
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.shape[0]

        # Learning updates
        self.learning_rate = 3e-4
        self.discount = 0.995

        # Exploration
        self.min_exploration_rate = 0.0005
        self.exploration_rate = 0.900
        self.exploration_decay = 0.95

        # Custom neural network; every X iterations update the target
        self.model = build_nn(self.observation_space, self.learning_rate)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=1.0
        )

        # Batching update
        self.batch_size = 256

        # Memory replay
        self.memories = deque(maxlen=2500)

    def act(self, state):
        action1 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        action2 = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        if np.random.random() < self.exploration_rate:
            actions = np.zeros(shape=(self.action_space,))
            for i in range(4):
                actions[i] = np.random.choice(action1, 1)[0]
            for i in range(4, self.action_space):
                actions[i] = np.random.choice(action2, 1)[0]
            return actions

        else:
            probabilities = self.model(state)

            actions = np.zeros(shape=(self.action_space,))
            actions[0] = np.random.choice(action1, p=np.squeeze(probabilities[0]))
            actions[1] = np.random.choice(action1, p=np.squeeze(probabilities[1]))
            actions[2] = np.random.choice(action1, p=np.squeeze(probabilities[2]))
            actions[3] = np.random.choice(action1, p=np.squeeze(probabilities[3]))

            actions[4] = np.random.choice(action2, p=np.squeeze(probabilities[4]))
            actions[5] = np.random.choice(action2, p=np.squeeze(probabilities[5]))
            actions[6] = np.random.choice(action2, p=np.squeeze(probabilities[6]))
            actions[7] = np.random.choice(action2, p=np.squeeze(probabilities[7]))

            return actions

    def learn(self, state, next_state, action, reward, done, *_):
        self.memories.append([state, next_state, action, reward, done])

        if done:
            self.memory_replay()

    # Perform a TD update on every memory
    def memory_replay(self):
        n = min(self.batch_size, len(self.memories))
        states, next_states, _, rewards, _ = self.batch_memories(n)

        with tf.GradientTape() as tape:

            *action_probs, values = self.model(states)
            returns = rewards + 0.995 * self.model(next_states)[-1]
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)

            loss = compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def batch_memories(self, n):
        np.random.shuffle(self.memories)

        states = np.empty(shape=(n, self.observation_space), dtype=float)
        next_states = np.empty(shape=(n, self.observation_space), dtype=float)
        actions = np.empty(shape=(n, self.action_space), dtype=float)
        rewards = np.empty(shape=(n, 1), dtype=int)
        dones = np.empty(shape=(n, 1), dtype=bool)

        for index in range(n):
            state, next_state, action, reward, done = self.memories[index]

            states[index] = state
            next_states[index] = next_state
            actions[index] = action
            rewards[index] = reward
            dones[index] = done

        return states, next_states, actions, rewards, dones

    def td_update(self, actions, rewards, index, state_qualities, next_state_qualities):
        action = actions[index]
        update = rewards[index] + self.discount * np.amax(next_state_qualities[index])
        state_qualities[index][action] = update

    # Exploration decay and plotting
    def finish_iteration(self, iteration):
        # Exploration rate adjustments
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

    def save_model(self, file_path):
        tf.keras.models.save_model(self.model, file_path)

    def test_model()