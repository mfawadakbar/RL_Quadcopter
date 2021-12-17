import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt

import gym
import gym_multirotor  # must keep this!
from gym import wrappers

import time
import matplotlib.pyplot as plt

from scores.score_logger import ScoreLogger

ENV_NAME = "QuadrotorPlusHoverEnv-v0"

# helper function to convert numpy arrays to tensors
def t(x):
    return torch.from_numpy(x).float()


# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.actions = [0.0, 0.2, 0.4, 0.6, 0.8]
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            nn.Softmax(dim=0),
        )

    def forward(self, X):
        return self.model(X)


# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, X):
        return self.model(X)


env = gym.make(ENV_NAME)

env = wrappers.Monitor(
    env,  # environment to watch
    f"./A2C testing/videos/",  # where to save videos
    force=True,  # clear old videos
    video_callable=lambda episode_id: episode_id % (100) == 0,
)
score_logger = ScoreLogger(ENV_NAME)

# config
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)
gamma = 0.95

episode_rewards = []

start = time.time()


def loss_func(rewards, values):
    expected_values = []
    discounted_sum = 0.0

    for r in rewards[::-1]:
        discounted_sum = r + gamma * discounted_sum
        expected_values.append(discounted_sum)
    expected_values.reverse()
    expected_values = torch.tensor(expected_values)

    actual_values = torch.stack(values)
    td_error = expected_values - actual_values

    return td_error


for i in range(5000):
    done = False
    total_reward = 0
    state = env.reset()

    rewards = []
    log_probs = []
    values = []
    states = []

    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        probs = actor(state)
        value = critic(state)

        dist = torch.distributions.Categorical(probs=probs)
        action_index = dist.sample()
        x = action_index.detach().numpy()
        action = actor.actions[x]

        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        log_probs.append(-dist.log_prob(action_index))
        values.append(value[0])
        states.append(state)

        total_reward += reward
        state = next_state

        if done:
            with open("./A2C testing/learning.txt", "a+") as f:
                f.write(
                    f"Run: {i}, exploration: {0}, score: {int(total_reward)}, time: {time.time() - start}\n"
                )
            score_logger.add_score(int(total_reward), i)
            break

    td_error = loss_func(rewards, values)

    # -log(distribution @ action_index)
    entropy = torch.stack(log_probs)
    actor_loss = (entropy * td_error).mean()
    adam_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    adam_actor.step()

    critic_loss = td_error.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward(retain_graph=True)
    adam_critic.step()

    episode_rewards.append(total_reward)
    print(
        f"Run: {i}, exploration: {0.0}, score: {total_reward}, time: {time.time() - start}"
    )

plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
plt.title("Total reward per episode (online)")
plt.ylabel("reward")
plt.xlabel("episode")
plt.show()
