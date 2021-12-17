import numpy as np
import gym
import matplotlib.pyplot as plt
import time

from scores.score_logger import ScoreLogger
from agents import ActorCritic

import gym
import gym_multirotor  # must keep this!
from gym import wrappers


def clear_line():
    print("\033[A                             \033[A")


def plot(rewards):
    plt.plot(rewards)
    plt.ylabel("Moving average ep reward")
    plt.xlabel("Step")
    plt.savefig(f"./A2C/images/a2c_quadcopter_learning_{time.time()}.png")


ENV_NAME = "TiltrotorPlus8DofHoverEnv-v0"
environment = gym.make(ENV_NAME)
environment.unwrapped.desired_position = np.array([0.0, 0.0, 3.0])
environment = wrappers.Monitor(
    environment,  # environment to watch
    f"./A2C/videos/",  # where to save videos
    force=True,  # clear old videos
    video_callable=lambda episode_id: episode_id % 100 == 0,
)

# config
state_dim = environment.observation_space.shape[0]
n_actions = environment.action_space.shape[0]
agent = ActorCritic(environment)

episode_rewards = []
score_logger = ScoreLogger(ENV_NAME)
start = time.time()

for epoch in range(100_000):
    done = False
    total_reward = 0

    state = environment.reset()
    state = np.reshape(state, [1, state_dim])

    while not done:
        actions = agent.act(state)
        next_state, reward, done, _ = environment.step(actions)

        next_state = np.reshape(next_state, [1, state_dim])
        agent.learn(state, next_state, actions, reward, done)

        state = next_state
        total_reward += reward

        clear_line()
        if done:
            with open("./A2C/learning.txt", "a+") as f:
                f.write(
                    f"Run: {epoch}, exploration: {0}, score: {total_reward}, time: {time.time() - start}\n"
                )

            break

    agent.finish_iteration(epoch)
    score_logger.add_score(int(total_reward), epoch)
    episode_rewards.append(total_reward)

    if epoch % 1000 == 0:
        agent.save_model("./A2C/models")
agent.test_model()

plot(episode_rewards)
environment.close()
