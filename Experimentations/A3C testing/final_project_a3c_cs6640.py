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
from utils import v_wrap, set_init, push, pull, record

# os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITERATION = 3
GAMMA = 0.995
MAX_EPISODE = 5000
ENV_NAME = "QuadrotorPlusHoverEnv-v0"
environment = gym.make(ENV_NAME)  # QuadrotorPlusHoverEnv-v0
observation_space = environment.observation_space.shape[0]
action_space = environment.action_space.shape[0]

environment = wrappers.Monitor(
    environment,  # environment to watch
    f"./A3C testing/videos/",  # where to save videos
    force=True,  # clear old videos
    video_callable=lambda episode_id: episode_id % (100 // 32) == 0,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.actor_input = nn.Linear(observation_space, 64)
        self.actor_hidden = nn.Linear(64, 64)
        self.actor_output = nn.Linear(64, 5)

        self.critic_input = nn.Linear(observation_space, 128)
        self.critic_output = nn.Linear(128, 1)

        set_init(
            [self.actor_input, self.actor_output, self.critic_input, self.critic_output]
        )

    def forward(self, inputs):
        hidden_state = torch.tanh(self.actor_input(inputs))
        hidden_state_2 = torch.tanh(self.actor_hidden(hidden_state))
        policy = self.actor_output(hidden_state_2)

        hidden_state = torch.tanh(self.critic_input(inputs))
        values = self.critic_output(hidden_state)

        return torch.softmax(policy, dim=1), values

    def choose_action(self, state: torch.Tensor) -> int:
        """
        1) Disable gradient calculations
        2) Forward pass to get the action probabilities
        3) Construct a continuous distribution from the discrete probabilities
        4) Sample an action from the continuous action probabilities
        5) Return the sampled action
        """

        self.train(False)

        probabilities, _ = self.forward(state)
        distribution_probabilities = torch.distributions.Categorical(probabilities)
        distribution_sample = distribution_probabilities.sample()

        return distribution_sample.numpy()[0]

    def loss_func(self, state, action, actual_rewards):
        """
        1) Enable gradient calculations
        2) Forward pass to get the action probabilities and values
        3) Compute how far off our advantage estimate is
            Actual rewards comes from the environment
            Expected rewards comes from the critic network
        4) Construct a continuous distribution from the discrete probabilities
        5) Compute the entropy of the distribution
        6) Compute the critic loss (huber error)
        7) Compute the actor loss (cross entropy)
        8) Compute and return the mean loss
        """

        self.train(True)
        probabilities, expected_rewards = self.forward(state)

        advantage = actual_rewards - expected_rewards
        distribution_probabilities = torch.distributions.Categorical(probabilities)
        entropy = -distribution_probabilities.log_prob(action)

        critic_loss = nn.HuberLoss(delta=1.0)(expected_rewards, actual_rewards)
        actor_loss = entropy * advantage.detach().squeeze()

        return (critic_loss + actor_loss).mean()


class Worker(mp.Process):
    def __init__(
        self,
        global_network,
        global_optimizer,
        global_episode,
        global_episode_reward,
        response_queue,
        worker_number,
        env,
    ):
        super(Worker, self).__init__()

        self.name = f"W{worker_number:02}"
        self.global_episode, self.global_episode_reward, self.response_queue = (
            global_episode,
            global_episode_reward,
            response_queue,
        )
        self.global_network, self.global_optimizer = global_network, global_optimizer
        self.local_network = Net()
        self.env = env

    def run(self):
        episode = 0
        start = time.time()

        while self.global_episode.value < MAX_EPISODE:
            state = self.env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0.0
            episode += 1
            done = False

            while not done:
                # Choose action and reshape and format it
                action = self.local_network.choose_action(v_wrap(state[None, :]))
                # Apply action and get new state, reward, and done
                next_state, reward, done, _ = self.env.step(action * 0.2)
                episode_reward += reward

                actions.append(action)
                states.append(state)
                rewards.append(reward)

                state = next_state

            # logging
            record(
                self.global_episode,
                self.global_episode_reward,
                episode_reward,
                self.response_queue,
                self.name,
                start,
            )

            # update global net - pull the local net parameters to the global net
            push(
                self.global_optimizer,
                self.local_network,
                self.global_network,
                states,
                actions,
                rewards,
                GAMMA,  # discount factor
            )
            states, actions, rewards = [], [], []

            # assign global net to local net - pull global net parameters to the local net
            if episode % UPDATE_GLOBAL_ITERATION == 0:
                pull(self.local_network, self.global_network)

            if self.name == "W01":
                with open("./A3C testing/learning.txt", "a+") as f:
                    f.write(f"{0}, {0}, {episode_reward}, {time.time() - start}\n")

        self.response_queue.put(None)


def parallel_training(global_episode, global_episode_reward, response_queue):
    workers = []

    for worker_number in range(mp.cpu_count()):
        worker = Worker(
            global_network,
            global_optimizer,
            global_episode,
            global_episode_reward,
            response_queue,
            worker_number,
            environment,
        )

        worker.start()
        workers.append(worker)

    rewards = []  # record episode reward to plot
    while True:
        r = response_queue.get()
        if r is not None:
            rewards.append(r)
        else:
            break

    # Wait for all worker to finish
    [w.join() for w in workers]

    return rewards


def plot(rewards):
    plt.plot(rewards)
    plt.ylabel("Moving average ep reward")
    plt.xlabel("Step")
    plt.savefig(f"./A3C testing/images/a3c_quadcopter_learning_{time.time()}.png")


def test(gnet):
    gnet.load_state_dict(torch.load("./A3C testing/models/a3c_quadcopter_model.pkl"))
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
    torch.save(
        global_network.state_dict(), "./A3C testing/models/a3c_quadcopter_model.pkl"
    )

    plot(rewards)
    test(global_network)

    environment.close()
