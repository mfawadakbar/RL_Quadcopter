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

UPDATE_GLOBAL_ITERATION = 5000
GAMMA = 0.95
MAX_EPISODE = 5000
ENV_NAME = "QuadrotorPlusHoverEnv-v0"
environment = gym.make(ENV_NAME)  # QuadrotorPlusHoverEnv-v0
observation_space = environment.observation_space.shape[0]
action_space = environment.action_space.shape[0]

environment = wrappers.Monitor(
    environment,  # environment to watch
    f"./A3C/videos/",  # where to save videos
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

        return torch.softmax(policy), values

    def choose_action(self, state):
        self.train(False)
        logits, _ = self.forward(state)
        probabilities = logits
        distribution_probabilities = torch.distributions.Categorical(probabilities)  #
        distribution_sample = distribution_probabilities.sample()

        # torch tensor -> numpy array
        return distribution_sample.numpy()[0]

    # Compute advantage
    def loss_func(self, state, action, G):
        self.train(True)
        logits, v_s = self.forward(state)
        advantage = G - v_s
        critic_loss = advantage.pow(2)

        probabilities = logits
        distribution_probabilities = torch.distributions.Categorical(probabilities)
        entropy = -distribution_probabilities.log_prob(action)
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
        start = time.time()

        while self.global_episode.value < MAX_EPISODE:
            state = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.0

            while True:
                action = self.local_network.choose_action(v_wrap(state[None, :]))
                next_state, reward, done, _ = self.env.step(action * 0.2)
                if done:
                    reward = -1
                episode_reward += reward

                buffer_action.append(action)
                buffer_state.append(state)
                buffer_reward.append(reward)

                if total_step % UPDATE_GLOBAL_ITERATION == 0:
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

                if done:
                    record(
                        self.global_episode,
                        self.global_episode_reward,
                        episode_reward,
                        self.response_queue,
                        self.name,
                        start,
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
    plt.savefig(f"./A3C/images/a3c_quadcopter_learning_{time.time()}.png")


def test(gnet):
    gnet.load_state_dict(torch.load("./A3C/models/a3c_quadcopter_model.pkl"))
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
    torch.save(global_network.state_dict(), "./A3C/models/a3c_quadcopter_model.pkl")

    plot(rewards)
    test(global_network)

    environment.close()
