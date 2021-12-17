"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
import time

from scores.score_logger import ScoreLogger

score_logger = ScoreLogger("QuadrotorPlusHoverEnv-v0")

# convert ot float32 and torch tensor
def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


# Intiitlaize the network  weights to normal distribution
def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


# Computer the discounted reward, format it using v_wrap
def push(optimizer, local_net, global_net, states, actions, rewards, discount):
    returns = []
    discounted_sum = 0.0
    for r in rewards[::-1]:  # reverse buffer r
        discounted_sum = r + discount * discounted_sum  # q learning
        returns.append(discounted_sum)
    returns.reverse()

    loss = local_net.loss_func(
        v_wrap(np.vstack(states)),
        v_wrap(np.array(actions), dtype=np.int64)
        if actions[0].dtype == np.int64
        else v_wrap(np.vstack(actions)),
        v_wrap(np.array(returns)[:, None]),
    )

    # calculate local gradients and push local parameters to global
    optimizer.zero_grad()
    loss.backward()
    for local_params, global_params in zip(
        local_net.parameters(), global_net.parameters()
    ):
        global_params._grad = local_params.grad
    optimizer.step()


def pull(local_net, global_net):
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        lp.data.copy_(gp.data)


def record(global_ep, global_ep_r, ep_r, res_queue, name, start):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.0:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)

    if name == "W01":
        with open("./A3C/learning.txt", "a+") as f:
            f.write(
                f"Run: {global_ep.value}, exploration: 0.00, score: {global_ep_r.value}, time: {time.time() - start}\n"
            )
        print(
            f"Run: {global_ep.value}, exploration: 0.00, score: {global_ep_r.value}, time: {time.time() - start}"
        )

        score_logger.add_score(int(global_ep_r.value), int(global_ep.value))
