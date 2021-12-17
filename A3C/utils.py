"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
import time

from scores.score_logger import ScoreLogger

score_logger = ScoreLogger("QuadrotorPlusHoverEnv-v0")


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.0  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    G = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_  # q learning
        G.append(v_s_)
    G.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64)
        if ba[0].dtype == np.int64
        else v_wrap(np.vstack(ba)),
        v_wrap(np.array(G)[:, None]),
    )

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


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
