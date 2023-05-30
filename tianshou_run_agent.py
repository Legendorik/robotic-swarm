"""This is a full example of using Tianshou with MARL to train agents, complete with argument parsing (CLI) and logging.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou_train import get_agents
import very_simple_env
import foraging_env

def get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(foraging_env.env(render_mode='human'))

# ======== a test function that tests a pre-trained agent ======
def watch() -> None:
    env = DummyVectorEnv([lambda: get_env()])
    policy, optim, agents = get_agents(loadTrainedModel=True)
    policy.eval()
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=0.1)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    
    watch()