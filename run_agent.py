
import argparse
import os

import numpy as np
import pygame
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
# from train import TorchMaskedActions

# from pettingzoo.classic import leduc_holdem_v4
import foraging_env
import very_simple_env
# parser = argparse.ArgumentParser(
#     description="Render pretrained policy loaded from checkpoint"
# )
# parser.add_argument(
#     "--checkpoint-path",
#     help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
# )

# args = parser.parse_args()

# checkpoint_path = os.path.expanduser(args.checkpoint_path)
checkpoint_path = os.path.expanduser('~/ray_results/DQN/DQN_very_simple_env_79f07_00000_0_2023-05-28_19-12-43/checkpoint_000300')
checkpoint_path = os.path.expanduser('~/ray_results/DQN/DQN_foraging_env_41ec5_00000_0_2023-05-28_20-08-25/checkpoint_000400')
checkpoint_path = os.path.expanduser('~/ray_results/DQN/DQN_foraging_env_06c54_00000_0_2023-05-28_21-32-40/checkpoint_000200')
checkpoint_path = os.path.expanduser('~/ray_results/DQN/DQN_foraging_env_91d01_00000_0_2023-05-28_21-43-42/checkpoint_000200')

alg_name = "DQN"
# ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
# function that outputs the environment you wish to register.


def env_creator():
    env = foraging_env.env(render_mode='human')
    return env


env = env_creator()
env_name = "foraging_env"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))



ray.init()
DQNAgent = Algorithm.from_checkpoint(checkpoint_path)

reward_sums = {a: 0 for a in env.possible_agents}
i = 0
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    #obs = observation["observation"]
    
    if termination or truncation:
        reward_sums[agent] += reward
        action = None
    else:
        print(DQNAgent.get_policy(agent))
        policy = DQNAgent.get_policy(agent)
        # batch_obs = {
        #     "obs": {
        #         "observation": np.expand_dims(observation["observation"], 0),
        #         "action_mask": np.expand_dims(observation["action_mask"], 0),
        #     }
        # }
        batch_obs = {
            "obs": np.expand_dims(observation, 0)
        }
        batched_action, state_out, info = policy.compute_actions_from_input_dict(
            batch_obs
        )
        single_action = batched_action[0]
        action = single_action
        # print('Observation shape', observation.shape, env.observation_space(agent))
        print('Agent observation', agent, observation)
        print('Agent reward', agent, reward)
        print('Agent action', action)

    env.step(action)
    i += 1
    # env.render()

print("rewards:")
print(reward_sums)
env.close()