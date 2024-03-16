
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy, DQNPolicy
from zoo_argos.envs.argos_env import ArgosEnv
import numpy as np
from gymnasium import spaces
if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = ArgosEnv()

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager(
        [
            RandomPolicy(
                # action_space=spaces.Box(np.array([0, 0]), np.array([+50, +50])),
                # observation_space=spaces.Box(0, 256),
            ), 
            RandomPolicy(
                # action_space=spaces.Box(np.array([0, 0]), np.array([+50, +50])),
                # observation_space=spaces.Box(0, 256),
            ),
        ], 
    env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.05)