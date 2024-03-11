import os
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import NoisyLinear
import very_simple_env
import foraging_env
def noisy_linear(x, y):
    return NoisyLinear(x, y, 0.1)

def get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    loadTrainedModel = False,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[512, 256, 256, 128],
            softmax=True,
            num_atoms=51,
            dueling_param=({
                "linear_layer": noisy_linear
            }, {
                "linear_layer": noisy_linear
            }),
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = RainbowPolicy(model=net, 
                                    optim=optim,
                                    discount_factor=0.9,
                                    estimation_step=3,
                                    target_update_freq=320,
                                    ).to("cuda" if torch.cuda.is_available() else "cpu")
        # agent_learn = DQNPolicy(
        #     model=net,
        #     optim=optim,
        #     discount_factor=0.9,
        #     estimation_step=3,
        #     target_update_freq=320,
        # )
        if (loadTrainedModel):
            ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            agent_learn.load_state_dict(torch.load('./log/foraging/rainbow/policy_1.pth'))

    if agent_opponent is None:
        # agent_opponent = DQNPolicy(
        #     model=net,
        #     optim=optim,
        #     discount_factor=0.9,
        #     estimation_step=3,
        #     target_update_freq=320,
        # )
        agent_opponent = RainbowPolicy(model=net, 
                                       optim=optim,
                                       discount_factor=0.9,
                                       estimation_step=3,
                                       target_update_freq=320,
                                       ).to("cuda" if torch.cuda.is_available() else "cpu")
        if (loadTrainedModel):
            agent_opponent.load_state_dict(torch.load('./log/foraging/rainbow/policy_2.pth'))

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(foraging_env.env())


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

        # ======== tensorboard logging setup =========
    log_path = os.path.join('./log', "foraging", "rainbow", "tensorboard")
    writer = SummaryWriter(log_path)
    writer.add_text("Parameters", 'hidden layers = [512, 256, 256, 128]')
    logger = TensorboardLogger(writer)

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path1 = os.path.join("log", "foraging", "rainbow", "policy_1.pth")
        model_save_path2 = os.path.join("log", "foraging", "rainbow", "policy_2.pth")
        os.makedirs(os.path.join("log", "foraging", "rainbow"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path1)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path2)

    def stop_fn(mean_rewards):
        return mean_rewards >= 600

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.1)
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.05)
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 0]

    # ======== Step 5: Run the trainer =========
    
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=450,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")