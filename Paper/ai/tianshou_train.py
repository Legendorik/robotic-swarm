import argparse
from copy import deepcopy
import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, RainbowPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from zoo_argos.envs.argos_env import ArgosEnv
from zoo_argos.envs.argos_foraging_env import ArgosForagingEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import NoisyLinear

def get_parser(watch: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--step-per-epoch', type=int, default=1500) #100
    parser.add_argument('--step-per-collect', type=int, default=50) #10 #50
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[512, 256, 256, 128]
    )
    parser.add_argument('--training-num', type=int, default=10) #10
    parser.add_argument('--test-num', type=int, default=10) #10
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.05)
    parser.add_argument(
        '--win-condition',
        type=float,
        default=5000,
    )
    parser.add_argument(
        '--watch',
        default=watch,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser

def get_args(watch: bool = False) -> argparse.Namespace:
    parser = get_parser(watch=watch)
    return parser.parse_known_args()[0]

def noisy_linear(x, y):
    return NoisyLinear(x, y, 0.1)

def _get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    env: PettingZooEnv = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = env if env != None else _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        # net = Net(
        #     state_shape=observation_space.shape or observation_space.n,
        #     action_shape=env.action_space.shape or env.action_space.n,
        #     hidden_sizes=args.hidden_sizes,
        #     softmax=True,
        #     num_atoms=51,
        #     dueling_param=({
        #         "linear_layer": noisy_linear
        #     }, {
        #         "linear_layer": noisy_linear
        #     }),
        #     device=args.device,
        # ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            action_space=env.action_space,
            is_double=False,
            discount_factor=args.gamma,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq,
        )
        
        # agent_learn = RainbowPolicy(
        #     model=net,
        #     optim=optim,
        #     action_space=env.action_space,
        #     discount_factor=args.gamma,
        #     estimation_step=args.n_step,
        #     target_update_freq=args.target_update_freq,
        # ).to(args.device)
        # TODO: watch_path & resume_path
        # if args.resume_path:
        #     agent_learn.load_state_dict(torch.load(args.resume_path))
        if (args.watch):
            ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            agent_learn.load_state_dict(torch.load('./log/policies/policy_0_rainbow.pth'))

    if agent_opponent is None:
        # agent_opponent = RandomPolicy(action_space=env.action_space)
        # agent_opponent = deepcopy(agent_learn)
        # agent_opponent.load_state_dict(torch.load('./log/ttt/dqn/policy_0 5(actually good).pth'))
        net2 = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=args.hidden_sizes,
            softmax=True,
            num_atoms=51,
            dueling_param=({
                "linear_layer": noisy_linear
            }, {
                "linear_layer": noisy_linear
            }),
            device=args.device,
        ).to(args.device)
        optim2 = torch.optim.Adam(net2.parameters(), lr=args.lr)

        agent_opponent = RainbowPolicy(
            model=net2,
            optim=optim2,
            action_space=env.action_space,
            discount_factor=args.gamma,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq,
        ).to(args.device)
        # if args.resume_path:
        #     agent_opponent.load_state_dict(torch.load(args.resume_path))
        if (args.watch):
            agent_opponent.load_state_dict(torch.load('./log/policies/policy_0_rainbow.pth'))

    agents = [agent_learn, agent_opponent]
    policy = MultiAgentPolicyManager(policies=agents, env=env)
    env_agents = env.agents
    env.close()
    return policy, optim, env_agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    # return PettingZooEnv(ArgosEnv(render_mode='no_render'))
    return PettingZooEnv(ArgosForagingEnv(render_mode='no_render'))


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([_get_env for _ in range(args.test_num)])

    # seed
    seed = 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num) 

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, 'tic_tac_toe', 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        if hasattr(args, 'model_save_path'):
            model_save_path1 = args.model_save_path
            model_save_path2 = args.model_save_path
        else: 
            model_save_path1 = os.path.join("log", "ttt", "dqn", "policy_0.pth")
            model_save_path2 = os.path.join("log", "ttt", "dqn", "policy_1.pth")
        # os.makedirs(model_save_path, exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path1)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path2)

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_condition

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(args.eps_train)
        policy.policies[agents[1]].set_eps(args.eps_train)
        #set eps for all policies

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(args.eps_test)
        policy.policies[agents[1]].set_eps(args.eps_test)
        #set eps for all policies

    def reward_metric(rews):
        # print('REWARD FROM REWARD METRIC:', rews[:, 0])
        return rews[:, 0]

    # ======== Step 5: Run the trainer =========
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    ).run()

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[0]])")
    train_envs.close()
    test_envs.close()