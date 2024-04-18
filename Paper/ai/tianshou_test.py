
import argparse
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou_train import _get_agents, get_args
from zoo_argos.envs.argos_env import ArgosEnv
from zoo_argos.envs.argos_foraging_env import ArgosForagingEnv


def get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(ArgosForagingEnv(render_mode='human', verbose=False))

# ======== a test function that tests a pre-trained agent ======
def watch(args: argparse.Namespace = get_args(watch = True),) -> None:
    env = DummyVectorEnv([lambda: get_env()])
    policy, optim, agents = _get_agents(args=args)
    policy.eval()
    policy.policies[agents[0]].set_eps(args.eps_test)
    # policy.policies[agents[0]].set_eps(args.eps_test) ?
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")
    env.close()


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    watch()