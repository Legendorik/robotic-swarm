
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou_train import _get_agents
from zoo_argos.envs.argos_env import ArgosEnv


def get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(ArgosEnv(render_mode='human'))

# ======== a test function that tests a pre-trained agent ======
def watch() -> None:
    env = DummyVectorEnv([lambda: get_env()])
    policy, optim, agents = _get_agents(loadTrainedModel=True)
    policy.eval()
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=0.05)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    watch()