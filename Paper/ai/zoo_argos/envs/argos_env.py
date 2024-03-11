import gymnasium as gym
import numpy as np
from gymnasium import spaces
from argos import argos_runner, argos_io
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import functools

class ArgosEnv(AECEnv):
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self):
        # initialize and run argos
        self.num_robots = 2
        self.argos = None
        self.argos_io = None

        self.possible_agents = ["robot_" + str(r) for r in range(self.num_robots)]
        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
                
        self._action_spaces = {
            agent: spaces.Box(np.array([0, 0]), np.array([+50, +50])) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(0, 256) for agent in self.possible_agents
        }

        self.render_mode = 'human'

        # self.action_space = spaces.Box(np.array([0, 0]), np.array([+50, +50]))
        # self.observation_space = spaces.Box(0, 256)


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self._observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
        # return Discrete(4)
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        self.argos.kill()
    
    def render(self):
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        
        if (self.argos is None):
            self.argos = argos_runner.Argos()
            self.argos_io = argos_io.ArgosIO(self.num_robots, verbose=False)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}

        self.observations = {agent: [0] for agent in self.agents}

        self.game_over = False

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        agent = self.agent_selection
        agent_id = self.agent_name_mapping[agent]
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            if (self.agent_selection == agent):
                self.agent_selection = self._agent_selector.next()
            return 



        # stores action of current agent
        self.state[self.agent_selection] = action

        # in messages
        in_msg = self.argos_io.receive_from(agent_id)
        in_msg = in_msg.split(";")
        # get floor color
        floor = float(in_msg[2].replace('\x00', '').replace(',', '.'))
        self.rewards[agent] = (floor - 128) / 2

        # out messages
        msg = str(action[0]/10.0) + ";" + str(action[1]/10.0)
        # msg = str(5) + ";" + str(5)
        self.argos_io.send_to(msg, agent_id)

        #print("action: ", action)

        self.observations[agent] = [floor]

        # if (self._cumulative_rewards[agent] < -800):
        #     self.truncations[agent] = True
        #     self.game_over = True

        # if (self.game_over):
        #     self.terminations[agent] = True

        if self._agent_selector.is_last():
            self._accumulate_rewards()
            self._clear_rewards()



        self.agent_selection = self._agent_selector.next()

        
