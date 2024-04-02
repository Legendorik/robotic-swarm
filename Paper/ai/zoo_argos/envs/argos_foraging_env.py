from math import cos, pi, sin
from time import sleep
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from argos.argos_runner import Argos
from argos.argos_io import ArgosIO
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import functools

class ArgosForagingEnv(AECEnv):
    metadata = {'render.modes': ['human', 'no_render'], "name": "ArgosEnv"}

    def __init__(self, render_mode='human'):
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
            # agent: spaces.Box(np.array([0, 0]), np.array([+50, +50]), shape=(2,)) for agent in self.possible_agents
            agent: spaces.Discrete(9) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(0, 256, shape=(1,)) for agent in self.possible_agents
        }

        self.render_mode = render_mode

        self.iter = 0

        self.previous_observations = None

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
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self.observations[agent]

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
            self.argos = Argos(self.num_robots, render_mode=self.render_mode, verbose=self.render_mode == 'human')
            
            # self.argos_io = ArgosIO(self.num_robots, verbose=False)
        else:
            # self.argos.kill()
            self.argos.send_to(str(-1) + ";" + "RESET")
            for i in range(1, self.num_agents):
                self.argos.send_to(str(-1))
            sleep(.05)
            # self.argos = Argos(self.num_robots, render_mode=self.render_mode, verbose=self.render_mode == 'human')
        self.iter = 0
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}

        self.observations = {agent: np.array([0]) for agent in self.agents}
        self.previous_observations = {agent: np.array([0]) for agent in self.agents}

        self.game_over = False

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def get_observation_and_mask(self, obs):
        return {"observation": obs, }#"action_mask": self.get_action_mask()}
    
    def get_action_mask(self):
        return [[1, 1], [1, 1]]

    def step(self, action):
        self.iter = self.iter + 1
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
        # in_msg = self.argos_io.receive_from(agent_id)
        in_msg = self.argos.receive_from(agent_id)
        in_msg = in_msg.split(";")
        mapped_observations = BotObservations(in_msg)
        
        # if (len(in_msg) >= 3):
            # get floor color
            # floor = float(in_msg[3].replace('\x00', '').replace(',', '.'))
            # self.previous_observations[agent] = self.observations[agent]
            # self.observations[agent] = np.array([floor])
            # if (np.array_equal(self.previous_observations[agent], self.observations[agent])):
            #     self.rewards[agent] = 0
            # else:
            #     self.rewards[agent] = 1 if self.previous_observations[agent][0] < self.observations[agent][0] else -1
        # else:
        self.rewards[agent] = 0
        action = 0
        # out messages
        velocity = 10
        heading = (velocity * cos(action * pi/8), velocity * sin(action * pi/8))
        msg = str(self.iter) + ";" + str(heading[0]) + ";" + str(heading[1])
        # self.argos_io.send_to(msg, agent_id)
        self.argos.send_to(msg, agent_id)

        #print("action: ", action)

        if (self.iter % 199 == 0 or self.iter % 199 == 1 ): 
            print(self.iter, "; AGENT: ", agent, "  IN MESSAGE: ", in_msg, "  OUT MESSAGE: ", msg, " REWARD: ", self.rewards[agent], " CUMULATIVE REWARD: ", self._cumulative_rewards[agent] ) 

        if (self._cumulative_rewards[agent] < -10):
            self.truncations[agent] = True
            self.game_over = True
        
        if (self.iter > 2000):
            self.truncations[agent] = True
            self.game_over = True

        if (self.game_over):
            self.terminations[agent] = True

        if self._agent_selector.is_last():
            self._accumulate_rewards()
            self._clear_rewards()



        self.agent_selection = self._agent_selector.next()

        
class BotObservations:
    
    def __init__(self, raw_observations: list[str]) -> None:
        self.iter = int(raw_observations[0]) if len(raw_observations) > 0 else -1
        self.xPos = float(raw_observations[1]) if len(raw_observations) > 1 else 0
        self.yPos = float(raw_observations[2]) if len(raw_observations) > 2 else 0
        self.inNest = bool(raw_observations[3]) if len(raw_observations) > 3 else False
        self.hasFood = bool(raw_observations[4]) if len(raw_observations) > 4 else False
        self.xLight = float(raw_observations[5]) if len(raw_observations) > 5 else 0
        self.yLight = float(raw_observations[6]) if len(raw_observations) > 6 else 0
        rabReadingsSize = int(raw_observations[7]) if len(raw_observations) > 7 else 0
        self.rabReadings = []
        next_index = 8

        for i in range(rabReadingsSize):
            if (len(raw_observations) > next_index + 3):
                self.rabReadings.append(RabReading(
                    range=float(raw_observations[next_index]),
                    bearing=float(raw_observations[next_index+1]),
                    has_food=bool(raw_observations[next_index+2]),
                    see_food=bool(raw_observations[next_index+3]),
                ))
                next_index += 4

        cameraReadingsSize = int(raw_observations[next_index]) if len(raw_observations) > next_index else 0
        self.cameraReadings = []
        next_index += 1

        for i in range(cameraReadingsSize):
            if (len(raw_observations) > next_index + 4):
                self.cameraReadings.append(CameraReading(
                    distance=float(raw_observations[next_index]),
                    angle=float(raw_observations[next_index+1]),
                    r=int(raw_observations[next_index+2]),
                    g=int(raw_observations[next_index+3]),
                    b=int(raw_observations[next_index+4]),
                ))
                next_index += 5


class RabReading:
    def __init__(self, range: float, bearing: float, has_food: bool, see_food: bool) -> None:
        self.range = range
        self.bearing = bearing
        self.has_food = has_food
        self.see_food = see_food

class CameraReading:
    def __init__(self, distance: float, angle: float, r: int, g: int, b: int) -> None:
        self.distance = distance
        self.angle = angle
        self.r = r
        self.g = g
        self.b = b

    def is_blue(self):
        return self.r == 0 and self.g == 0 and self.b == 255
    
    def is_green(self):
        return self.r == 0 and self.g == 255 and self.b == 0