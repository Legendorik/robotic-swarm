import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict, flatten_space
from gymnasium.wrappers import flatten_observation

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from supersuit import flatten_v0

import pygame
from enum import Enum

class State(Enum):

    MOVE = 1
    BUMP = 2
    PICK = 3
    DROP = 4
    PICK_ERROR = 5
    DROP_ERROR = 6


MOVES = ['RIGHT', 'TOP', 'LEFT', 'DOWN', 'PICK', 'DROP']
NUM_ITERS = 75
REWARD_MAP = {
    State.MOVE: -1,
    State.BUMP: -10,
    State.PICK: 30,
    State.DROP: 60,
    State.PICK_ERROR: -5,
    State.DROP_ERROR: -5,
}

# np.random.seed(1)




def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode #if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # env = flatten_observation.FlattenObservation(env)
    # env = flatten_v0(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2", "render_fps": 4}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        # super().__init__()

        self.size = 5  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        self.texts = []
        self.textRects = []

        self.base_position = np.array([self.size / 2 - 1, self.size - 1], dtype=int)


        self.possible_agents = ["player_" + str(r) for r in range(2)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }
        # 0, 1, 2, 3 - movement
        # 4 - pick up food, 5 - deliver food
        self._observation_spaces = {
            agent: flatten_space(Dict({
                "agent_position": Box(0, self.size - 1, shape=(2,), dtype=int), 
                "base_position": Box(0, self.size - 1, shape=(2,), dtype=int), 
                "has_food": Discrete(1),
                "agent_surroundings": Box(-1, 1, (9,), dtype=int),
                # -2 - unknown, 0 - just cells, 1 - food, 2 - other agents
                "discovered_map": Box(-2, 2, (self.size, self.size), dtype=int)
            })) for agent in self.possible_agents
        }

        

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None

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

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        
        if self.render_mode == "human":
            return self._render_frame()

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * self.base_position,
                (pix_square_size, pix_square_size),
            ),
        )


        for i in range(len(self.field)):
            for j in range(len(self.field[i])):
                if (self.field[i][j] == 1):
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),
                        (np.array([i, j]) + 0.5) * pix_square_size,
                        pix_square_size / 2.5,
                        # (2,2),
                        # pix_square_size/3
                    )

        self.texts = []
        self.textRects = []

        # Now we draw the agent
        for agent in self.possible_agents:
            position = self.agents_positions[agent] #self._observation_spaces[agent].get('agent_position').
        
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (position + 0.5) * pix_square_size,
                pix_square_size / 3,
                # (2,2),
                # pix_square_size/3
            )
            if (self.agents_having_food[agent] == 1):
                pygame.draw.circle(
                    canvas,
                    (255, 0, 0),
                    (position + 0.5) * pix_square_size,
                    pix_square_size / 6,
                    # (2,2),
                    # pix_square_size/3
                )
            # create a font object.
            # 1st parameter is the font file
            # which is present in pygame.
            # 2nd parameter is size of the font
            font = pygame.font.Font('freesansbold.ttf', 24)
            
            # create a text surface object,
            # on which text is drawn on it.
            text = font.render(agent, True, (0, 0, 0))
            
            # create a rectangular object for the
            # text surface object
            textRect = text.get_rect()
            
            # set the center of the rectangular object.
            textRect.center = ((position[0] + 0.5) * pix_square_size, (position[1] + 0.87) * pix_square_size)

            self.texts.append(text)
            self.textRects.append(textRect)

            

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            for i in range(len(self.texts)):
                self.window.blit(self.texts[i], self.textRects[i])
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

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
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # def generate_observation(self):
    #             self._observation_spaces = {
    #         agent: Dict({
    #             "agent_position": Box(0, self.size - 1, shape=(2,), dtype=int), 
    #             "agent_surroundings": Box(0, 1, (3,3), dtype=int),
    #         }) for agent in self.possible_agents
    #     }

    def update_observations(self):
        self.observations = {agent: self.flatten_observations(agent) for agent in self.agents}
        # self.observations = {agent: {'agent_position': self.agents_positions[agent],
        #                              'base_position': self.base_position, 
        #                              'has_food': self.agents_having_food[agent],
        #                              'agents_surroundings': self.agents_surroundings[agent],
        #                              'discovered_map': self.discovered_map,
        #                             } for agent in self.agents}
        
    def flatten_observations(self, agent):
        # return np.concatenate([self.agents_positions[agent], self.agents_surroundings[agent], self.base_position, [self.agents_having_food[agent]],])
        return np.concatenate([self.agents_positions[agent], self.agents_surroundings[agent], self.base_position, self.discovered_map.flatten(), [self.agents_having_food[agent]],])
        # return np.concatenate([self.agents_positions[agent], self.base_position, [self.agents_having_food[agent]], self.agents_surroundings[agent]])

    def generate_food(self, num: int):
        for i in range(num):
            position = np.random.randint(0, self.size, size=2, dtype=int)
            self.field[position[0], position[1]] = 1

    def is_outside(self, position):
        if position[0] < 0 or position[0] >= self.size or position[1] < 0 or position[1] >= self.size:
            return True
        return False
    
    def get_field_value(self, position):
        if self.is_outside(position):
            return -1
        return self.field[position[0], position[1]]

    def get_surroundings(self, agent):
        agent_position = self.agents_positions[agent]
        return np.array([
            self.get_field_value([agent_position[0]-1, agent_position[1]-1]),
            self.get_field_value([agent_position[0]-0, agent_position[1]-1]),
            self.get_field_value([agent_position[0]+1, agent_position[1]-1]),
            self.get_field_value([agent_position[0]-1, agent_position[1]-0]),
            self.get_field_value([agent_position[0]-0, agent_position[1]-0]),
            self.get_field_value([agent_position[0]+1, agent_position[1]-0]),
            self.get_field_value([agent_position[0]-1, agent_position[1]+1]),
            self.get_field_value([agent_position[0]-0, agent_position[1]+1]),
            self.get_field_value([agent_position[0]+1, agent_position[1]+1]),
        ], dtype=int)
    
    def count_food(self):
        counter = 0
        for i in range(len(self.field)):
            for j in range(len(self.field[i])):
                if (self.field[i][j] == 1):
                    counter+=1

        return counter
    
    def update_discovered_map(self):
        current_agent_positions = []
        for agent in self.agents:
            agent_position = self.agents_positions[agent]
            current_agent_positions.append(agent_position)
            for shift in ([[-1, -1], [0, -1],[1, -1],[-1, 0], [0, 0],[1, 0],[-1, 1], [0, 1],[1, 1],]):
                field_value = self.get_field_value([agent_position[0] + shift[0], agent_position[1] + shift[1]])
                if field_value != -1:
                    # print('UPDATE POSITION', [agent_position[0] + shift[0], agent_position[1] + shift[1]])
                    self.discovered_map[agent_position[0] + shift[0], agent_position[1] + shift[1]] = field_value
                    # print('UPDATE MAP\n', self.discovered_map)
        for i in range(len(current_agent_positions)):
            self.discovered_map[current_agent_positions[i][0], current_agent_positions[i][1]] = 2

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
        
        self.field = np.zeros((self.size, self.size))
        self.generate_food(15)
        self.food_spawned = self.count_food()
        self.food_delivered = 0

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}

        self.agents_positions = {agent: np.random.randint([0, self.size - 3], self.size, size=2, dtype=int) for agent in self.agents}
        self.agents_surroundings = {agent: self.get_surroundings(agent) for agent in self.agents}
        self.agents_having_food = {agent: 0 for agent in self.agents}
        self.discovered_map = np.full((self.size, self.size), -2, dtype=int)
        # print('START DISCOVERED MAP\n', self.discovered_map)
        self.update_discovered_map()
        # print('Agent positions', self.agents_positions)
        # print('Field\n', self.field)
        # print('NEW DISCOVERED MAP\n', self.discovered_map)

        # self.observations = {agent: {'agent_position': self.agents_positions[agent], 'agents_surroundings': self.agents_surroundings[agent]} for agent in self.agents}
        self.update_observations()
        self.game_over = False
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        # self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        current_position = self.agents_positions[agent]
        if (action < 4):
            direction = self._action_to_direction[action]
            new_position = self.agents_positions[agent] + direction
            if self.is_outside(new_position):
                self.rewards[agent] = REWARD_MAP[State.BUMP]
            else:
                # We use `np.clip` to make sure we don't leave the grid
                self.agents_positions[agent] = np.clip(
                    self.agents_positions[agent] + direction, 0, self.size - 1
                )
                if (self.field[self.agents_positions[agent][0], self.agents_positions[agent][1]] == 1 and self.agents_having_food[agent] == 0):
                    self.rewards[agent] = REWARD_MAP[State.PICK]
                    self.field[self.agents_positions[agent][0], self.agents_positions[agent][1]] = 0
                    self.agents_having_food[agent] = 1
                    # self.food_delivered += 1
                elif (self.agents_having_food[agent] == 1 and np.array_equal(current_position, self.base_position)):
                    self.food_delivered += 1
                    self.agents_having_food[agent] = 0
                    self.rewards[agent] = REWARD_MAP[State.DROP]
                else:
                    self.rewards[agent] = REWARD_MAP[State.MOVE]
        elif (action == 4):
            if (self.field[current_position[0], current_position[1]] == 1):
                self.rewards[agent] = REWARD_MAP[State.PICK]
                self.field[current_position[0], current_position[1]] = 0
                self.food_delivered += 1
            else:
                self.rewards[agent] = REWARD_MAP[State.PICK_ERROR]
            # if (self.agents_having_food[agent] == 0 and self.field[current_position[0], current_position[1]] == 1):
            #     self.agents_having_food[agent] = 1
            #     self.field[current_position[0], current_position[1]] = 0
            #     self.rewards[agent] = REWARD_MAP[State.PICK]
            # else:
            #     self.rewards[agent] = REWARD_MAP[State.PICK_ERROR]
        # elif (action == 5):
        #     if (self.agents_having_food[agent] == 1 and np.array_equal(current_position, self.base_position)):
        #         self.food_delivered += 1
        #         self.agents_having_food[agent] = 0
        #         self.rewards[agent] = REWARD_MAP[State.DROP]
        #     else: 
        #         self.rewards[agent] = REWARD_MAP[State.DROP_ERROR]

        self.agents_surroundings[agent] = self.get_surroundings(agent)

        

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
            #     (self.state[self.agents[0]], self.state[self.agents[1]])
            # ]

            self.update_discovered_map()
            

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            # Adds .rewards to ._cumulative_rewards
            self._accumulate_rewards()
            self._clear_rewards()

            # observe the current state
            # for i in self.agents:
            #     self.observations[i] = self.state[
            #         self.agents[1 - self.agent_name_mapping[i]]
            #     ]
        # else:
            # necessary so that observe() returns a reasonable observation at all times.
            # self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            # self._clear_rewards()

        self.update_observations()

        if (self._cumulative_rewards[agent] < -100):
            self.game_over = True

        if (self.food_spawned <= self.food_delivered):
            self.terminations[agent] = True
        if (self.game_over):
            self.terminations[agent] = True
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()




        if self.render_mode == "human":
            self.render()