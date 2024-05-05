from enum import Enum
from math import cos, pi, sin
import math
import os
from time import sleep
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from argos.argos_runner import Argos
from argos.argos_io import ArgosIO
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import functools

class State(Enum):

    MOVE = 1
    BUMP = 2
    PICK = 3
    DROP = 4
    GETTING_CLOSE = 5
    GETTING_CLOSE_TO_NEST = 6
    GETTING_AWAY_FROM_NEST = 7
    WEAK_GETTING_CLOSE = 8
    WEAK_GETTING_CLOSE_TO_NEST = 9
    WEAK_GETTING_AWAY_FROM_NEST = 10
    TRY_TO_AVOID_COLLISION = 11
    TRY_TO_MOVE_AWAY_FROM_COLLISION = 12
    POINTLESS_ROTATING = 13


REWARD_MAP = {
    State.MOVE: -0.5,
    State.BUMP: -15,
    State.PICK: 100,
    State.DROP: 300,
    State.GETTING_CLOSE: 1,
    State.GETTING_CLOSE_TO_NEST: 1,
    State.GETTING_AWAY_FROM_NEST: 1,
    State.WEAK_GETTING_CLOSE: -0.1,
    State.WEAK_GETTING_CLOSE_TO_NEST: -0.1,
    State.WEAK_GETTING_AWAY_FROM_NEST: -0.1,
    State.TRY_TO_AVOID_COLLISION: -0.75,
    State.TRY_TO_MOVE_AWAY_FROM_COLLISION: -0.5,
    State.POINTLESS_ROTATING: -15
}

FAST_CHANGE = 0.0085
ROTATION_PART = 1
MAX_ITER = 1500
MAX_ITER_STEP = 500
MIN_REWARD = -400
TARGET = 5

class ArgosForagingEnv(AECEnv):
    metadata = {'render.modes': ['human', 'no_render'], "name": "ArgosEnv"}

    def __init__(self, render_mode='human', verbose = None):
        # initialize and run argos
        self.num_robots = 2
        self.argos = None
        self.argos_io = None
        self.verbose = verbose if verbose != None else render_mode == 'human'

        self.env_id = 0
        self.delivered_food = 0

        self.possible_agents = ["robot_" + str(r) for r in range(self.num_robots)]

        self.actions_history = {agent: [] for agent in self.possible_agents}
        self.actions_history_int = {agent: [] for agent in self.possible_agents}
        self.obs_history = {agent: [] for agent in self.possible_agents}

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
                
        self._action_spaces = {
            # agent: spaces.Box(np.array([0, 0]), np.array([+50, +50]), shape=(2,)) for agent in self.possible_agents
            agent: spaces.Discrete(ROTATION_PART + ROTATION_PART + 1, start=-ROTATION_PART) for agent in self.possible_agents
        }
        # self._observation_spaces = {
        #     agent: spaces.Box(0, 256, shape=(1,)) for agent in self.possible_agents
        # }


        self._observation_spaces = {
            agent: spaces.flatten_space(spaces.Dict({
                "pos": spaces.Box(-2, 1, shape=(2,), dtype=float), 
                "rot": spaces.Box(-1, 1, shape=(2,), dtype=float), 
                "light_vector": spaces.Box(-2, 1, shape=(2,), dtype=float), 
                "in_nest": spaces.Discrete(1),
                "has_food": spaces.Discrete(1),
                "prox_vector": spaces.Box(-2, 1, shape=(2,), dtype=float), 
                "is_collision": spaces.Discrete(1),
                "rab_readings": spaces.Box(-2, 1, (self.num_robots -1, 4), dtype=float),
                "camera_readings": spaces.Box(-2, 1, (1, 3), dtype=float),
            })) for agent in self.possible_agents
        }

        self.render_mode = render_mode

        self.iter = 0
        self.max_iter = MAX_ITER

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
            self.argos = Argos(self.num_robots, render_mode=self.render_mode, verbose=self.verbose)
            
            # self.argos_io = ArgosIO(self.num_robots, verbose=False)
        else:
            # print("TRY TO RESET", self.argos.files_id)
            # self.argos.kill()
            self.argos.send_to(str(-1) + ";" + "RESET")
            for i in range(1, self.num_robots):
                self.argos.send_to(str(-1)+ ";" + "RESET", i)
            sleep(.15)
            if (not self.argos.receive_from(0).startswith('0')):
                print("I HATE THIS", self.argos.files_id, self.argos.receive_from(0))
                # self.argos.send_to(str(-1) + ";" + "RESET")
                # for i in range(1, self.num_robots):
                #     self.argos.send_to(str(-1)+ ";" + "RESET", i)
                # sleep(1)
                # os.sched_yield()
                # self.reset()
            # self.argos = Argos(self.num_robots, render_mode=self.render_mode, verbose=self.render_mode == 'human')


        self.iter = 0
        self.max_iter = MAX_ITER
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.env_id = self.env_id + 1
        self.delivered_food = 0

        self.actions_history = {agent: [] for agent in self.agents}
        self.actions_history_int = {agent: [] for agent in self.agents}
        self.obs_history = {agent: [] for agent in self.agents}

        self.observations = {agent: BotObservations([], self.num_robots).flatten_observations() for agent in self.agents}
        self.previous_observations = {agent: BotObservations([], self.num_robots) for agent in self.agents}

        self.game_over = False

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        
        
        self.iter = self.iter + 1
        agent = self.agent_selection
        self.actions_history[agent].append(f"{self.iter}: {action}")
        self.actions_history_int[agent].append(action)
        agent_id = self.agent_name_mapping[agent]
        startingProblem = False
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
        mapped_obs = BotObservations(in_msg, self.num_robots)
        prev_obs = self.previous_observations[agent]
        # if (mapped_observations.hasFood):
        #     a = 1
        # if (self.iter > 402):
        #     a = 1
        if (not mapped_obs.isValid):
            self.obs_history[agent].append(f"NOT VALID!!!!! {mapped_obs.iter if hasattr(mapped_obs, 'iter') else ''}")
        if (mapped_obs.isValid):
            self.obs_history[agent].append(f"{mapped_obs.iter}: {mapped_obs.xPos}; {mapped_obs.yPos}; {mapped_obs.zRot}")
            obs_iter_diff = mapped_obs.iter - prev_obs.iter

            if agent_id == 0:
                debug_point = 1


            # if (agent_id == 0 and len(mapped_obs.cameraReadings) != 0):
            #     print(mapped_obs.cameraReadings[0].x, mapped_obs.cameraReadings[0].y)

            # if (agent_id == 0):
            #     print(cos(mapped_obs.zRot), sin(mapped_obs.zRot))

            # if (agent_id == 0):
            #     print(mapped_obs.xPos, mapped_obs.yPos, mapped_obs.prox_vector_length())

            
            self.observations[agent] = mapped_obs.flatten_observations()
            if (obs_iter_diff == 0):
                # print("Same iter!",self.argos.files_id, mapped_obs.iter)
                self.rewards[agent] = 0
            elif (self.iter == 1 and mapped_obs.iter != 0):
                print("We have a starting problem")
                self.rewards[agent] = 0
                startingProblem = True
                self.iter-=1
            else:
                if (prev_obs.iter >= mapped_obs.iter):
                    print(agent, "OBS RUINED ",prev_obs.iter, " > ",  mapped_obs.iter)
                    a = 1
                # if (obs_iter_diff >=2 or obs_iter_diff == 0):
                #     print(agent, "BAD STEP", obs_iter_diff)

                just_rotating = False
                # if (len(self.actions_history_int[agent]) > 35):
                #     last_actions = self.actions_history_int[agent][len(self.actions_history_int[agent])-1-35:]
                #     rotating_left = False
                #     rotating_right = False
                #     forward_in_row = 0
                #     its_fine = False
                #     for e in last_actions:
                #         if e == 1:
                #             rotating_left = True
                #             forward_in_row = 0
                #         if e == 2:
                #             rotating_right = True
                #             forward_in_row = 0
                #         if e==0:
                #             forward_in_row += 1
                #         if (forward_in_row >= 4):
                #             its_fine = True
                #     just_rotating = not its_fine


                # if (len(self.actions_history_int[agent]) > 35):
                #     rotating_left = False
                #     rotating_right = False
                #     forward_in_row = 0
                #     rotating_iter = 0
                #     pattern_continues = 0
                #     last_actions = self.actions_history_int[agent][len(self.actions_history_int[agent])-1-35::-1]
                #     for e in last_actions:
                #         if e == 0:
                #             forward_in_row+=1
                #         if e==1:
                #             rotating_left = True
                #             forward_in_row=0
                #         if e==2:
                #             rotating_right = True
                #             forward_in_row=0
                #         if forward_in_row > 3:
                #             just_rotating = False
                #             break
                #         if rotating_left and rotating_right:
                #             pattern_continues+=1
                #         if pattern_continues > 10 and action != 0:
                #             just_rotating = True
                #             break

                # if (len(self.actions_history_int[agent]) > 35):
                #     rotating_left = False
                #     rotating_right = False
                #     forward_in_row = 0
                #     rotating_iter = 0
                #     pattern_continues = 0
                #     last_actions = self.actions_history_int[agent][len(self.actions_history_int[agent])-1-35::-1]

                        
                    

                if (prev_obs.hasFood and not mapped_obs.hasFood):
                    self.rewards[agent] = REWARD_MAP[State.DROP]
                    self.delivered_food += 1
                    if (self.delivered_food >=2):
                        self.rewards[agent]+=50*(self.delivered_food-1)
                        # self.max_iter += MAX_ITER_STEP
                    # print(agent, "ENV", self.env_id, " ACTUALLY DROPPED FOOD AT ", mapped_observations.xPos, "  ", mapped_observations.yPos)
                elif (not prev_obs.hasFood and mapped_obs.hasFood):
                    self.rewards[agent] = REWARD_MAP[State.PICK]
                    # print(agent, "ENV", self.env_id, " ACTUALLY PICKED FOOD AT ", mapped_observations.xPos, "  ", mapped_observations.yPos)
                elif (mapped_obs.isCollision):
                    if (not prev_obs.isCollision):
                        self.rewards[agent] = REWARD_MAP[State.BUMP]
                        if (agent_id == 0 and self.render_mode == 'human'):
                            print(self.iter, "BUMP", REWARD_MAP[State.BUMP])
                    else:
                        if (prev_obs.xPos == mapped_obs.xPos and prev_obs.yPos == mapped_obs.yPos and prev_obs.zRot != mapped_obs.zRot):
                            self.rewards[agent] = REWARD_MAP[State.TRY_TO_AVOID_COLLISION]
                            if (agent_id == 0 and self.render_mode == 'human'):
                                print(self.iter, "TRY TO AVOID COLLISION", REWARD_MAP[State.TRY_TO_AVOID_COLLISION])
                        elif (prev_obs.prox_vector_length() > mapped_obs.prox_vector_length()):
                            self.rewards[agent] = REWARD_MAP[State.TRY_TO_MOVE_AWAY_FROM_COLLISION]
                            if (agent_id == 0 and self.render_mode == 'human'):
                                print(self.iter, "TRY TO MOVE AWAY FROM COLLISION", REWARD_MAP[State.TRY_TO_MOVE_AWAY_FROM_COLLISION])
                        else:
                            self.rewards[agent] = REWARD_MAP[State.BUMP]
                            if (agent_id == 0 and self.render_mode == 'human'):
                                print(self.iter, "BUMP", REWARD_MAP[State.BUMP])
                elif (not mapped_obs.isCollision and prev_obs.isCollision):
                    self.rewards[agent] = -REWARD_MAP[State.BUMP]
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "UNBUMP", -REWARD_MAP[State.BUMP])
                elif (just_rotating):
                    self.rewards[agent] = REWARD_MAP[State.POINTLESS_ROTATING]
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "POINTLESS ROTATING", REWARD_MAP[State.POINTLESS_ROTATING])
                elif (not mapped_obs.hasFood and mapped_obs.food_approaching_diff(prev_obs) > FAST_CHANGE * obs_iter_diff):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "closer to food", REWARD_MAP[State.GETTING_CLOSE])
                    self.rewards[agent] = REWARD_MAP[State.GETTING_CLOSE]
                elif (not mapped_obs.hasFood and mapped_obs.food_approaching_diff(prev_obs) > 0):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "weak closer to food", REWARD_MAP[State.WEAK_GETTING_CLOSE])
                    self.rewards[agent] = REWARD_MAP[State.WEAK_GETTING_CLOSE]
                elif (not mapped_obs.hasFood and mapped_obs.has_food() and mapped_obs.food_approaching_diff(prev_obs) < 0):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "far from food", -2*REWARD_MAP[State.GETTING_CLOSE])
                    self.rewards[agent] = -2*REWARD_MAP[State.GETTING_CLOSE]


                # DONT USE
                # По какой-то причине вектор действительно увеличивается, когда робот едет к выходу из гнезда. Этот эффект исчезает примерно на границе гнезда с остальным пространством
                # elif (not mapped_observations.hasFood and mapped_observations.inNest and mapped_observations.light_vector_length() > self.previous_observations[agent].light_vector_length()):
                #     self.rewards[agent] = REWARD_MAP[State.GETTING_AWAY_FROM_NEST]
                # elif (not mapped_observations.hasFood and mapped_observations.inNest and mapped_observations.light_vector_length() < self.previous_observations[agent].light_vector_length()):
                #     self.rewards[agent] = -2*REWARD_MAP[State.GETTING_AWAY_FROM_NEST]




                elif (not mapped_obs.hasFood and mapped_obs.inNest and mapped_obs.xPos - prev_obs.xPos > FAST_CHANGE * obs_iter_diff):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "from nest", REWARD_MAP[State.GETTING_AWAY_FROM_NEST])
                    self.rewards[agent] = REWARD_MAP[State.GETTING_AWAY_FROM_NEST]
                elif (not mapped_obs.hasFood and mapped_obs.inNest and mapped_obs.xPos - prev_obs.xPos > 0):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "weak from nest", REWARD_MAP[State.WEAK_GETTING_AWAY_FROM_NEST])
                    self.rewards[agent] = REWARD_MAP[State.WEAK_GETTING_AWAY_FROM_NEST]
                elif (not mapped_obs.hasFood and mapped_obs.inNest and mapped_obs.xPos < prev_obs.xPos):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "closer to nest", -2*REWARD_MAP[State.GETTING_AWAY_FROM_NEST])
                    self.rewards[agent] = -2*REWARD_MAP[State.GETTING_AWAY_FROM_NEST]


                # DONT USE
                # elif (mapped_observations.hasFood and mapped_observations.light_vector_length() > self.previous_observations[agent].light_vector_length()):
                #     self.rewards[agent] = REWARD_MAP[State.GETTING_CLOSE_TO_NEST]
                # elif (mapped_observations.hasFood and mapped_observations.light_vector_length() < self.previous_observations[agent].light_vector_length()):
                #     self.rewards[agent] = -2*REWARD_MAP[State.GETTING_CLOSE_TO_NEST]



                elif (mapped_obs.hasFood and prev_obs.xPos - mapped_obs.xPos > FAST_CHANGE * obs_iter_diff):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "closer to nest with food", REWARD_MAP[State.GETTING_CLOSE_TO_NEST], )
                    self.rewards[agent] = REWARD_MAP[State.GETTING_CLOSE_TO_NEST]
                elif (mapped_obs.hasFood and prev_obs.xPos - mapped_obs.xPos > 0):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "weak closer to nest with food", REWARD_MAP[State.WEAK_GETTING_CLOSE_TO_NEST])
                    self.rewards[agent] = REWARD_MAP[State.WEAK_GETTING_CLOSE_TO_NEST]
                elif (mapped_obs.hasFood and mapped_obs.xPos > prev_obs.xPos):
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "from nest with food", -2*REWARD_MAP[State.GETTING_CLOSE_TO_NEST])
                    self.rewards[agent] = -2*REWARD_MAP[State.GETTING_CLOSE_TO_NEST]
                else:
                    if (agent_id == 0 and self.render_mode == 'human'):
                        print(self.iter, "just move", REWARD_MAP[State.MOVE])
                    self.rewards[agent] = REWARD_MAP[State.MOVE]

        else:
            self.rewards[agent] = 0
            action = 0

        if (mapped_obs.isValid):
            self.previous_observations[agent] = mapped_obs
        # out messages
        velocity = 10
        # if (self.iter < 5):
        #     action = 1
        # else:
        #     action = 9
        if (action > ROTATION_PART):
            action = ROTATION_PART - action
        heading = (velocity * cos(action * pi/ROTATION_PART), velocity * sin(action * pi/ROTATION_PART))
        msg = str(self.iter) + ";" + str(heading[0]) + ";" + str(heading[1])
        # if (self.iter == 400):
        #     self.argos.send_to(str(-1) + ";" + "RESET")
        #     for i in range(1, self.num_robots):
        #         self.argos.send_to(str(-1)+ ";" + "RESET")
        #     sleep(.05)
        # else:
        if (startingProblem):
            self.argos.send_to(str(-1) + ";" + "RESET", agent_id)
        else:
            self.argos.send_to(msg, agent_id)

        # print(self.iter, "; REWARD: ", self.rewards[agent])

        # if (self.iter % 199 == 0 or self.iter % 199 == 1 ): 
        #     print(self.iter, "; AGENT: ", agent, "  IN MESSAGE: ", in_msg, "  OUT MESSAGE: ", msg, " REWARD: ", self.rewards[agent], " CUMULATIVE REWARD: ", self._cumulative_rewards[agent] ) 

        if (agent_id == 0 and self._cumulative_rewards[agent] < MIN_REWARD):
            # if (agent_id == 0):
            self.terminations[agent] = True
            self.game_over = True
        
        # if (agent_id == 1 and self._cumulative_rewards[agent] < -600):
        #     # if (agent_id == 0):
        #     self.terminations[agent] = True
        #     self.game_over = True
        
        if (self.iter > self.max_iter):
            self.terminations[agent] = True
            self.game_over = True

        if (self.delivered_food >= TARGET):
            self.terminations[agent] = True
            self.game_over = True

        if (self.game_over):
            # print('REWARD FROM ROBOTS:', agent_id, self._cumulative_rewards[agent])
            self.terminations[agent] = True
            if agent_id == 0 and self._cumulative_rewards[agent] > 200:
                with open(f"robot_0_myfile.txt", "w") as file1:
                    file1.write("\n".join(map(str, self.actions_history['robot_0'])))
                    print("DONE!")
                with open(f"robot_0_myfileobs.txt", "w") as file1:
                    file1.write("\n".join(map(str, self.obs_history['robot_0'])))

                with open(f"robot_1_myfile.txt", "w") as file1:
                    file1.write("\n".join(map(str, self.actions_history['robot_1'])))
                with open(f"robot_1_myfileobs.txt", "w") as file1:
                    file1.write("\n".join(map(str, self.obs_history['robot_1'])))

        if self._agent_selector.is_last():
            self._accumulate_rewards()
            self._clear_rewards()





        self.agent_selection = self._agent_selector.next()

        
class BotObservations:
    
    def __init__(self, raw_observations: list[str], num_robots) -> None:
        self.isValid = False
        self.num_robots = num_robots
        try:
            # for e in raw_observations:
            #     if e.find('\\') != -1 or e.find('x') != -1 or e.find('00') != -1:
            #         print("OOOOOOOOF SOMETHING IS GOING OFF")
            self.iter = int(raw_observations[0]) if len(raw_observations) > 0 else -1
            self.xPos = float(raw_observations[1]) if len(raw_observations) > 1 else 0
            self.yPos = float(raw_observations[2]) if len(raw_observations) > 2 else 0
            self.zRot = float(raw_observations[3]) if len(raw_observations) > 3 else 0
            self.inNest = bool(int(raw_observations[4])) if len(raw_observations) > 4 else False
            self.hasFood = bool(int(raw_observations[5])) if len(raw_observations) > 5 else False
            self.xLight = float(raw_observations[6]) if len(raw_observations) > 6 else 0
            self.yLight = float(raw_observations[7]) if len(raw_observations) > 7 else 0
            self.xProx = float(raw_observations[8]) if len(raw_observations) > 8 else 0
            self.yProx = float(raw_observations[9]) if len(raw_observations) > 9 else 0
            self.isCollision = bool(int(raw_observations[10])) if len(raw_observations) > 10 else False
            rabReadingsSize = int(raw_observations[11]) if len(raw_observations) > 11 else 0
            self.rabReadings = []
            next_index = 12

            for i in range(rabReadingsSize):
                if (len(raw_observations) > next_index + 3):
                    self.rabReadings.append(RabReading(
                        x=float(raw_observations[next_index]),
                        y=float(raw_observations[next_index+1]),
                        has_food=bool(int(raw_observations[next_index+2])),
                        see_food=bool(int(raw_observations[next_index+3])),
                    ))
                    if self.rabReadings[len(self.rabReadings)-1].has_food and self.rabReadings[len(self.rabReadings)-1].see_food:
                        debug_point = 1
                    next_index += 4

            cameraReadingsSize = int(raw_observations[next_index]) if len(raw_observations) > next_index else 0
            self.cameraReadings = []
            next_index += 1

            for i in range(cameraReadingsSize):
                if (len(raw_observations) > next_index + 5):
                    self.cameraReadings.append(CameraReading(
                        x=float(raw_observations[next_index]),
                        y=float(raw_observations[next_index+1]),
                        angle=float(raw_observations[next_index+2]),
                        r=int(raw_observations[next_index+3]),
                        g=int(raw_observations[next_index+4]),
                        b=int(raw_observations[next_index+5]),
                    ))
                    next_index += 6
            self.isValid = True
        except:
            self.isValid = False
            #print("ALERT: Failed to parse observations")
            

    def flatten_observations(self):
        return np.concatenate([ [self.xPos, self.yPos, cos(self.zRot), sin(self.zRot), self.xLight, self.yLight, 
                                 int(self.inNest), int(self.hasFood), self.xProx, self.yProx, int(self.isCollision)], self.get_max_len_rab_readings(), self.get_max_len_camera_readings() ])
    
    def get_max_len_rab_readings(self):
        max_len = self.num_robots - 1
        result = []
        for i in range(len(self.rabReadings)):
            result.append(self.rabReadings[i].flatten_observations())
        for i in range(max_len - len(result)):
            result.append(RabReading(0,0,0,0).flatten_observations())
        return np.concatenate(result) if len(result) > 0 else []
    
    def get_closest_food(self):
        return functools.reduce(lambda a, b: b if b.is_blue() and self.vector_length(b.x, b.y) < self.vector_length(a.x, a.y) else a, 
                                    self.cameraReadings, CameraReading(5, 5, 0, 0, 0, 0))
    
    def get_max_len_camera_readings(self):
        max_len = 1#4
        result = []
        closestFood = self.get_closest_food()

        if closestFood.is_blue():
            result.append(closestFood.flatten_observations())
        # for i in range(len(self.cameraReadings)):
        #     if (self.cameraReadings[i].is_blue() and len(result) < max_len):
        #         result.append(self.cameraReadings[i].flatten_observations())
        # for i in range(len(self.cameraReadings)):
        #     if (not self.cameraReadings[i].is_blue() and len(result) < max_len):
        #         result.append(self.cameraReadings[i].flatten_observations())
        for i in range(max_len - len(result)):
            result.append(CameraReading(0,0,0,0,0,0).flatten_observations())
        return np.concatenate(result)

    def food_approaching_diff(self, other):
        if (not other.isValid):
            return False
        
        selfClosestFood = self.get_closest_food()
        otherClosestFood = other.get_closest_food()

        if (selfClosestFood.is_blue() and otherClosestFood.is_blue()):
            return self.vector_length(otherClosestFood.x, otherClosestFood.y) - self.vector_length(selfClosestFood.x, selfClosestFood.y)
        if (selfClosestFood.is_blue() and not otherClosestFood.is_blue()):
            return 1
        if (not selfClosestFood.is_blue() and otherClosestFood.is_blue()):
            return -1
        return 0    


        # blueReadings = list(filter(lambda e: e.is_blue(), self.cameraReadings))
        # otherBlueReadings = list(filter(lambda e: e.is_blue(), other.cameraReadings))
        # if (len(blueReadings) == 0): 
        #     return False
        # if (len(otherBlueReadings) == 0):
        #     return True
        
        # for e in blueReadings:
        #     for ee in otherBlueReadings:
        #         # if e.distance < ee.distance:
        #         if self.vector_length(e.x, e.y) < self.vector_length(ee.x, ee.y):
        #             # if (len(blueReadings) == 1):
        #                 # print("Found close one:", e.x, e.y, ee.x, ee.y, self.vector_length(e.x, e.y) , self.vector_length(ee.x, ee.y))
        #                 # a = 1
        #             return True
        # return False
    
    def has_food(self) -> bool:
        return len(self.cameraReadings) > 0 and len(list(filter(lambda e: e.is_blue(), self.cameraReadings))) > 0
    
    def light_vector_length(self):
        return math.sqrt(self.xLight * self.xLight + self.yLight * self.yLight)
    
    def prox_vector_length(self):
        return math.sqrt(self.xProx * self.xProx + self.yProx * self.yProx)
    
    def vector_length(self, x, y):
        return math.sqrt(x * x + y * y)


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BotObservations):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if (not other.isValid):
            return False
        return self.xPos == other.xPos and \
               self.yPos == other.yPos and \
               self.inNest == other.inNest and \
               self.hasFood == other.hasFood and \
               self.xLight == other.xLight and \
               self.yLight == other.yLight and \
               self.rabReadings == other.rabReadings and \
               self.cameraReadings == other.cameraReadings
                                       


class RabReading:
    def __init__(self, x: float, y: float, has_food: bool, see_food: bool) -> None:
        self.x = x
        self.y = y
        self.has_food = has_food
        self.see_food = see_food

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RabReading):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.x == other.x and \
               self.y == other.y and \
               self.has_food == other.has_food and \
               self.see_food == other.see_food
    
    def flatten_observations(self):
        return np.array([self.x, self.y, int(self.has_food), int(self.see_food)])

class CameraReading:
    def __init__(self, x: float, y: float, angle: float, r: int, g: int, b: int) -> None:
        self.x = x
        self.y = y
        self.angle = angle
        self.r = r
        self.g = g
        self.b = b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CameraReading):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.x == other.x and \
               self.y == other.y and \
               self.angle == other.angle and \
               self.r == other.r and \
               self.g == other.g and \
               self.b == other.b
    
    def flatten_observations(self):
        return np.array([self.x, self.y, int(self.is_blue())])

    def is_blue(self):
        return self.r == 0 and self.g == 0 and self.b == 255
    
    def is_green(self):
        return self.r == 0 and self.g == 255 and self.b == 0