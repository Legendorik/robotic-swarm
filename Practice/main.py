import pygame
import foraging_env
import very_simple_env
# import single_env
# env = foraging_env.env(render_mode='human')

# env.reset()
# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#     action = env.action_space(agent).sample() # this is where you would insert your policy
#     env.step(action)


import gymnasium

# env = single_env.GridWorldEnv('human')
env = foraging_env.env(render_mode='human')
env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
# print('space:', env.observation_spaces)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    # print('Observation shape', observation.shape, env.observation_space(agent))
    print('Agent observation', agent, observation)
    print('Agent reward', agent, reward)
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy
    print('Agent action', agent, action)
    env.step(action)

env.close()
