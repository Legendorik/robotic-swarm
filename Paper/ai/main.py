from time import sleep, time
from zoo_argos.envs.argos_env import ArgosEnv
# env = ArgosEnv(render_mode='no_render')
env = ArgosEnv(render_mode='human')
env.reset()

start_time = time()
cur_time = time()
iter = 0
for agent in env.agent_iter(max_iter=2000):
    if (iter == 2):
        start_time = time()
    iter += 1
    cur_time = time()
    diff_time = cur_time - start_time
    if (diff_time == 0):
        diff_time = 1
    iter_per_second = iter / diff_time
    print('Per second: ', iter_per_second)
    observation, reward, termination, truncation, info = env.last()
    # print('Observation shape', observation.shape, env.observation_space(agent))
    print('Agent [', agent, '] sees: ', observation)
    print('Agent [', agent, '] reward: ', reward)
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy
    # print('Agent action', agent, action)
    env.step(action)
    if (env.render_mode == 'human'):
        sleep(0.05) 

print('Total iterations: ', iter)
env.close()

# Next: count ticks in python and argos
# argos doing 10000 ticks per robot, python only 300. out of sync. Check if socket messages queue exists?
# in visualization mode experiments are pointless. you can increase ticks, but does it really speeds things up? I dont think so, because it just steps more often with the same visualisation speed
# There are two ways: find a new way to transfer data (Tensorswarm with ROS? Looks very hard) or ask if i can set a simulation speed somewhere
# 
# i cant launch with -z from python without launching simulation from the terminal. It just stops when python env trying to get a value. maybe not important, as long i can start it manually


## NOTES
# It is possible to set how many iterations this physics engine performs between\n" "each simulation step. By default, this physics engine performs 10 steps every\n" "two simulation steps. This means that, if the simulation step is 100ms, the\n" "physics engine step is, by default, 10ms. Sometimes, collisions and joints are\n" "not simulated with sufficient precision using these parameters. By increasing\n" "the number of iterations, the temporal granularity of the solver increases and\n" "with it its accuracy, at the cost of higher computational cost. To change the\n" "number of iterations per simulation step use this syntax:\n\n" " <physics_engines>\n" " ...\n" " <dynamics2d id=\"dyn2d\"\n" " iterations=\"20\" />\n" " 

#This environment allows agents to spawn and die, so it requires using SuperSuit’s Black Death wrapper, which provides blank observations to dead agents rather than removing them from the environment.

