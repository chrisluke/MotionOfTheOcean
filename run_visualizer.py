import time
from train_model import build_world, update_world
import sys

update_timestep = 1. / 240.
animating = True
step = False
total_reward = 0
steps = 0


args = sys.argv[1:]


if __name__ == '__main__':

  world = build_world(args, True)
  env = world.env
  state = env._humanoid.getState()
  agentoo7 = world.agents[0]

  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
    if (animating or step):
      action = agentoo7.act(state)
      agentoo7._apply_action(action)
      state, reward, done = update_world(world)

      step = False
