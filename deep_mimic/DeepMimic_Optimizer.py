import numpy as np
import sys
import os
import inspect

from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv
from pybullet_envs.deep_mimic.learning.rl_world import RLWorld
from pybullet_utils.logger import Logger
# from pybullet_envs.deep_mimic.testrl import build_world
from testrl import update_world, update_timestep, build_world
import pybullet_utils.mpi_util as MPIUtil

args = []
world = None


def run():
  global update_timestep
  global world
  done = False
  while not (done):
    update_world(world, update_timestep)

  return


def shutdown():
  global world

  Logger.print2('Shutting down...')
  world.shutdown()
  return


def main():
  global args
  global world

  # Command line arguments
  args = sys.argv[1:]
  enable_draw = False
  world = build_world(args, enable_draw)

  run()
  shutdown()

  return


def call_opti_main():
  main()

if __name__ == '__main__':
  main()
