import sys
import subprocess
from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger
from DeepMimic_Optimizer import call_opti_main 


def main():
  # Command line arguments
  args = sys.argv[1:]
  arg_parser = ArgParser()
  arg_parser.load_args(args)

  num_workers = arg_parser.parse_int('num_workers', 1)
  assert (num_workers > 0)

  Logger.print2('Running with {:d} workers'.format(num_workers))
  call_opti_main() # start the training by calling the optimizer
  # subprocess.call(cmd, shell=True)
  return


if __name__ == '__main__':
  main()
