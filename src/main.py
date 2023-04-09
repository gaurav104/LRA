import argparse
from utils.config import *

from agents import *
import torch
import numpy as np
import random


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')

    arg_parser.add_argument('--benchmark', action='store_true')

    arg_parser.add_argument('--cudnn_disabled', action='store_true')

    arg_parser.add_argument('--use_cluster', action='store_true')

    args = arg_parser.parse_args()

    if args.cudnn_disabled:

        torch.backends.cudnn.enabled = False
    

    if args.benchmark:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    # parse the config json file
    config = process_config(args.config, cluster=args.use_cluster)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
