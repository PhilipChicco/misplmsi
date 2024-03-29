import warnings
warnings.filterwarnings("ignore")

import argparse
import yaml
import os, torch, numpy as np, random

# trainers
from evaluators import get_tester
from utils.misc import get_logger


def main(cfg):
    print(cfg)
    print()

    # setup logdir, writer and logger
    logdir = os.path.join(cfg['root'], cfg['testing']['logdir'])

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    tester_name = cfg['evaluator']

    with open(os.path.join(logdir, tester_name + '.yml'), 'w') as fp:
        yaml.dump(cfg, fp)

    logger = get_logger(logdir)

    print('Tester ', tester_name, __file__)
    Tester = get_tester(tester_name)(cfg, logdir, logger)
    print()

    # start testing
    Tester.test()

    #Tester.visualize_features()

if __name__ == '__main__':

    # get configs
    parser = argparse.ArgumentParser(description="Test a Network")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="research_mil/configs/amc_2020/deepset.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    seed = cfg['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(cfg)