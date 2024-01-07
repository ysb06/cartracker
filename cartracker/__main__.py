import argparse

import pytrainer

parser = argparse.ArgumentParser()
parser.add_argument("--module", type=str)
parser.add_argument("--config", type=str)

args = parser.parse_args()

pytrainer.execute(args)
