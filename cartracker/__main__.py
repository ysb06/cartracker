import cartracker.core
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./cartracker/config.yaml")
args = parser.parse_args()

cartracker.core.main(args.config)