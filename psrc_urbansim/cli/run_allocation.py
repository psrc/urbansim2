import argparse
import sys

def add_run_args(parser, multiprocess=True):
    """
    Run command args
    """
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )

def run_allocation(configs_dir):
    print("Running allocation with configs dir:", configs_dir)
    # Here you would add the code to run the allocation model using the provided configs_dir
    # For example, you might call a function like run_allocation_model(configs_dir=args.configs_dir)

def run(args):
    run_allocation(args.configs_dir)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))