#!/usr/bin/env python3
import argparse
from otitenet.data.make_dataset2 import build_from_config


def main():
    parser = argparse.ArgumentParser(description="Build processed otitis dataset from JSON config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing_config.json",
        help="Path to preprocessing JSON config",
    )
    args = parser.parse_args()
    build_from_config(args.config)


if __name__ == "__main__":
    main()
