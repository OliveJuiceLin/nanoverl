"""Train nanoverl from a config file."""

from __future__ import annotations

import argparse

from nanoverl.config import TrainerConfig
from nanoverl.trainer import build_trainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a nanoverl RL job.")
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML trainer config.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = TrainerConfig.load(args.config)
    trainer = build_trainer(config)
    try:
        trainer.fit()
    finally:
        trainer.close()


if __name__ == "__main__":  # pragma: no cover
    main()
