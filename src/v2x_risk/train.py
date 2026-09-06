from __future__ import annotations

import argparse

from .config import load_config
from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the LSTM-GAT attention-fusion model")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = train_model(load_config(args.config))
    fused = results["test"]["fused"]
    print(f"Regenerated test accuracy={fused['accuracy']:.4f}, macro_f1={fused['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
