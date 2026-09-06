from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import load_config
from .data import GraphWindowDataset
from .model import V2XRiskModel
from .reproducibility import seed_everything, write_json
from .training import evaluate_loader, load_checkpoint, make_loader, select_device


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str | Path | None = None,
    split: str = "test",
    output_path: str | Path | None = None,
) -> dict:
    seed = int(config["seed"])
    seed_everything(seed, bool(config.get("deterministic", True)))
    training_config = config["training"]
    device = select_device(training_config["device"])
    dataset = GraphWindowDataset(config["data"]["processed_path"])
    loader = make_loader(
        dataset,
        dataset.split_indices(split),
        int(training_config["batch_size"]),
        False,
        seed,
    )
    resolved_checkpoint = Path(checkpoint_path or training_config["checkpoint_path"])
    checkpoint = load_checkpoint(resolved_checkpoint, device)
    model = V2XRiskModel(checkpoint.get("model_config", config["model"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    payload = {
        "status": "regenerated",
        "split": split,
        "checkpoint": str(resolved_checkpoint),
        "branches": {},
    }
    prediction_rows: list[dict] = []
    for branch in ("temporal", "spatial", "fused"):
        payload["branches"][branch], rows = evaluate_loader(model, loader, device, branch)
        if branch == "fused":
            prediction_rows = rows

    destination = Path(output_path or resolved_checkpoint.with_name(f"{split}_metrics.json"))
    write_json(destination, payload)
    predictions_path = destination.with_name(f"{split}_predictions.csv")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained V2X risk checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate_checkpoint(
        load_config(args.config), args.checkpoint, args.split, args.output
    )
    fused = payload["branches"]["fused"]
    print(f"{args.split} accuracy={fused['accuracy']:.4f}, macro_f1={fused['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
