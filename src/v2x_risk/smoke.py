from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config, with_overrides
from .evaluate import evaluate_checkpoint
from .preprocess import preprocess
from .training import train_model


def write_synthetic_ngsim(path: str | Path, vehicles: int = 6, frames: int = 45) -> Path:
    """Create deterministic schema-compatible data for execution tests only."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for frame_id in range(frames):
        risk_class = frame_id % 3
        for vehicle_id in range(1, vehicles + 1):
            rows.append(
                {
                    "Vehicle_ID": vehicle_id,
                    "Frame_ID": frame_id,
                    "Global_Time": frame_id * 100,
                    "Local_X": float((vehicle_id - 1) % 3) * 3.5,
                    "Local_Y": float(frame_id) * 0.9 + float(vehicle_id - 1) * 2.5,
                    "v_Vel": 10.0 + vehicle_id * 0.3 - risk_class * 0.5,
                    "v_Acc": [0.1, -2.2, -4.0][risk_class],
                    "Time_Headway": [3.0, 1.5, 0.7][risk_class],
                    "Space_Headway": [45.0, 25.0, 10.0][risk_class],
                    "Risk_Class": risk_class,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_smoke(config_path: str | Path, work_dir: str | Path | None = None) -> dict:
    config = load_config(config_path)
    if work_dir is not None:
        work_dir = Path(work_dir)
        config = with_overrides(
            config,
            {
                "data": {
                    "raw_csv": str(work_dir / "ngsim_smoke.csv"),
                    "processed_path": str(work_dir / "dataset.npz"),
                    "scaler_path": str(work_dir / "scaler.json"),
                    "manifest_path": str(work_dir / "manifest.json"),
                },
                "training": {
                    "checkpoint_path": str(work_dir / "best_model.pt"),
                    "metrics_path": str(work_dir / "metrics.json"),
                },
            },
        )
    write_synthetic_ngsim(config["data"]["raw_csv"])
    preprocess_manifest = preprocess(config)
    training_results = train_model(config)
    evaluation_results = evaluate_checkpoint(config)
    return {
        "preprocess": preprocess_manifest,
        "training": training_results,
        "evaluation": evaluation_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a one-epoch synthetic execution test")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--work-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_smoke(args.config, args.work_dir)
    metrics = result["evaluation"]["branches"]["fused"]
    print(
        "Smoke test completed; synthetic metrics are execution checks only: "
        f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
