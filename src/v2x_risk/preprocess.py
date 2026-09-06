from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import (
    apply_minmax,
    build_aligned_graph_windows,
    chronological_split_codes,
    fit_minmax,
    load_and_clean_ngsim,
    save_graph_dataset,
)
from .reproducibility import sha256_file, write_json


def preprocess(config: dict, input_path: str | Path | None = None) -> dict:
    data_config = config["data"]
    csv_path = Path(input_path or data_config["raw_csv"])
    raw_columns = set(pd.read_csv(csv_path, nrows=0).columns)
    frame = load_and_clean_ngsim(csv_path, data_config)
    samples = build_aligned_graph_windows(frame, data_config)
    if data_config["split"].get("strategy") != "chronological":
        raise ValueError("Only chronological splitting is supported to limit temporal leakage")
    split_codes = chronological_split_codes(len(samples), data_config["split"])
    scaler = fit_minmax(samples, split_codes)
    apply_minmax(samples, scaler)
    save_graph_dataset(samples, split_codes, data_config["processed_path"])

    scaler_payload = {
        **scaler,
        "feature_columns": data_config["feature_columns"],
        "fit_split": "train",
    }
    write_json(data_config["scaler_path"], scaler_payload)
    split_names = {0: "train", 1: "validation", 2: "test", -1: "gap_excluded"}
    manifest = {
        "raw_csv": str(csv_path),
        "raw_sha256": sha256_file(csv_path),
        "cleaned_rows": len(frame),
        "graph_windows": len(samples),
        "node_histories": int(sum(len(sample.x) for sample in samples)),
        "sequence_length": int(data_config["sequence_length"]),
        "graph_radius_m": float(data_config["graph_radius_m"]),
        "class_encoding": {"low": 0, "medium": 1, "high": 2},
        "graph_label_reduction": data_config["graph_label_reduction"],
        "labeling_source": resolve_labeling_source(data_config, raw_columns),
        "split_counts": {
            split_names[code]: int(count) for code, count in Counter(split_codes.tolist()).items()
        },
        "label_counts": {
            str(label): int(count)
            for label, count in Counter(sample.y for sample in samples).items()
        },
        "processed_sha256": sha256_file(data_config["processed_path"]),
    }
    write_json(data_config["manifest_path"], manifest)
    return manifest


def resolve_labeling_source(data_config: dict, raw_columns: set[str]) -> str:
    configured = data_config["labeling"].get("source", "existing_or_heuristic")
    if configured == "existing" or (
        configured == "existing_or_heuristic" and data_config["label_column"] in raw_columns
    ):
        return "existing Risk_Class column"
    return "configured heuristic assumptions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess NGSIM into aligned graph windows")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", help="Override data.raw_csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = preprocess(load_config(args.config), args.input)
    print(
        f"Wrote {manifest['graph_windows']} aligned graph windows "
        f"from {manifest['cleaned_rows']} cleaned rows."
    )


if __name__ == "__main__":
    main()
