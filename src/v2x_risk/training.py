from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import GraphWindowDataset, collate_graphs
from .metrics import classification_metrics
from .model import ModelOutput, V2XRiskModel
from .reproducibility import seed_everything, sha256_file, write_json


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    dataset: GraphWindowDataset,
    indices: Iterable[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(map(int, indices))),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graphs,
        generator=generator,
        num_workers=0,
    )


def balanced_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts.sum() / (num_classes * counts[nonzero])
    return torch.from_numpy(weights)


def combined_loss(
    output: ModelOutput, targets: torch.Tensor, criterion: nn.Module, training_config: dict
) -> torch.Tensor:
    return (
        float(training_config["fused_loss_weight"]) * criterion(output.fused_logits, targets)
        + float(training_config["temporal_aux_loss_weight"])
        * criterion(output.temporal_logits, targets)
        + float(training_config["spatial_aux_loss_weight"])
        * criterion(output.spatial_logits, targets)
    )


def evaluate_loader(
    model: V2XRiskModel,
    loader: DataLoader,
    device: torch.device,
    branch: str = "fused",
) -> tuple[dict, list[dict]]:
    logits_name = {
        "fused": "fused_logits",
        "temporal": "temporal_logits",
        "spatial": "spatial_logits",
    }[branch]
    model.eval()
    true_labels: list[int] = []
    predictions: list[int] = []
    rows: list[dict] = []
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(device)
            output = model(graph)
            logits = getattr(output, logits_name)
            probabilities = torch.softmax(logits, dim=-1)
            predicted = probabilities.argmax(dim=-1)
            for index in range(len(graph.y)):
                true_value = int(graph.y[index].cpu())
                predicted_value = int(predicted[index].cpu())
                true_labels.append(true_value)
                predictions.append(predicted_value)
                row = {
                    "frame_id": int(graph.frame_ids[index].cpu()),
                    "true_class": true_value,
                    "predicted_class": predicted_value,
                    "probability_low": float(probabilities[index, 0].cpu()),
                    "probability_medium": float(probabilities[index, 1].cpu()),
                    "probability_high": float(probabilities[index, 2].cpu()),
                }
                if branch == "fused":
                    row["fusion_weight_temporal"] = float(output.fusion_weights[index, 0].cpu())
                    row["fusion_weight_spatial"] = float(output.fusion_weights[index, 1].cpu())
                rows.append(row)
    if not true_labels:
        raise ValueError("Evaluation split is empty")
    metrics = classification_metrics(np.asarray(true_labels), np.asarray(predictions))
    if branch == "fused":
        metrics["mean_fusion_weights"] = {
            "temporal": float(np.mean([row["fusion_weight_temporal"] for row in rows])),
            "spatial": float(np.mean([row["fusion_weight_spatial"] for row in rows])),
        }
    return metrics, rows


def train_model(config: dict) -> dict:
    seed = int(config["seed"])
    seed_everything(seed, bool(config.get("deterministic", True)))
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    device = select_device(training_config["device"])

    dataset = GraphWindowDataset(data_config["processed_path"])
    train_indices = dataset.split_indices("train")
    validation_indices = dataset.split_indices("validation")
    test_indices = dataset.split_indices("test")
    train_loader = make_loader(
        dataset, train_indices, int(training_config["batch_size"]), True, seed
    )
    validation_loader = make_loader(
        dataset, validation_indices, int(training_config["batch_size"]), False, seed
    )
    test_loader = make_loader(
        dataset, test_indices, int(training_config["batch_size"]), False, seed
    )

    model = V2XRiskModel(model_config).to(device)
    if training_config.get("class_weighting") == "balanced":
        class_weights = balanced_class_weights(dataset.arrays["labels"][train_indices]).to(device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    checkpoint_path = Path(training_config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data_sha256 = sha256_file(data_config["processed_path"])
    history: list[dict] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        example_count = 0
        for graph in train_loader:
            graph = graph.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(graph)
            loss = combined_loss(output, graph.y, criterion, training_config)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(graph.y)
            example_count += len(graph.y)
        validation_metrics, _ = evaluate_loader(model, validation_loader, device, "fused")
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(example_count, 1),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(epoch_record)
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_macro_f1": best_macro_f1,
                    "processed_data_sha256": processed_data_sha256,
                },
                checkpoint_path,
            )

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["state_dict"])
    results = {
        "status": "regenerated",
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
        "validation": {},
        "test": {},
        "data": {
            "processed_path": str(data_config["processed_path"]),
            "processed_sha256": processed_data_sha256,
            "train_graphs": len(train_indices),
            "validation_graphs": len(validation_indices),
            "test_graphs": len(test_indices),
        },
    }
    for branch in ("temporal", "spatial", "fused"):
        results["validation"][branch], _ = evaluate_loader(model, validation_loader, device, branch)
        results["test"][branch], _ = evaluate_loader(model, test_loader, device, branch)
    write_json(training_config["metrics_path"], results)
    return results


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location=device)
