from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RISK_NAME_TO_CLASS = {
    "low": 0,
    "low risk": 0,
    "safe": 0,
    "medium": 1,
    "medium risk": 1,
    "warning": 1,
    "high": 2,
    "high risk": 2,
    "emergency": 2,
}
TRAJECTORY_SEGMENT_COLUMN = "_Trajectory_Segment"
ZIP_SIGNATURE = b"PK\x03\x04"


@dataclass
class NumpyGraphSample:
    x: np.ndarray
    sequences: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    y: int
    frame_id: int
    vehicle_ids: np.ndarray


@dataclass
class GraphSample:
    x: torch.Tensor
    sequences: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    frame_id: torch.Tensor
    vehicle_ids: torch.Tensor


@dataclass
class GraphBatch:
    x: torch.Tensor
    sequences: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor
    frame_ids: torch.Tensor
    vehicle_ids: torch.Tensor

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(**{name: value.to(device) for name, value in vars(self).items()})


def coerce_risk_classes(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
    else:
        numeric = values.astype(str).str.strip().str.lower().map(RISK_NAME_TO_CLASS)
    valid = numeric.dropna()
    if not valid.isin([0, 1, 2]).all():
        invalid = sorted(set(valid.astype(int)) - {0, 1, 2})
        raise ValueError(f"Risk labels must use classes 0, 1, 2; found {invalid}")
    return numeric.astype("Int64")


def assign_risk_labels(frame: pd.DataFrame, labeling: dict) -> pd.Series:
    """Apply the repository's explicit, configurable risk-label assumptions.

    The paper names time headway, space headway, and acceleration as label inputs but
    does not publish their numerical cutoffs. The defaults therefore live in YAML and
    are intentionally not represented as paper-derived constants.
    """

    required = {"Time_Headway", "Space_Headway", "v_Acc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Heuristic labeling requires columns " + ", ".join(missing) + ". "
            "Provide Risk_Class in the CSV or add the missing inputs."
        )

    time_headway = pd.to_numeric(frame["Time_Headway"], errors="coerce")
    space_headway = pd.to_numeric(frame["Space_Headway"], errors="coerce")
    acceleration = pd.to_numeric(frame["v_Acc"], errors="coerce")

    high = (
        (time_headway <= float(labeling["high_time_headway_s"]))
        | (space_headway <= float(labeling["high_space_headway_ft"]))
        | (acceleration <= float(labeling["high_acceleration_fps2"]))
    )
    medium = (
        (time_headway <= float(labeling["medium_time_headway_s"]))
        | (space_headway <= float(labeling["medium_space_headway_ft"]))
        | (acceleration <= float(labeling["medium_acceleration_fps2"]))
    )

    labels = np.full(len(frame), int(labeling["low_class"]), dtype=np.int64)
    labels[medium.to_numpy()] = int(labeling["medium_class"])
    labels[high.to_numpy()] = int(labeling["high_class"])
    return pd.Series(labels, index=frame.index, name="Risk_Class")


def trajectory_group_columns(frame: pd.DataFrame, data_config: dict) -> list[str]:
    """Return columns that uniquely identify a trajectory in a combined NGSIM CSV."""

    columns: list[str] = []
    location_col = data_config.get("location_column")
    if location_col and location_col in frame.columns:
        columns.append(location_col)
    columns.append(data_config["vehicle_column"])
    if TRAJECTORY_SEGMENT_COLUMN in frame.columns:
        columns.append(TRAJECTORY_SEGMENT_COLUMN)
    return columns


def add_trajectory_segments(frame: pd.DataFrame, data_config: dict) -> pd.DataFrame:
    """Split reused vehicle IDs into contiguous observation runs."""

    vehicle_col = data_config["vehicle_column"]
    frame_col = data_config["frame_column"]
    location_col = data_config.get("location_column")
    base_groups = [vehicle_col]
    if location_col and location_col in frame.columns:
        base_groups.insert(0, location_col)
    time_col = data_config.get("time_column")
    sort_columns = [*base_groups]
    if time_col and time_col in frame.columns:
        sort_columns.append(time_col)
    sort_columns.append(frame_col)
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    observation_key = [*base_groups, frame_col]
    if time_col and time_col in frame.columns:
        observation_key.append(time_col)
    frame = frame.drop_duplicates(observation_key, keep="last").reset_index(drop=True)

    grouped = frame.groupby(base_groups, sort=False, dropna=False)
    first = grouped.cumcount().eq(0)
    discontinuity = grouped[frame_col].diff().ne(1)
    if time_col and time_col in frame.columns:
        time_difference = grouped[time_col].diff()
        discontinuity |= time_difference.le(0)
        kinematics = data_config.get("kinematics", {})
        expected_step = float(kinematics.get("frame_interval_s", 0.1)) / float(
            kinematics.get("time_scale_to_s", 0.001)
        )
        discontinuity |= ~np.isclose(time_difference, expected_step, rtol=0.0, atol=1e-6)
    starts = (~first & discontinuity).astype(np.int64)
    frame[TRAJECTORY_SEGMENT_COLUMN] = starts.groupby(
        [frame[column] for column in base_groups], sort=False
    ).cumsum()
    return frame


def derive_missing_kinematics(frame: pd.DataFrame, data_config: dict) -> list[str]:
    """Derive absent speed/acceleration fields from position and elapsed time.

    NGSIM positions are in feet by default. The absolute derivative of longitudinal
    position is therefore speed in ft/s, and its derivative is acceleration in
    ft/s^2. Existing columns are never replaced.
    """

    options = data_config.get("kinematics", {})
    if not bool(options.get("derive_missing", False)):
        return []
    missing = [column for column in ("v_Vel", "v_Acc") if column not in frame.columns]
    if not missing:
        return []

    position_col = options.get("position_column", "Local_Y")
    if position_col not in frame.columns:
        raise ValueError(
            f"Cannot derive {', '.join(missing)} without position column {position_col}"
        )

    time_col = data_config.get("time_column")
    frame_col = data_config["frame_column"]
    time_scale_to_s = float(options.get("time_scale_to_s", 0.001))
    frame_interval_s = float(options.get("frame_interval_s", 0.1))
    groups = trajectory_group_columns(frame, data_config)
    derived = {column: np.full(len(frame), np.nan, dtype=np.float64) for column in missing}

    for positions in frame.groupby(groups, sort=False, dropna=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        trajectory = frame.iloc[positions]
        if time_col and time_col in trajectory.columns:
            elapsed = trajectory[time_col].to_numpy(dtype=np.float64) * time_scale_to_s
        else:
            elapsed = trajectory[frame_col].to_numpy(dtype=np.float64) * frame_interval_s
        if len(elapsed) < 2 or not np.all(np.diff(elapsed) > 0):
            elapsed = trajectory[frame_col].to_numpy(dtype=np.float64) * frame_interval_s
        if len(elapsed) < 2 or not np.all(np.diff(elapsed) > 0):
            continue

        if "v_Vel" in missing:
            longitudinal = trajectory[position_col].to_numpy(dtype=np.float64)
            velocity = np.abs(np.gradient(longitudinal, elapsed))
            derived["v_Vel"][positions] = velocity
        else:
            velocity = trajectory["v_Vel"].to_numpy(dtype=np.float64)
        if "v_Acc" in missing:
            acceleration = np.gradient(velocity, elapsed)
            derived["v_Acc"][positions] = acceleration

    for column, values in derived.items():
        values[~np.isfinite(values)] = np.nan
        frame[column] = values
    return missing


def input_is_excel(path: str | Path) -> bool:
    """Detect OOXML workbooks by content, including files with a .csv suffix."""

    with Path(path).open("rb") as handle:
        return handle.read(len(ZIP_SIGNATURE)) == ZIP_SIGNATURE


def read_input_columns(path: str | Path) -> set[str]:
    if input_is_excel(path):
        return set(pd.read_excel(path, engine="openpyxl", nrows=0).columns)
    return set(pd.read_csv(path, nrows=0).columns)


def load_and_clean_ngsim(csv_path: str | Path, data_config: dict) -> pd.DataFrame:
    csv_path = Path(csv_path)
    features = list(data_config["feature_columns"])
    vehicle_col = data_config["vehicle_column"]
    frame_col = data_config["frame_column"]
    label_col = data_config["label_column"]

    raw_columns = read_input_columns(csv_path)
    location_col = data_config.get("location_column")
    active_location_col = location_col if location_col in raw_columns else None

    sample_rows = data_config.get("sample_rows")
    if sample_rows is not None:
        frame = read_complete_trajectory_sample(
            csv_path,
            vehicle_col=vehicle_col,
            target_rows=int(sample_rows),
            seed=int(data_config["sample_seed"]),
            chunk_rows=int(data_config.get("csv_chunk_rows", 250_000)),
            location_col=active_location_col,
        )
    elif input_is_excel(csv_path):
        frame = pd.read_excel(csv_path, engine="openpyxl")
    else:
        frame = pd.read_csv(csv_path)

    derivable = set()
    if data_config.get("kinematics", {}).get("derive_missing", False):
        derivable = {"v_Vel", "v_Acc"}
    required = set(features) | {vehicle_col, frame_col}
    missing = sorted((required - set(frame.columns)) - derivable)
    if missing:
        raise ValueError(f"Missing required NGSIM columns: {', '.join(missing)}")

    numeric_candidates = [column for column in required if column in frame.columns]
    for optional in (data_config.get("time_column"), "Time_Headway", "Space_Headway"):
        if optional and optional in frame.columns:
            numeric_candidates.append(optional)
    for column in dict.fromkeys(numeric_candidates):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=[vehicle_col, frame_col]).copy()
    frame[vehicle_col] = frame[vehicle_col].astype(np.int64)
    frame[frame_col] = frame[frame_col].astype(np.int64)
    frame = add_trajectory_segments(frame, data_config)
    group_columns = trajectory_group_columns(frame, data_config)
    derive_missing_kinematics(frame, data_config)

    missing_after_derivation = sorted(required - set(frame.columns))
    if missing_after_derivation:
        raise ValueError(
            "Missing required NGSIM columns after kinematic derivation: "
            + ", ".join(missing_after_derivation)
        )

    interpolate_columns = [column for column in features if column in frame.columns]
    for column in ("Time_Headway", "Space_Headway"):
        if column in frame.columns:
            interpolate_columns.append(column)
    frame[interpolate_columns] = frame.groupby(group_columns, sort=False)[
        interpolate_columns
    ].transform(lambda group: group.interpolate(limit_direction="both"))
    frame = frame.dropna(subset=features)

    source = data_config["labeling"].get("source", "existing_or_heuristic")
    if label_col in frame.columns and source in {"existing", "existing_or_heuristic"}:
        frame[label_col] = coerce_risk_classes(frame[label_col])
        frame = frame.dropna(subset=[label_col])
        frame[label_col] = frame[label_col].astype(np.int64)
    elif source in {"heuristic", "existing_or_heuristic"}:
        frame[label_col] = assign_risk_labels(frame, data_config["labeling"])
    else:
        raise ValueError(f"Unsupported labeling source: {source}")

    return frame.reset_index(drop=True)


def read_complete_trajectory_sample(
    csv_path: str | Path,
    vehicle_col: str,
    target_rows: int,
    seed: int,
    chunk_rows: int,
    location_col: str | None = None,
) -> pd.DataFrame:
    """Read an approximately sized sample without loading the full CSV at once."""

    if input_is_excel(csv_path):
        frame = pd.read_excel(csv_path, engine="openpyxl")
        if len(frame) <= target_rows:
            return frame
        return sample_complete_vehicles(
            frame,
            vehicle_col=vehicle_col,
            target_rows=target_rows,
            seed=seed,
            location_col=location_col,
        )

    usecols = [vehicle_col] if location_col is None else [location_col, vehicle_col]
    counts: dict[int | tuple[str, int], int] = {}
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_rows):
        vehicle_ids = pd.to_numeric(chunk[vehicle_col], errors="coerce")
        valid = vehicle_ids.notna()
        if location_col is not None:
            valid &= chunk[location_col].notna()
            keys = zip(
                chunk.loc[valid, location_col].astype(str),
                vehicle_ids.loc[valid].astype(np.int64),
            )
        else:
            keys = vehicle_ids.loc[valid].astype(np.int64)
        for raw_key, count in pd.Series(list(keys)).value_counts().items():
            key = raw_key if isinstance(raw_key, tuple) else int(raw_key)
            counts[key] = counts.get(key, 0) + int(count)
    if not counts:
        raise ValueError(f"No valid {vehicle_col} values found in {csv_path}")

    trajectory_ids = sorted(counts, key=repr)
    order = np.random.default_rng(seed).permutation(len(trajectory_ids))
    selected: set[int | tuple[str, int]] = set()
    selected_rows = 0
    for index in order:
        key = trajectory_ids[int(index)]
        selected.add(key)
        selected_rows += counts[key]
        if selected_rows >= target_rows:
            break

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
        numeric_ids = pd.to_numeric(chunk[vehicle_col], errors="coerce")
        if location_col is None:
            mask = numeric_ids.isin(selected)
        else:
            keys = pd.Series(
                list(zip(chunk[location_col].astype(str), numeric_ids)), index=chunk.index
            )
            mask = keys.isin(selected)
        selected_chunk = chunk[mask]
        if not selected_chunk.empty:
            chunks.append(selected_chunk)
    if not chunks:
        raise ValueError("Vehicle sampling selected no readable rows")
    return pd.concat(chunks, ignore_index=True)


def sample_complete_vehicles(
    frame: pd.DataFrame,
    vehicle_col: str,
    target_rows: int,
    seed: int,
    location_col: str | None = None,
) -> pd.DataFrame:
    """Sample whole vehicle trajectories so temporal windows remain intact."""

    if location_col and location_col in frame.columns:
        row_keys = pd.Series(
            list(zip(frame[location_col].astype(str), frame[vehicle_col])), index=frame.index
        )
    else:
        row_keys = frame[vehicle_col]
    counts = row_keys.value_counts(sort=False)
    trajectory_ids = list(counts.index)
    order = np.random.default_rng(seed).permutation(len(trajectory_ids))
    selected: list[object] = []
    total = 0
    for index in order:
        trajectory_id = trajectory_ids[int(index)]
        selected.append(trajectory_id)
        total += int(counts.loc[trajectory_id])
        if total >= target_rows:
            break
    return frame[row_keys.isin(selected)].copy()


def build_proximity_edges(
    xy: np.ndarray, radius_m: float, coordinate_scale_to_m: float
) -> tuple[np.ndarray, np.ndarray]:
    positions_m = np.asarray(xy, dtype=np.float32) * float(coordinate_scale_to_m)
    deltas = positions_m[:, None, :] - positions_m[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    mask = (distances > 0.0) & (distances < float(radius_m))
    source, target = np.nonzero(mask)
    edge_index = np.vstack([source, target]).astype(np.int64)
    if source.size:
        inverse_distance = 1.0 / np.maximum(distances[source, target], 1e-6)
        edge_attr = inverse_distance.astype(np.float32).reshape(-1, 1)
    else:
        edge_attr = np.empty((0, 1), dtype=np.float32)
    return edge_index, edge_attr


def build_aligned_graph_windows(frame: pd.DataFrame, data_config: dict) -> list[NumpyGraphSample]:
    """Create graph snapshots whose nodes have aligned 15-step histories."""

    features = list(data_config["feature_columns"])
    sequence_length = int(data_config["sequence_length"])
    vehicle_col = data_config["vehicle_column"]
    frame_col = data_config["frame_column"]
    label_col = data_config["label_column"]
    require_consecutive = bool(data_config.get("require_consecutive_frames", True))

    group_columns = trajectory_group_columns(frame, data_config)
    location_col = data_config.get("location_column")
    use_location = bool(location_col and location_col in frame.columns)
    by_frame: dict[tuple[str, int] | int, list[tuple[int, np.ndarray, int]]] = {}
    frame_ids_by_graph: dict[tuple[str, int] | int, int] = {}
    time_col = data_config.get("time_column")
    use_time = bool(time_col and time_col in frame.columns)
    for _, trajectory in frame.groupby(group_columns, sort=False):
        vehicle_id = int(trajectory[vehicle_col].iloc[0])
        location = trajectory[location_col].iloc[0] if use_location else None
        trajectory = trajectory.sort_values(frame_col, kind="stable")
        values = trajectory[features].to_numpy(dtype=np.float32)
        frame_ids = trajectory[frame_col].to_numpy(dtype=np.int64)
        graph_times = trajectory[time_col].to_numpy(dtype=np.int64) if use_time else frame_ids
        labels = trajectory[label_col].to_numpy(dtype=np.int64)
        for end in range(sequence_length - 1, len(trajectory)):
            start = end - sequence_length + 1
            window_frames = frame_ids[start : end + 1]
            if require_consecutive and not np.all(np.diff(window_frames) == 1):
                continue
            final_frame = int(frame_ids[end])
            final_time = int(graph_times[end])
            graph_key = (str(location), final_time) if use_location else final_time
            frame_ids_by_graph[graph_key] = final_frame
            by_frame.setdefault(graph_key, []).append(
                (int(vehicle_id), values[start : end + 1].copy(), int(labels[end]))
            )

    candidate_frames = [
        graph_key
        for graph_key in sorted(
            by_frame,
            key=lambda key: (key[1], key[0]) if isinstance(key, tuple) else (key, ""),
        )
        if len(by_frame[graph_key]) >= int(data_config["min_graph_nodes"])
    ]
    max_graphs = data_config.get("max_graphs")
    if max_graphs is not None and len(candidate_frames) > int(max_graphs):
        indices = np.linspace(0, len(candidate_frames) - 1, int(max_graphs), dtype=int)
        candidate_frames = [candidate_frames[index] for index in indices]

    reduction = data_config.get("graph_label_reduction", "max")
    samples: list[NumpyGraphSample] = []
    for graph_key in candidate_frames:
        records = sorted(by_frame[graph_key], key=lambda item: item[0])
        frame_id = frame_ids_by_graph[graph_key]
        vehicle_ids = np.asarray([item[0] for item in records], dtype=np.int64)
        sequences = np.stack([item[1] for item in records]).astype(np.float32)
        node_labels = np.asarray([item[2] for item in records], dtype=np.int64)
        x = sequences[:, -1, :].copy()
        edge_index, edge_attr = build_proximity_edges(
            x[:, :2],
            radius_m=float(data_config["graph_radius_m"]),
            coordinate_scale_to_m=float(data_config["coordinate_scale_to_m"]),
        )
        if reduction == "max":
            graph_label = int(node_labels.max())
        elif reduction == "mode":
            graph_label = int(np.bincount(node_labels, minlength=3).argmax())
        else:
            raise ValueError(f"Unsupported graph_label_reduction: {reduction}")
        samples.append(
            NumpyGraphSample(
                x=x,
                sequences=sequences,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=graph_label,
                frame_id=frame_id,
                vehicle_ids=vehicle_ids,
            )
        )
    if not samples:
        raise ValueError(
            "No aligned graph windows were produced. Check sequence length, frame continuity, "
            "and min_graph_nodes."
        )
    return samples


def chronological_split_codes(num_samples: int, split_config: dict) -> np.ndarray:
    gap = int(split_config.get("gap_frames", 0))
    usable = num_samples - (2 * gap)
    if usable < 3:
        raise ValueError("Not enough graph windows for train/validation/test after split gaps")
    train_count = max(1, int(usable * float(split_config["train"])))
    validation_count = max(1, int(usable * float(split_config["validation"])))
    test_count = usable - train_count - validation_count
    if test_count < 1:
        validation_count = max(1, validation_count - (1 - test_count))
        test_count = usable - train_count - validation_count
    codes = np.full(num_samples, -1, dtype=np.int8)
    train_end = train_count
    validation_start = train_end + gap
    validation_end = validation_start + validation_count
    test_start = validation_end + gap
    codes[:train_end] = 0
    codes[validation_start:validation_end] = 1
    codes[test_start : test_start + test_count] = 2
    return codes


def fit_minmax(samples: Sequence[NumpyGraphSample], split_codes: np.ndarray) -> dict[str, list]:
    train_sequences = [sample.sequences for sample, code in zip(samples, split_codes) if code == 0]
    if not train_sequences:
        raise ValueError("Training split is empty")
    stacked = np.concatenate(train_sequences, axis=0).reshape(-1, train_sequences[0].shape[-1])
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    scale = np.where(maximum > minimum, maximum - minimum, 1.0)
    return {
        "minimum": minimum.astype(float).tolist(),
        "maximum": maximum.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }


def apply_minmax(samples: Sequence[NumpyGraphSample], scaler: dict[str, list]) -> None:
    minimum = np.asarray(scaler["minimum"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    for sample in samples:
        sample.sequences = (sample.sequences - minimum) / scale
        sample.x = sample.sequences[:, -1, :].copy()


def save_graph_dataset(
    samples: Sequence[NumpyGraphSample], split_codes: np.ndarray, output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_offsets = [0]
    edge_offsets = [0]
    for sample in samples:
        node_offsets.append(node_offsets[-1] + len(sample.x))
        edge_offsets.append(edge_offsets[-1] + sample.edge_index.shape[1])
    np.savez_compressed(
        output_path,
        x=np.concatenate([sample.x for sample in samples], axis=0),
        sequences=np.concatenate([sample.sequences for sample in samples], axis=0),
        edge_index=np.concatenate([sample.edge_index for sample in samples], axis=1),
        edge_attr=np.concatenate([sample.edge_attr for sample in samples], axis=0),
        node_offsets=np.asarray(node_offsets, dtype=np.int64),
        edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
        labels=np.asarray([sample.y for sample in samples], dtype=np.int64),
        frame_ids=np.asarray([sample.frame_id for sample in samples], dtype=np.int64),
        vehicle_ids=np.concatenate([sample.vehicle_ids for sample in samples]),
        split_codes=np.asarray(split_codes, dtype=np.int8),
    )


class GraphWindowDataset(Dataset[GraphSample]):
    def __init__(self, path: str | Path, split: str | None = None):
        archive = np.load(Path(path), allow_pickle=False)
        self.arrays = {key: archive[key] for key in archive.files}
        code_by_name = {"train": 0, "validation": 1, "test": 2}
        if split is None:
            self.indices = np.arange(len(self.arrays["labels"]), dtype=np.int64)
        else:
            if split not in code_by_name:
                raise ValueError(f"Unknown split: {split}")
            self.indices = np.flatnonzero(self.arrays["split_codes"] == code_by_name[split])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> GraphSample:
        sample_index = int(self.indices[index])
        node_start, node_end = self.arrays["node_offsets"][sample_index : sample_index + 2]
        edge_start, edge_end = self.arrays["edge_offsets"][sample_index : sample_index + 2]
        edge_index = self.arrays["edge_index"][:, edge_start:edge_end].copy()
        return GraphSample(
            x=torch.from_numpy(self.arrays["x"][node_start:node_end]).float(),
            sequences=torch.from_numpy(self.arrays["sequences"][node_start:node_end]).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(self.arrays["edge_attr"][edge_start:edge_end]).float(),
            y=torch.tensor(int(self.arrays["labels"][sample_index]), dtype=torch.long),
            frame_id=torch.tensor(int(self.arrays["frame_ids"][sample_index]), dtype=torch.long),
            vehicle_ids=torch.from_numpy(
                self.arrays["vehicle_ids"][node_start:node_end].copy()
            ).long(),
        )

    @property
    def labels(self) -> np.ndarray:
        return self.arrays["labels"][self.indices].copy()

    def split_indices(self, split: str) -> np.ndarray:
        code_by_name = {"train": 0, "validation": 1, "test": 2}
        if split not in code_by_name:
            raise ValueError(f"Unknown split: {split}")
        return np.flatnonzero(self.arrays["split_codes"] == code_by_name[split])


def collate_graphs(samples: Iterable[GraphSample]) -> GraphBatch:
    samples = list(samples)
    if not samples:
        raise ValueError("Cannot collate an empty graph batch")
    xs, sequences, edges, edge_attrs, batches, vehicle_ids = [], [], [], [], [], []
    labels, frame_ids = [], []
    node_offset = 0
    for graph_index, sample in enumerate(samples):
        xs.append(sample.x)
        sequences.append(sample.sequences)
        edges.append(sample.edge_index + node_offset)
        edge_attrs.append(sample.edge_attr)
        batches.append(torch.full((sample.x.shape[0],), graph_index, dtype=torch.long))
        vehicle_ids.append(sample.vehicle_ids)
        labels.append(sample.y)
        frame_ids.append(sample.frame_id)
        node_offset += sample.x.shape[0]
    return GraphBatch(
        x=torch.cat(xs, dim=0),
        sequences=torch.cat(sequences, dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0),
        batch=torch.cat(batches, dim=0),
        y=torch.stack(labels),
        frame_ids=torch.stack(frame_ids),
        vehicle_ids=torch.cat(vehicle_ids, dim=0),
    )
