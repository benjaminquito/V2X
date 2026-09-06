from pathlib import Path

import numpy as np
import pandas as pd

from v2x_risk.config import load_config
from v2x_risk.data import (
    assign_risk_labels,
    build_aligned_graph_windows,
    build_proximity_edges,
    chronological_split_codes,
    read_complete_trajectory_sample,
)
from v2x_risk.smoke import write_synthetic_ngsim

ROOT = Path(__file__).resolve().parents[1]


def test_risk_label_thresholds_are_ordered() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    frame = pd.DataFrame(
        {
            "Time_Headway": [3.0, 1.5, 0.5],
            "Space_Headway": [50.0, 25.0, 10.0],
            "v_Acc": [0.0, -7.0, -12.0],
        }
    )
    assert assign_risk_labels(frame, config["data"]["labeling"]).tolist() == [0, 1, 2]


def test_proximity_graph_is_directed_and_excludes_far_nodes() -> None:
    xy = np.asarray([[0.0, 0.0], [3.0, 4.0], [30.0, 0.0]], dtype=np.float32)
    edge_index, edge_attr = build_proximity_edges(xy, radius_m=20.0, coordinate_scale_to_m=1.0)
    assert edge_index.T.tolist() == [[0, 1], [1, 0]]
    np.testing.assert_allclose(edge_attr[:, 0], [0.2, 0.2])


def test_aligned_windows_have_expected_shape(tmp_path: Path) -> None:
    csv_path = write_synthetic_ngsim(tmp_path / "smoke.csv", vehicles=4, frames=20)
    frame = pd.read_csv(csv_path)
    config = load_config(ROOT / "configs/smoke.yaml")
    samples = build_aligned_graph_windows(frame, config["data"])
    assert samples[0].sequences.shape == (4, 15, 4)
    assert samples[0].x.shape == (4, 4)
    assert samples[0].frame_id == 14


def test_chunked_sampling_preserves_complete_trajectories(tmp_path: Path) -> None:
    csv_path = write_synthetic_ngsim(tmp_path / "smoke.csv", vehicles=5, frames=20)
    sample = read_complete_trajectory_sample(
        csv_path, vehicle_col="Vehicle_ID", target_rows=35, seed=9, chunk_rows=17
    )
    counts = sample.groupby("Vehicle_ID").size()
    assert len(sample) >= 35
    assert counts.eq(20).all()


def test_chronological_split_includes_gaps() -> None:
    codes = chronological_split_codes(
        40, {"train": 0.70, "validation": 0.15, "test": 0.15, "gap_frames": 2}
    )
    assert set(codes) == {-1, 0, 1, 2}
    assert np.flatnonzero(codes == 0).max() < np.flatnonzero(codes == 1).min()
    assert np.flatnonzero(codes == 1).max() < np.flatnonzero(codes == 2).min()
