from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from v2x_risk.config import load_config
from v2x_risk.report import generate_run_report
from v2x_risk.reproducibility import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_run_report_shows_success_and_results(tmp_path: Path) -> None:
    config = deepcopy(load_config(ROOT / "configs/default.yaml"))
    config["data"]["manifest_path"] = str(tmp_path / "manifest.json")
    config["training"]["metrics_path"] = str(tmp_path / "metrics.json")
    config["training"]["checkpoint_path"] = str(tmp_path / "best_model.pt")
    write_json(
        config["data"]["manifest_path"],
        {
            "cleaned_rows": 100,
            "graph_windows": 20,
            "split_counts": {"train": 12, "validation": 4, "test": 4},
            "kinematics_source": "derived test values",
            "labeling_source": "configured test labels",
            "raw_sha256": "a" * 64,
        },
    )
    write_json(config["training"]["metrics_path"], {"best_epoch": 2})
    write_json(
        tmp_path / "test_metrics.json",
        {
            "branches": {
                name: {
                    "accuracy": accuracy,
                    "macro_f1": accuracy / 2,
                    "weighted_f1": accuracy / 2,
                    **(
                        {"mean_fusion_weights": {"temporal": 0.4, "spatial": 0.6}}
                        if name == "fused"
                        else {}
                    ),
                }
                for name, accuracy in {"temporal": 0.5, "spatial": 0.4, "fused": 0.6}.items()
            }
        },
    )

    result = generate_run_report(
        config, completed_at=datetime(2026, 9, 6, 20, 0, tzinfo=timezone.utc)
    )

    summary = result["summary_path"].read_text(encoding="utf-8")
    report = result["report_path"].read_text(encoding="utf-8")
    assert "V2X RUN COMPLETED SUCCESSFULLY" in summary
    assert "Fused accuracy: 60.00%" in summary
    assert "Completed successfully" in report
    assert "Held-out test performance" in report
