from pathlib import Path

from v2x_risk.smoke import run_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_smoke_pipeline(tmp_path: Path) -> None:
    result = run_smoke(ROOT / "configs/smoke.yaml", tmp_path)
    assert result["preprocess"]["graph_windows"] > 0
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "test_metrics.json").is_file()
    assert result["evaluation"]["status"] == "regenerated"
