from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def generate_run_report(config: dict, completed_at: datetime | None = None) -> dict:
    data_config = config["data"]
    training_config = config["training"]
    manifest_path = Path(data_config["manifest_path"])
    metrics_path = Path(training_config["metrics_path"])
    checkpoint_path = Path(training_config["checkpoint_path"])
    test_metrics_path = checkpoint_path.with_name("test_metrics.json")

    manifest = read_json(manifest_path)
    metrics = read_json(metrics_path)
    test_payload = read_json(test_metrics_path)
    branches = test_payload["branches"]
    fused = branches["fused"]
    split_counts = manifest["split_counts"]
    timestamp = completed_at or datetime.now(timezone.utc)
    timestamp_text = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    output_dir = metrics_path.parent
    summary_path = output_dir / "run_summary.txt"
    report_path = output_dir / "run_report.html"

    summary = "\n".join(
        [
            "V2X RUN COMPLETED SUCCESSFULLY",
            f"Completed: {timestamp_text}",
            "",
            "DATA",
            f"Cleaned rows: {manifest['cleaned_rows']:,}",
            f"Graph windows: {manifest['graph_windows']:,}",
            (
                "Split: "
                f"{split_counts['train']:,} train / "
                f"{split_counts['validation']:,} validation / "
                f"{split_counts['test']:,} test"
            ),
            f"Kinematics: {manifest['kinematics_source']}",
            f"Labels: {manifest['labeling_source']}",
            "",
            "TEST RESULTS",
            f"Fused accuracy: {percent(fused['accuracy'])}",
            f"Fused macro F1: {fused['macro_f1']:.4f}",
            f"Fused weighted F1: {fused['weighted_f1']:.4f}",
            f"Temporal accuracy: {percent(branches['temporal']['accuracy'])}",
            f"Spatial accuracy: {percent(branches['spatial']['accuracy'])}",
            f"Best epoch: {metrics['best_epoch']}",
            "",
            "OUTPUTS",
            f"Visual report: {report_path.resolve()}",
            f"Metrics: {metrics_path.resolve()}",
            f"Test metrics: {test_metrics_path.resolve()}",
            f"Predictions: {(output_dir / 'test_predictions.csv').resolve()}",
            "",
            (
                "Interpretation: successful execution only; derived kinematics or heuristic "
                "labels are not directly comparable to the manuscript results."
            ),
        ]
    )
    summary_path.write_text(summary + "\n", encoding="utf-8")

    branch_rows = []
    for name in ("temporal", "spatial", "fused"):
        values = branches[name]
        branch_rows.append(
            f"""
            <article class="branch">
              <h3>{html.escape(name.title())}</h3>
              <div class="measure"><span>Accuracy</span><strong>{percent(values["accuracy"])}</strong></div>
              <div class="track" role="img" aria-label="{name.title()} accuracy {percent(values["accuracy"])}">
                <span style="width:{100.0 * float(values["accuracy"]):.2f}%"></span>
              </div>
              <div class="measure"><span>Macro F1</span><strong>{values["macro_f1"]:.4f}</strong></div>
              <div class="track secondary" role="img" aria-label="{name.title()} macro F1 {values["macro_f1"]:.4f}">
                <span style="width:{100.0 * float(values["macro_f1"]):.2f}%"></span>
              </div>
            </article>
            """
        )

    fusion_weights = fused.get("mean_fusion_weights", {})
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2X run results</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4f7fb; color: #172033; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }}
    .status {{ background: #e9f8ef; border-left: 6px solid #18864b; padding: 22px 24px; border-radius: 12px; }}
    .status h1 {{ margin: 0; color: #116638; font-size: clamp(1.6rem, 4vw, 2.4rem); }}
    .status p {{ margin: 8px 0 0; color: #275a3e; }}
    h2 {{ margin-top: 32px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .stat, .branch {{ background: #fff; border: 1px solid #d9e1ec; border-radius: 12px; padding: 18px; }}
    .stat span {{ color: #5d6b82; display: block; font-size: .9rem; }}
    .stat strong {{ display: block; margin-top: 6px; font-size: 1.55rem; }}
    .branch h3 {{ margin: 0 0 18px; }}
    .measure {{ display: flex; justify-content: space-between; gap: 12px; margin: 12px 0 6px; }}
    .track {{ height: 10px; background: #e5eaf1; border-radius: 999px; overflow: hidden; }}
    .track span {{ display: block; height: 100%; background: #3066d6; }}
    .track.secondary span {{ background: #7a5bc7; }}
    .details {{ width: 100%; border-collapse: collapse; background: #fff; }}
    .details th, .details td {{ text-align: left; padding: 11px 12px; border-bottom: 1px solid #d9e1ec; }}
    .details th {{ width: 34%; color: #5d6b82; font-weight: 500; }}
    .note {{ margin-top: 28px; padding: 16px 18px; background: #fff6dc; color: #664b00; border-radius: 10px; }}
    .files a {{ color: #2458bb; }}
    @media (max-width: 680px) {{ .grid {{ grid-template-columns: 1fr; }} .details th {{ width: auto; }} }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #111722; color: #edf2fa; }}
      .status {{ background: #133523; border-color: #4bd484; }}
      .status h1, .status p {{ color: #b7f3ce; }}
      .stat, .branch, .details {{ background: #1a2230; border-color: #344054; }}
      .stat span, .details th {{ color: #aab6c8; }}
      .details th, .details td {{ border-color: #344054; }}
      .track {{ background: #354052; }}
      .note {{ background: #3b3013; color: #ffe9a8; }}
      .files a {{ color: #8db4ff; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="status" aria-label="Run status">
      <h1>✓ Completed successfully</h1>
      <p>{html.escape(timestamp_text)} · Best checkpoint selected at epoch {metrics["best_epoch"]}</p>
    </section>

    <h2>Processed data</h2>
    <section class="grid" aria-label="Dataset summary">
      <div class="stat"><span>Cleaned observations</span><strong>{manifest["cleaned_rows"]:,}</strong></div>
      <div class="stat"><span>Aligned graph windows</span><strong>{manifest["graph_windows"]:,}</strong></div>
      <div class="stat"><span>Test graphs</span><strong>{split_counts["test"]:,}</strong></div>
    </section>

    <h2>Held-out test performance</h2>
    <section class="grid" aria-label="Branch performance">{"".join(branch_rows)}</section>

    <h2>Run details</h2>
    <table class="details">
      <tr><th>Train / validation / test</th><td>{split_counts["train"]:,} / {split_counts["validation"]:,} / {split_counts["test"]:,}</td></tr>
      <tr><th>Fused weighted F1</th><td>{fused["weighted_f1"]:.4f}</td></tr>
      <tr><th>Mean fusion attention</th><td>Temporal {percent(fusion_weights.get("temporal", 0))} · Spatial {percent(fusion_weights.get("spatial", 0))}</td></tr>
      <tr><th>Kinematics</th><td>{html.escape(manifest["kinematics_source"])}</td></tr>
      <tr><th>Labels</th><td>{html.escape(manifest["labeling_source"])}</td></tr>
      <tr><th>Input SHA-256</th><td><code>{html.escape(manifest["raw_sha256"])}</code></td></tr>
    </table>

    <p class="note"><strong>Run status and model quality are different.</strong> The pipeline completed, but these results use derived kinematics and heuristic labels. They are execution-test results and are not directly comparable with the manuscript.</p>

    <h2>Result files</h2>
    <ul class="files">
      <li><a href="metrics.json">Training and branch metrics</a></li>
      <li><a href="test_metrics.json">Independent test metrics</a></li>
      <li><a href="test_predictions.csv">Per-graph test predictions</a></li>
      <li><a href="run_summary.txt">Text completion summary</a></li>
    </ul>
  </main>
</body>
</html>
"""
    report_path.write_text(report, encoding="utf-8")
    return {"summary": summary, "summary_path": summary_path, "report_path": report_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a visual summary of a completed V2X run")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_run_report(load_config(args.config))
    print(result["summary"])


if __name__ == "__main__":
    main()
