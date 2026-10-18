"""Collect completed ablation test summaries into one CSV and Markdown table."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation.config import ABLATION_VARIANTS  # noqa: E402


RESULTS = ROOT / "results_l4_2k_100k"
RUNS = ROOT / "runs_l4_2k_100k"


def main() -> None:
    rows = []
    missing = []
    for variant in ABLATION_VARIANTS:
        summary_path = RESULTS / variant / "summary.json"
        config_path = RUNS / variant / "config.json"
        if not summary_path.exists() or not config_path.exists():
            missing.append(variant)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_args = config["model_args"]
        log_text = (RUNS / variant / "train.log").read_text(encoding="utf-8")
        parameter_match = re.search(
            r"params_encoder=(\d+) params_decoder=(\d+) params_total=(\d+)",
            log_text,
        )
        if parameter_match is None:
            raise RuntimeError(f"parameter counts not found for {variant}")
        rows.append(
            {
                "variant": variant,
                "params_encoder": int(parameter_match.group(1)),
                "params_decoder": int(parameter_match.group(2)),
                "params_total": int(parameter_match.group(3)),
                "num_structures": int(summary["num_structures"]),
                "mae": float(summary["mae"]),
                "nmae": float(summary["nmae"]),
                "global_nmae": float(summary["global_nmae"]),
                "checkpoint": summary["checkpoint"],
                "model_name": model_args["model_name"],
            }
        )

    output_csv = ROOT / "ablation_results_l4_2k_100k.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "variant",
            "params_encoder",
            "params_decoder",
            "params_total",
            "num_structures",
            "mae",
            "nmae",
            "global_nmae",
            "checkpoint",
            "model_name",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("| Variant | Params | Structures | MAE | NMAE (%) | Global NMAE (%) |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['variant']} | {row['params_total']} | "
            f"{row['num_structures']} | "
            f"{row['mae']:.8g} | {100.0 * row['nmae']:.6f} | "
            f"{100.0 * row['global_nmae']:.6f} |"
        )
    if missing:
        print("\nMissing variants: " + ", ".join(missing))
    print(f"\nCSV: {output_csv}")


if __name__ == "__main__":
    main()
