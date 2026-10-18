"""Launch QM9 counterparts of the MP per-sample NMAE and charge reports.

The report engines live in ``violin-infer`` and are deliberately reused here:
this wrapper supplies the QM9 data root and the five QM9 best-NMAE checkpoints,
then writes all results into ``qm9-violin``.  That keeps the two datasets on
the same metric, split, checkpoint, and CSV conventions.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
QM9_DATA_ROOT = REPO_ROOT / "data" / "aiden-qm9"
ELEMENT_CSV = REPO_ROOT / "data" / "elem-des.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        choices=["nmae", "charge", "all"],
        default="all",
        help="Which CSV report(s) to generate.",
    )
    parser.add_argument("--data-root", type=Path, default=QM9_DATA_ROOT)
    parser.add_argument("--elem-csv", type=Path, default=ELEMENT_CSV)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite through to the selected report engine(s).",
    )
    return parser.parse_args()


def qm9_checkpoint_arguments() -> list[str]:
    arguments: list[str] = []
    for l_value in range(5):
        checkpoint = REPO_ROOT / "runs" / f"aiden-qm9-l{l_value}" / "best_nmae.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing QM9 l={l_value} checkpoint: {checkpoint}")
        arguments.extend(["--checkpoint", f"{l_value}={checkpoint}"])
    return arguments


def run_report(args: argparse.Namespace, report: str) -> None:
    if report == "nmae":
        engine = REPO_ROOT / "violin-infer" / "infer_mp_lmax_nmae.py"
        output = SCRIPT_DIR / "qm9_train_val_lmax_nmae.csv"
    elif report == "charge":
        engine = REPO_ROOT / "violin-infer" / "infer_mp_lmax_charge_5000.py"
        output = SCRIPT_DIR / "qm9_train_val_lmax_charge_5000.csv"
    else:
        raise ValueError(f"unknown report: {report}")

    command = [
        sys.executable,
        str(engine),
        "--data-root",
        str(args.data_root),
        "--elem-csv",
        str(args.elem_csv),
        "--output",
        str(output),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        *qm9_checkpoint_arguments(),
    ]
    if args.overwrite:
        command.append("--overwrite")
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"QM9 packed data root does not exist: {args.data_root}")
    if not args.elem_csv.is_file():
        raise FileNotFoundError(f"element CSV does not exist: {args.elem_csv}")

    reports = ("nmae", "charge") if args.report == "all" else (args.report,)
    for report in reports:
        run_report(args, report)


if __name__ == "__main__":
    main()
