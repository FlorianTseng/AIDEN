"""Evaluate checkpoints produced by ``ablation/run_train.py``."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate as production_eval  # noqa: E402
from ablation.data import AIDENRapidAblationDataset  # noqa: E402
from ablation.variants import AIDENAblation  # noqa: E402


production_eval.Aiden = AIDENAblation
production_eval.CHGCARRhoDataset = AIDENRapidAblationDataset


if __name__ == "__main__":
    production_eval.main()
