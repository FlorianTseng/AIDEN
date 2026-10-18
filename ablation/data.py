"""Deterministic PBE data view for the L=4 rapid ablation study."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from aiden.data import CHGCARRhoDataset, PackedH5Record


FULL_PBE_MAX_FILES = 150_000
TRAIN_STRUCTURES = 2_000
SPLIT_SEED = 42
SUBSET_SEED = 42
VALIDATION_STRUCTURES = 256
TEST_STRUCTURES = 2_000


def _record_identifier(record: object) -> str:
    if isinstance(record, PackedH5Record):
        return f"{record.shard_path.name}:{record.group_key}:{record.order}"
    return str(Path(record))


class AIDENRapidAblationDataset(CHGCARRhoDataset):
    """Use 2,000 random members of the existing full-PBE training split.

    The parent class first constructs exactly the production seed-42 80/10/10
    split over the full packed dataset.  Only then is the training split
    reduced.  Validation and test retain their original full-PBE split so
    callers can choose fixed evaluation subsets without contaminating train.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        max_files = kwargs.get("max_files")
        if max_files not in (None, FULL_PBE_MAX_FILES):
            raise ValueError(
                "rapid ablation data must be split from the full PBE manifest; "
                f"got max_files={max_files!r}"
            )
        kwargs["max_files"] = FULL_PBE_MAX_FILES
        super().__init__(*args, **kwargs)

        if self.split.lower() == "train":
            if len(self.files) < TRAIN_STRUCTURES:
                raise RuntimeError(
                    f"full PBE train split has only {len(self.files)} records; "
                    f"cannot select {TRAIN_STRUCTURES}"
                )
            rng = np.random.default_rng(SUBSET_SEED)
            indices = np.sort(
                rng.choice(len(self.files), size=TRAIN_STRUCTURES, replace=False)
            )
            self.files = [self.files[int(index)] for index in indices]

        identifiers = "\n".join(_record_identifier(record) for record in self.files)
        self.selection_sha256 = hashlib.sha256(identifiers.encode("utf-8")).hexdigest()
        print(
            f"[ablation-data] split={self.split} structures={len(self.files)} "
            f"selection_sha256={self.selection_sha256}",
            flush=True,
        )


__all__ = [
    "AIDENRapidAblationDataset",
    "FULL_PBE_MAX_FILES",
    "SPLIT_SEED",
    "SUBSET_SEED",
    "TEST_STRUCTURES",
    "TRAIN_STRUCTURES",
    "VALIDATION_STRUCTURES",
]
