"""Train Aiden: equivariant Cartesian ACE/TECE encoder + continuous GTO decoder."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from aiden.model import Aiden, AidenOutput
from aiden.data import CHGCARRhoDataset, collate_rho, graph_backend_name


TRAINING_OBJECTIVE = "aiden_mae_v1"


def log_message(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(
    batch_list: Sequence[Dict[str, object]],
    device: torch.device,
) -> List[Dict[str, object]]:
    moved_batches: List[Dict[str, object]] = []
    for batch in batch_list:
        moved: Dict[str, object] = {}
        for key, value in batch.items():
            moved[key] = (
                value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
        moved_batches.append(moved)
    return moved_batches


def structure_nmae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().sum() / target.abs().sum().clamp_min(1.0e-8)


def mae_objective(
    outputs: Sequence[AidenOutput],
    batch_list: Sequence[Dict[str, object]],
    density_scale: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Grid-averaged MAE objective (paper Eq. 3) with per-structure NMAE logging."""
    if len(outputs) != len(batch_list):
        raise ValueError(
            f"received {len(outputs)} predictions for {len(batch_list)} structures"
        )
    absolute_error_sum: torch.Tensor | None = None
    nmae_sum: torch.Tensor | None = None
    num_points = 0
    for output, batch in zip(outputs, batch_list):
        target = batch["target"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("batch target must be a tensor")
        target = target.to(output.pred)
        if output.pred.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(output.pred.shape)} != target shape "
                f"{tuple(target.shape)} for {batch.get('name', '<unknown>')}"
            )
        error_sum = (output.pred - target).abs().sum()
        sample_nmae = error_sum / target.abs().sum().clamp_min(1.0e-8)
        absolute_error_sum = (
            error_sum
            if absolute_error_sum is None
            else absolute_error_sum + error_sum
        )
        nmae_sum = sample_nmae if nmae_sum is None else nmae_sum + sample_nmae
        num_points += int(target.numel())

    if absolute_error_sum is None or nmae_sum is None or num_points == 0:
        raise RuntimeError("empty batch in MAE objective")
    scale = density_scale.to(absolute_error_sum).clamp_min(1.0e-8)
    loss = absolute_error_sum / (float(num_points) * scale)
    return loss, {
        "absolute_error_sum": float(absolute_error_sum.detach().cpu()),
        "nmae_sum": float(nmae_sum.detach().cpu()),
        "num_points": float(num_points),
        "num_structures": float(len(outputs)),
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.absolute_error_sum = 0.0
        self.nmae_sum = 0.0
        self.num_points = 0.0
        self.num_structures = 0.0

    def update(self, metrics: Dict[str, float]) -> None:
        self.absolute_error_sum += metrics["absolute_error_sum"]
        self.nmae_sum += metrics["nmae_sum"]
        self.num_points += metrics["num_points"]
        self.num_structures += metrics["num_structures"]

    def values(self, density_scale: float) -> Dict[str, float]:
        physical_mae = self.absolute_error_sum / max(self.num_points, 1.0)
        return {
            "loss": physical_mae / max(float(density_scale), 1.0e-8),
            "mae": physical_mae,
            "nmae": self.nmae_sum / max(self.num_structures, 1.0),
            "num_points": self.num_points,
            "num_structures": self.num_structures,
        }


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach(),
                alpha=1.0 - self.decay,
            )

    def state_dict(self) -> Dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        loaded = state.get("shadow")
        if not isinstance(loaded, dict):
            raise TypeError("EMA checkpoint does not contain a shadow state dict")
        missing = set(self.shadow).difference(loaded)
        if missing:
            raise RuntimeError(f"EMA checkpoint missing parameters: {sorted(missing)[:5]}")
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {
            name: loaded[name].detach().clone().to(reference)
            for name, reference in self.shadow.items()
        }

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        if self.backup:
            raise RuntimeError("EMA parameters are already applied")
        for name, parameter in model.named_parameters():
            self.backup[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            parameter.copy_(self.backup[name])
        self.backup = {}

    def merged_model_state(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        for name, value in self.shadow.items():
            state[name] = value.detach().clone()
        return state


def optimizer_lr_string(optimizer: torch.optim.Optimizer) -> str:
    return ",".join(
        f"{group.get('name', f'group{index}')}={float(group['lr']):.3e}"
        for index, group in enumerate(optimizer.param_groups)
    )


def estimate_target_std(
    dataset: CHGCARRhoDataset,
    max_structures: int,
) -> Tuple[float, float, int, int]:
    if len(dataset) == 0:
        raise RuntimeError("cannot estimate density statistics from an empty dataset")
    count = min(max(int(max_structures), 1), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    value_sum = 0.0
    square_sum = 0.0
    value_count = 0
    for index in indices.tolist():
        values = np.asarray(dataset.load_density_values(index), dtype=np.float64)
        value_sum += float(values.sum(dtype=np.float64))
        square_sum += float(np.square(values).sum(dtype=np.float64))
        value_count += int(values.size)
    mean = value_sum / max(value_count, 1)
    variance = max(square_sum / max(value_count, 1) - mean * mean, 1.0e-16)
    return mean, math.sqrt(variance), count, value_count


def evenly_spaced_subset(dataset, max_structures: int):
    if max_structures <= 0 or max_structures >= len(dataset):
        return dataset
    indices = np.linspace(
        0,
        len(dataset) - 1,
        int(max_structures),
        dtype=np.int64,
    ).tolist()
    return Subset(dataset, indices)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
    num_workers: int | None = None,
) -> DataLoader:
    worker_count = int(args.num_workers if num_workers is None else num_workers)
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": worker_count,
        "pin_memory": bool(args.device.startswith("cuda")),
        "collate_fn": collate_rho,
        "drop_last": False,
    }
    if worker_count > 0:
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return DataLoader(**kwargs)


class ReplayableBatchSchedule:
    """Deterministic shuffled batches with an explicitly consumed cursor."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        initial_epoch: int,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = max(int(initial_epoch), 0)
        self.batch_position = 0
        self._order = self._make_order()

    def _make_order(self) -> List[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    @property
    def num_batches(self) -> int:
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def current_indices(self) -> List[int]:
        start = self.batch_position * self.batch_size
        return self._order[start : start + self.batch_size]

    def remaining_batches(self) -> List[List[int]]:
        start = self.batch_position * self.batch_size
        return [
            self._order[offset : offset + self.batch_size]
            for offset in range(start, self.dataset_size, self.batch_size)
        ]

    def advance(self) -> None:
        self.batch_position += 1
        if self.batch_position >= self.num_batches:
            self.epoch += 1
            self.batch_position = 0
            self._order = self._make_order()


def is_shared_memory_worker_error(error: BaseException) -> bool:
    """Recognize the catchable parent-side form of a DataLoader SIGBUS."""
    messages: List[str] = []
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    message = "\n".join(messages)
    return (
        "bus error" in message
        or "shared memory" in message
        or (
            "dataloader worker" in message
            and "exited unexpectedly" in message
        )
    )


def shared_memory_status() -> str:
    try:
        stats = os.statvfs("/dev/shm")
    except OSError as error:
        return f"/dev/shm unavailable ({error})"
    total = stats.f_blocks * stats.f_frsize
    available = stats.f_bavail * stats.f_frsize
    gib = float(1024**3)
    return f"/dev/shm available={available / gib:.2f}GiB total={total / gib:.2f}GiB"


class ResilientTrainingBatches:
    """Use worker prefetch normally and replay one batch locally after SIGBUS."""

    _FAILURE_THRESHOLD = 3
    _FAILURE_WINDOW_STEPS = 1000
    _COOLDOWN_STEPS = 100

    def __init__(
        self,
        dataset,
        batch_size: int,
        args: argparse.Namespace,
        log_path: Path,
        initial_epoch: int,
    ) -> None:
        self.dataset = dataset
        self.args = args
        self.log_path = log_path
        self.fast_workers = int(args.num_workers)
        self.schedule = ReplayableBatchSchedule(
            dataset_size=len(dataset),
            batch_size=batch_size,
            seed=args.seed,
            initial_epoch=initial_epoch,
        )
        self.loader: DataLoader | None = None
        self.iterator = None
        self.active_workers = self.fast_workers
        self.failure_steps: List[int] = []
        self.cooldown_until_step = 0
        self._rebuild(self.fast_workers)

    def _shutdown(self) -> None:
        iterator = self.iterator
        self.iterator = None
        self.loader = None
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        del iterator
        gc.collect()

    def _rebuild(self, num_workers: int) -> None:
        self._shutdown()
        self.active_workers = int(num_workers)
        kwargs = {
            "dataset": self.dataset,
            "batch_sampler": self.schedule.remaining_batches(),
            "num_workers": self.active_workers,
            "pin_memory": bool(self.args.device.startswith("cuda")),
            "collate_fn": collate_rho,
        }
        if self.active_workers > 0:
            kwargs["prefetch_factor"] = int(self.args.prefetch_factor)
            kwargs["persistent_workers"] = bool(self.args.persistent_workers)
        self.loader = DataLoader(**kwargs)
        self.iterator = iter(self.loader)

    def _load_current_batch_in_main_process(self) -> List[Dict[str, object]]:
        indices = self.schedule.current_indices()
        return collate_rho([self.dataset[index] for index in indices])

    def next(self, step: int) -> Tuple[int, List[Dict[str, object]]]:
        if (
            self.active_workers == 0
            and self.fast_workers > 0
            and step > self.cooldown_until_step
        ):
            log_message(
                self.log_path,
                f"[dataloader-recovery] cooldown complete at step={step}; "
                f"restoring workers={self.fast_workers}",
            )
            self._rebuild(self.fast_workers)

        while True:
            batch_epoch = self.schedule.epoch
            try:
                assert self.iterator is not None
                cpu_batch = next(self.iterator)
            except StopIteration:
                # The consumed cursor has already advanced to the next epoch.
                self._rebuild(self.active_workers)
                continue
            except RuntimeError as error:
                if not is_shared_memory_worker_error(error):
                    raise

                failed_indices = self.schedule.current_indices()
                self.failure_steps = [
                    failed_step
                    for failed_step in self.failure_steps
                    if step - failed_step <= self._FAILURE_WINDOW_STEPS
                ]
                self.failure_steps.append(step)
                log_message(
                    self.log_path,
                    "[dataloader-recovery] worker/shared-memory failure "
                    f"before step={step} epoch={batch_epoch} "
                    f"batch_position={self.schedule.batch_position} "
                    f"indices={failed_indices[:4]}...; {shared_memory_status()}; "
                    "replaying this batch in the main process",
                )

                # The failed iterator cannot be reused.  Tear it down first so
                # its queued shared tensors no longer occupy /dev/shm, then
                # load exactly the next unconsumed batch without worker IPC.
                self._shutdown()
                cpu_batch = self._load_current_batch_in_main_process()
                names = [str(item.get("name", "<unknown>")) for item in cpu_batch[:3]]
                self.schedule.advance()

                if len(self.failure_steps) >= self._FAILURE_THRESHOLD:
                    self.cooldown_until_step = step + self._COOLDOWN_STEPS
                    next_workers = 0
                    log_message(
                        self.log_path,
                        "[dataloader-recovery] repeated failures detected; "
                        f"using workers=0 through step={self.cooldown_until_step}",
                    )
                    self.failure_steps.clear()
                else:
                    next_workers = self.fast_workers

                # Start prefetch for the following batch while the recovered
                # batch is being processed by the GPU.
                self._rebuild(next_workers)
                log_message(
                    self.log_path,
                    "[dataloader-recovery] replay succeeded "
                    f"names={names}; next_workers={next_workers}",
                )
                return batch_epoch, cpu_batch

            self.schedule.advance()
            return batch_epoch, cpu_batch

    def close(self) -> None:
        self._shutdown()


@torch.inference_mode()
def evaluate(
    model: Aiden,
    loader: DataLoader,
    device: torch.device,
    initial_guess_only: bool = False,
) -> Dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    for cpu_batch in loader:
        batch = move_batch_to_device(cpu_batch, device)
        outputs = (
            model.forward_initial_guess(batch)
            if initial_guess_only
            else model(batch)
        )
        _, metrics = mae_objective(outputs, batch, model.density_scale)
        accumulator.update(metrics)
    return accumulator.values(float(model.density_scale.detach().cpu()))


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        # Keep the previous checkpoint intact until the replacement has been
        # fully written and made durable.  This matters on network filesystems:
        # rename alone can otherwise expose a truncated PyTorch ZIP archive if
        # the job is killed shortly after torch.save returns.
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        # Opening the central directory is inexpensive and catches the exact
        # truncation mode that would later fail in torch.load.
        with zipfile.ZipFile(temporary, mode="r") as archive:
            if not archive.infolist():
                raise RuntimeError(f"checkpoint archive is empty: {temporary}")

        temporary.replace(path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            # The file itself is already durable; directory fsync is only an
            # additional rename-durability guarantee where it is supported.
            return
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                # Some network filesystems do not support fsync on a directory
                # descriptor.  The checkpoint file itself was already fsynced.
                pass
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint(
    path: Path,
    model: Aiden,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ModelEMA,
    args: argparse.Namespace,
    model_args: Dict[str, object],
    global_step: int,
    epoch: int,
    best_val_mae: float,
    best_val_nmae: float,
) -> None:
    atomic_torch_save(
        {
            "training_objective": TRAINING_OBJECTIVE,
            "model": model.state_dict(),
            "ema_model": ema.merged_model_state(model),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "model_args": model_args,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "best_val_mae": float(best_val_mae),
            "best_val_nmae": float(best_val_nmae),
        },
        path,
    )


def cycle_loader(loader: DataLoader):
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


def pretrain_initial_guess(
    model: Aiden,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    save_dir: Path,
    log_path: Path,
) -> None:
    if args.initial_guess_steps <= 0:
        log_message(log_path, "Initial-guess pretraining disabled by configuration")
        return

    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.initial_guess.parameters():
        parameter.requires_grad_(True)

    radial_correction_enabled = model.decoder.gto.use_radial_correction
    model.decoder.gto.use_radial_correction = False
    optimizer = torch.optim.Adam(
        model.decoder.initial_guess.parameters(),
        lr=float(args.initial_guess_lr),
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(args.initial_guess_restart_steps),
        T_mult=1,
    )
    iterator = cycle_loader(train_loader)
    window = MetricAccumulator()
    window_start = time.time()
    model.train()
    log_message(
        log_path,
        "Start one-centre density pretraining | "
        f"steps={args.initial_guess_steps} requested_probes={args.initial_guess_probes} "
        f"batch_size={args.initial_guess_batch_size} lr={args.initial_guess_lr:.3e} "
        "radial_correction=False",
    )

    for step in range(1, int(args.initial_guess_steps) + 1):
        _, cpu_batch = next(iterator)
        batch = move_batch_to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_initial_guess(batch)
        loss, metrics = mae_objective(outputs, batch, model.density_scale)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite initial-guess loss at pretrain step {step}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.decoder.initial_guess.parameters(),
            max_norm=float(args.grad_clip),
        )
        optimizer.step()
        scheduler.step(step)
        window.update(metrics)

        if step % int(args.pretrain_log_interval) == 0 or step == args.initial_guess_steps:
            values = window.values(float(model.density_scale.detach().cpu()))
            elapsed = max(time.time() - window_start, 1.0e-8)
            log_message(
                log_path,
                f"InitialGuess {step:05d}/{args.initial_guess_steps:05d} | "
                f"train_mae={values['mae']:.8f} train_nmae={values['nmae']:.8f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"points_per_s={values['num_points'] / elapsed:.1f} | {elapsed:.1f}s",
            )
            window.reset()
            window_start = time.time()

    val_metrics = evaluate(model, val_loader, device, initial_guess_only=True)
    log_message(
        log_path,
        "InitialGuess complete | "
        f"val_mae={val_metrics['mae']:.8f} val_nmae={val_metrics['nmae']:.8f}",
    )
    atomic_torch_save(
        {
            "initial_guess": model.decoder.initial_guess.state_dict(),
            "steps": int(args.initial_guess_steps),
            "val_metrics": val_metrics,
        },
        save_dir / "initial_guess.pt",
    )

    model.decoder.gto.use_radial_correction = radial_correction_enabled
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Aiden on packed CHGCAR charge-density data"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--elem-csv", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--save-dir", default="./runs/aiden-l3")
    parser.add_argument("--max-files", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--num-queries-train", type=int, default=5000)
    parser.add_argument("--num-queries-val", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--val-batch-size", type=int, default=12)
    parser.add_argument("--val-max-structures", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--in-memory-cache-size", type=int, default=0)

    parser.add_argument("--initial-guess-steps", type=int, default=1000)
    parser.add_argument("--initial-guess-probes", type=int, default=5000)
    parser.add_argument("--initial-guess-batch-size", type=int, default=12)
    parser.add_argument("--initial-guess-lr", type=float, default=0.1)
    parser.add_argument("--initial-guess-restart-steps", type=int, default=250)
    parser.add_argument("--initial-guess-checkpoint", default=None)
    parser.add_argument("--pretrain-log-interval", type=int, default=100)

    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--encoder-lr", type=float, default=2.0e-4)
    parser.add_argument("--decoder-lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--target-std", type=float, default=0.0)
    parser.add_argument("--density-stats-max-structures", type=int, default=4096)
    parser.add_argument("--scale-factor", type=float, default=0.1)

    parser.add_argument("--num-channels", type=int, default=48)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--mmax", type=int, default=3)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--edge-correlation", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--edge-hidden", type=int, default=48)
    parser.add_argument("--atom-cutoff", type=float, default=4.0)
    parser.add_argument("--max-atom-neighbors", type=int, default=100)
    parser.add_argument("--num-atom-rbf", type=int, default=8)
    parser.add_argument("--radial-hidden", type=int, default=128)
    parser.add_argument("--atom-radial-type", choices=["bessel", "gaussian"], default="bessel")
    parser.add_argument("--encoder-interaction-mode", choices=["stacked", "alternating", "ace_only", "tece_only"], default="alternating")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--disable-gie", action="store_true")
    parser.add_argument("--disable-edge-attention", action="store_true")
    parser.add_argument("--disable-rmsnorm", action="store_true")
    parser.add_argument("--disable-tece", action="store_true")
    parser.add_argument("--disable-ece", action="store_true")
    parser.add_argument("--disable-rra", action="store_true")
    parser.add_argument("--disable-radial-phase", action="store_true")
    parser.add_argument("--disable-temperature", action="store_true")
    parser.add_argument("--so2-weight-type", default="w1")
    parser.add_argument("--wigner-type", choices=["recursive", "direct"], default="recursive")

    parser.add_argument("--num-orbitals", type=int, default=8)
    parser.add_argument("--num-gto-radial", type=int, default=8)
    parser.add_argument("--initial-guess-rank", type=int, default=1)
    parser.add_argument("--orbital-cutoff", type=float, default=3.0)
    parser.add_argument("--gto-min-exponent", type=float, default=0.15)
    parser.add_argument("--gto-max-exponent", type=float, default=256.0)
    parser.add_argument("--max-atomic-number", type=int, default=118)
    parser.add_argument("--disable-radial-correction", action="store_true")
    parser.add_argument("--disable-smooth-abs", action="store_true")
    parser.add_argument("--abs-scale", type=float, default=1.0e3)
    parser.add_argument("--gto-edge-chunk-size", type=int, default=32768)
    parser.add_argument(
        "--disable-gto-checkpointing",
        action="store_true",
        help="Retain GTO edge intermediates instead of recomputing chunks in backward",
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-mode", choices=["full", "model_only"], default="full")
    parser.add_argument("--model-name", default="Aiden")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "num_queries_train",
        "num_queries_val",
        "batch_size",
        "val_batch_size",
        "max_steps",
        "log_interval",
        "val_interval",
        "checkpoint_interval",
        "num_orbitals",
        "num_gto_radial",
    )
    for name in positive_integer_fields:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.initial_guess_steps < 0:
        raise ValueError("--initial-guess-steps must be non-negative")
    if args.num_workers < 0 or args.val_num_workers < 0:
        raise ValueError("DataLoader worker counts must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.encoder_lr <= 0.0 or args.decoder_lr <= 0.0:
        raise ValueError("encoder and decoder learning rates must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must lie in (0, 1)")
    if not 0 <= args.lmax <= 4:
        raise ValueError("the Cartesian implementation supports 0 <= lmax <= 4")
    if not 0 <= args.mmax <= args.lmax:
        raise ValueError("--mmax must satisfy 0 <= mmax <= lmax")


def model_arguments(
    args: argparse.Namespace,
    atom_feature_dim: int,
    target_std: float,
) -> Dict[str, object]:
    return {
        "atom_feat_dim": atom_feature_dim,
        "num_channels": args.num_channels,
        "num_encoder_layers": args.num_encoder_layers,
        "lmax": args.lmax,
        "correlation": args.correlation,
        "atom_cutoff": args.atom_cutoff,
        "max_atom_neighbors": args.max_atom_neighbors,
        "num_atom_rbf": args.num_atom_rbf,
        "radial_hidden": args.radial_hidden,
        "atom_radial_type": args.atom_radial_type,
        "dropout": args.dropout,
        "use_gie": not args.disable_gie,
        "use_attention": not args.disable_edge_attention,
        "use_rmsnorm": not args.disable_rmsnorm,
        "use_tece": not args.disable_tece,
        "mmax": args.mmax,
        "edge_correlation": args.edge_correlation,
        "num_heads": args.num_heads,
        "edge_hidden": args.edge_hidden,
        "use_ece": not args.disable_ece,
        "use_rra": not args.disable_rra,
        "use_radial_phase": not args.disable_radial_phase,
        "use_temperature": not args.disable_temperature,
        "so2_weight_type": args.so2_weight_type,
        "wigner_type": args.wigner_type,
        "encoder_interaction_mode": args.encoder_interaction_mode,
        "num_orbitals": args.num_orbitals,
        "num_gto_radial": args.num_gto_radial,
        "initial_guess_rank": args.initial_guess_rank,
        "orbital_cutoff": args.orbital_cutoff,
        "gto_min_exponent": args.gto_min_exponent,
        "gto_max_exponent": args.gto_max_exponent,
        "max_atomic_number": args.max_atomic_number,
        "use_radial_correction": not args.disable_radial_correction,
        "use_abs": not args.disable_smooth_abs,
        "abs_scale": args.abs_scale,
        "gto_edge_chunk_size": args.gto_edge_chunk_size,
        "checkpoint_gto_chunks": not args.disable_gto_checkpointing,
        "scale_factor": args.scale_factor,
        "target_std": target_std,
        "model_name": args.model_name,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None and any(
        (save_dir / name).exists()
        for name in ("last.pt", "best_mae.pt", "best_nmae.pt")
    ):
        raise FileExistsError(
            f"{save_dir} already contains a main-training checkpoint. "
            "Aiden never resumes automatically; choose a new --save-dir, remove "
            "the old run explicitly, or pass --resume."
        )
    log_path = save_dir / "train.log"
    if args.resume is None:
        log_path.write_text("", encoding="utf-8")

    split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)
    common_dataset_args = {
        "data_root": args.data_root,
        "elem_csv": args.elem_csv,
        "split_ratio": split_ratio,
        "seed": args.seed,
        "cache_dir": args.cache_dir,
        "max_files": args.max_files,
        "atom_cutoff": args.atom_cutoff,
        "max_atom_neighbors": args.max_atom_neighbors,
        "query_cutoff": args.orbital_cutoff,
        "max_probe_neighbors": 0,
        "in_memory_cache_size": args.in_memory_cache_size,
        "use_sample_probs": False,
    }
    train_set = CHGCARRhoDataset(
        split="train",
        num_queries=args.num_queries_train,
        fixed_queries=False,
        **common_dataset_args,
    )
    val_set = CHGCARRhoDataset(
        split="val",
        num_queries=args.num_queries_val,
        fixed_queries=True,
        **common_dataset_args,
    )
    val_subset = evenly_spaced_subset(val_set, args.val_max_structures)

    if args.target_std > 0.0:
        target_std = float(args.target_std)
        target_mean = math.nan
        stats_structures = 0
        stats_points = 0
    else:
        target_mean, target_std, stats_structures, stats_points = estimate_target_std(
            train_set,
            args.density_stats_max_structures,
        )

    model_args = model_arguments(args, train_set.feature_dim, target_std)
    model = Aiden(**model_args)
    device = torch.device(args.device)
    model.to(device)

    parameter_count_encoder = sum(p.numel() for p in model.encoder.parameters())
    parameter_count_decoder = sum(p.numel() for p in model.decoder.parameters())
    log_message(
        log_path,
        f"model={args.model_name} geometry=Cartesian-ACE+TECE "
        f"density=low-rank-GTO lmax={args.lmax} mmax={args.mmax} "
        f"channels={args.num_channels} encoder_layers={args.num_encoder_layers} "
        f"correlation={args.correlation} edge_correlation={args.edge_correlation} "
        f"orbitals={args.num_orbitals} gto_radial={args.num_gto_radial} "
        f"orbital_cutoff={args.orbital_cutoff:.3f} "
        f"radial_correction={not args.disable_radial_correction} "
        f"smooth_abs={not args.disable_smooth_abs} "
        f"params_encoder={parameter_count_encoder} "
        f"params_decoder={parameter_count_decoder} "
        f"params_total={parameter_count_encoder + parameter_count_decoder}",
    )
    log_message(
        log_path,
        f"data train={len(train_set)} val={len(val_set)} "
        f"val_fixed_subset={len(val_subset)} backend={train_set.storage_backend} "
        f"graph_backend={graph_backend_name()} train_probes={args.num_queries_train} "
        f"val_probes={args.num_queries_val} batch={args.batch_size} "
        f"train_workers={args.num_workers} val_workers={args.val_num_workers} "
        f"prefetch_factor={args.prefetch_factor}",
    )
    log_message(log_path, f"dataloader_memory {shared_memory_status()}")
    log_message(
        log_path,
        f"target_stats mean={target_mean} std={target_std:.10g} "
        f"structures={stats_structures} points={stats_points} "
        f"density_scale={float(model.density_scale):.10g}",
    )

    first_available_probes = int(train_set.load_density_values(0).size)
    if first_available_probes < args.initial_guess_probes:
        log_message(
            log_path,
            "[data-limit] Initial-guess pretraining requested "
            f"{args.initial_guess_probes} probes, but this cache exposes only "
            f"{first_available_probes} for the first structure. All available "
            "labels will be used; rebuilding from full grids is required for an "
            "exact 48,000-probe pretraining stage.",
        )
    if first_available_probes <= args.num_queries_train:
        log_message(
            log_path,
            "[data-limit] The packed training cache contains no additional grid "
            "points to resample for its first structure; repeated epochs reuse "
            "that stored point set.",
        )

    val_loader = make_loader(
        val_subset,
        args.val_batch_size,
        False,
        args,
        num_workers=args.val_num_workers,
    )

    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu")
        if resume_payload.get("training_objective") != TRAINING_OBJECTIVE:
            raise RuntimeError(
                "checkpoint objective/architecture is not Aiden MAE v1"
            )
        model.load_state_dict(resume_payload["model"], strict=True)
        log_message(log_path, f"Loaded explicit checkpoint: {args.resume}")
    elif args.initial_guess_checkpoint is not None:
        initial_payload = torch.load(args.initial_guess_checkpoint, map_location="cpu")
        state = initial_payload.get("initial_guess", initial_payload)
        model.decoder.initial_guess.load_state_dict(state, strict=True)
        log_message(
            log_path,
            f"Loaded explicit initial-guess checkpoint: {args.initial_guess_checkpoint}",
        )
    else:
        pretrain_set = CHGCARRhoDataset(
            split="train",
            num_queries=args.initial_guess_probes,
            fixed_queries=False,
            **common_dataset_args,
        )
        pretrain_loader = make_loader(
            pretrain_set,
            args.initial_guess_batch_size,
            True,
            args,
        )
        pretrain_initial_guess(
            model,
            pretrain_loader,
            val_loader,
            device,
            args,
            save_dir,
            log_path,
        )

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.encoder.parameters(),
                "lr": float(args.encoder_lr),
                "name": "encoder",
            },
            {
                "params": model.decoder.parameters(),
                "lr": float(args.decoder_lr),
                "name": "decoder",
            },
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.max_steps),
        eta_min=0.0,
    )
    ema = ModelEMA(model, args.ema_decay)
    global_step = 0
    epoch = 0
    best_val_mae = math.inf
    best_val_nmae = math.inf

    if resume_payload is not None:
        if args.resume_mode == "full":
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            ema.load_state_dict(resume_payload["ema"])
            global_step = int(resume_payload["global_step"])
            epoch = int(resume_payload.get("epoch", 0))
            best_val_mae = float(resume_payload.get("best_val_mae", math.inf))
            best_val_nmae = float(resume_payload.get("best_val_nmae", math.inf))
            log_message(
                log_path,
                f"Full resume at step={global_step} epoch={epoch} "
                f"lr={optimizer_lr_string(optimizer)}",
            )
        else:
            ema = ModelEMA(model, args.ema_decay)
            log_message(log_path, "Model-only resume; optimizer and step reset")

    config_payload = {
        "training_objective": TRAINING_OBJECTIVE,
        "args": vars(args),
        "model_args": model_args,
        "target_std": target_std,
    }
    (save_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if global_step == 0:
        ema.apply(model)
        try:
            initial_val = evaluate(model, val_loader, device)
        finally:
            ema.restore(model)
        log_message(
            log_path,
            f"Step 0000000 | val_mae={initial_val['mae']:.8f} "
            f"val_nmae={initial_val['nmae']:.8f} | "
            "pretrained_initial_guess_plus_untrained_environment_model",
        )

    log_message(
        log_path,
        f"Start main training | objective=MAE "
        f"fixed_loss_scale={float(model.density_scale):.8g} optimizer=Adam "
        f"encoder_lr={args.encoder_lr:.3e} decoder_lr={args.decoder_lr:.3e} "
        f"scheduler=CosineAnnealingLR T_max={args.max_steps} "
        f"grad_clip={args.grad_clip} ema={args.ema_decay} "
        f"max_steps={args.max_steps} val_interval={args.val_interval}",
    )

    training_batches = ResilientTrainingBatches(
        train_set,
        args.batch_size,
        args,
        log_path,
        initial_epoch=epoch,
    )
    window = MetricAccumulator()
    window_start = time.time()
    model.train()
    while global_step < int(args.max_steps):
        epoch, cpu_batch = training_batches.next(global_step + 1)
        batch = move_batch_to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, metrics = mae_objective(outputs, batch, model.density_scale)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite MAE at step {global_step + 1}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(args.grad_clip),
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at step {global_step + 1}"
            )
        optimizer.step()
        scheduler.step()
        ema.update(model)
        global_step += 1
        window.update(metrics)

        should_log = global_step % int(args.log_interval) == 0
        should_validate = (
            global_step % int(args.val_interval) == 0
            or global_step == int(args.max_steps)
        )
        should_checkpoint = global_step % int(args.checkpoint_interval) == 0

        if should_log:
            train_values = window.values(float(model.density_scale.detach().cpu()))
            elapsed = max(time.time() - window_start, 1.0e-8)
            log_message(
                log_path,
                f"Step {global_step:07d} | epoch={epoch:04d} | "
                f"train_mae={train_values['mae']:.8f} "
                f"train_nmae={train_values['nmae']:.8f} | "
                f"lr={optimizer_lr_string(optimizer)} | "
                f"grad_norm={float(grad_norm.detach().cpu()):.4e} | "
                f"points_per_s={train_values['num_points'] / elapsed:.1f} "
                f"structures_per_s={train_values['num_structures'] / elapsed:.2f} | "
                f"{elapsed:.1f}s",
            )
            if not should_validate:
                window.reset()
                window_start = time.time()

        if should_validate:
            train_values = window.values(float(model.density_scale.detach().cpu()))
            ema.apply(model)
            try:
                val_values = evaluate(model, val_loader, device)
            finally:
                ema.restore(model)
            log_message(
                log_path,
                f"Step {global_step:07d} | "
                f"train_mae={train_values['mae']:.8f} "
                f"train_nmae={train_values['nmae']:.8f} | "
                f"val_mae={val_values['mae']:.8f} "
                f"val_nmae={val_values['nmae']:.8f} | "
                f"lr={optimizer_lr_string(optimizer)}",
            )

            improved_mae = val_values["mae"] < best_val_mae
            improved_nmae = val_values["nmae"] < best_val_nmae
            best_val_mae = min(best_val_mae, val_values["mae"])
            best_val_nmae = min(best_val_nmae, val_values["nmae"])
            save_checkpoint(
                save_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                ema,
                args,
                model_args,
                global_step,
                epoch,
                best_val_mae,
                best_val_nmae,
            )
            if improved_mae:
                save_checkpoint(
                    save_dir / "best_mae.pt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    args,
                    model_args,
                    global_step,
                    epoch,
                    best_val_mae,
                    best_val_nmae,
                )
                log_message(log_path, f"Saved new best-MAE checkpoint at step {global_step}")
            if improved_nmae:
                save_checkpoint(
                    save_dir / "best_nmae.pt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    args,
                    model_args,
                    global_step,
                    epoch,
                    best_val_mae,
                    best_val_nmae,
                )
                log_message(log_path, f"Saved new best-NMAE checkpoint at step {global_step}")
            window.reset()
            window_start = time.time()
            model.train()
        elif should_checkpoint:
            save_checkpoint(
                save_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                ema,
                args,
                model_args,
                global_step,
                epoch,
                best_val_mae,
                best_val_nmae,
            )

    log_message(
        log_path,
        f"Training complete at step={global_step} | "
        f"best_val_mae={best_val_mae:.8f} best_val_nmae={best_val_nmae:.8f}",
    )
    training_batches.close()


if __name__ == "__main__":
    main()
