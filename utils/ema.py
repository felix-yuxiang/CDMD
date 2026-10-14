"""The original training EMA and FACM reporting PostHocEMA (sigma_rel=0.2)."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch


class FACMEvaluationEMA:
    """Track FACM's two PostHocEMA averages and report sigma_rel=0.2 directly.

    Construct this after restoring the raw student, then load its full saved
    history when resuming. The reporting weights use sigma_rel=0.2.
    """

    filename = "evaluation_ema.pt"
    sigma_rels = (0.05, 0.2)
    reporting_index = 1

    def __init__(self, student, *, start_step, checkpoint_folder, device="cpu"):
        try:
            from ema_pytorch import PostHocEMA
        except ImportError as error:
            raise ImportError(
                "Reporting EMA requires ema-pytorch; install requirements-train.txt"
            ) from error

        self._require_nonnegative_int(start_step, "start_step")
        self.start_step = start_step
        self.storage_device = torch.device(device)
        self._ema = PostHocEMA(
            student,
            ema_model=lambda: self._copy_student(student),
            sigma_rels=self.sigma_rels,
            update_every=1,
            checkpoint_every_num_steps="manual",
            checkpoint_folder=checkpoint_folder,
            allow_different_devices=True,
        ).to(self.storage_device)

    def _copy_student(self, student):
        model = deepcopy(student)
        # Accelerate's forward closure survives deepcopy and still calls the
        # online student. The copied bound method uses the copied EMA weights.
        # Evaluation enters autocast explicitly, so it needs no copied wrapper.
        if hasattr(model, "_original_forward"):
            model.forward = model._original_forward
            del model._original_forward
        return model.to(self.storage_device).eval()

    @staticmethod
    def _require_nonnegative_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"FACM evaluation EMA {name} must be a nonnegative integer")

    @property
    def model(self):
        return self._ema.ema_models[self.reporting_index].ema_model

    @property
    def num_updates(self):
        return int(self._ema.step.item())

    def metadata(self):
        return {
            "format_version": 1,
            "implementation": "ema_pytorch.PostHocEMA",
            "sigma_rels": self.sigma_rels,
            "gammas": tuple(self._ema.gammas),
            "reporting_index": self.reporting_index,
            "update_every": 1,
            "start_step": self.start_step,
            "num_updates": self.num_updates,
            "training_step": self.start_step + self.num_updates,
        }

    def _validate_clock(self, training_step):
        self._require_nonnegative_int(training_step, "training_step")
        expected_updates = training_step - self.start_step
        steps = [int(ema.step.item()) for ema in self._ema.ema_models]
        if expected_updates < 0 or steps != [expected_updates] * len(self.sigma_rels):
            raise RuntimeError(
                f"FACM evaluation EMA clock mismatch at training step {training_step}: "
                f"start_step={self.start_step}, EMA steps={steps}"
            )
        if any(bool(ema.initted.item()) != (expected_updates > 0)
               for ema in self._ema.ema_models):
            raise RuntimeError("FACM evaluation EMA initialization state disagrees with its clock")

    @torch.no_grad()
    def update(self, *, training_step):
        """Call once immediately after each successful student optimizer update."""
        self._require_nonnegative_int(training_step, "training_step")
        self._validate_clock(training_step - 1)
        self._ema.update()
        self._validate_clock(training_step)

    @contextmanager
    def on_device(self, device):
        """Temporarily move only the reporting average to the evaluation device."""
        try:
            yield self.model.to(device).eval()
        finally:
            self.model.to(self.storage_device)

    def save(self, checkpoint_dir, *, training_step):
        self._validate_clock(training_step)
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / self.filename
        temporary_path = path.with_suffix(".tmp")
        torch.save({**self.metadata(), "ema_state": self._ema.state_dict()}, temporary_path)
        temporary_path.replace(path)

    def load(self, checkpoint_dir, *, training_step):
        """Restore the complete reporting history required by native resume."""
        path = Path(checkpoint_dir) / self.filename
        payload = torch.load(path, map_location=self.storage_device, weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError("FACM evaluation EMA checkpoint must be a mapping")
        for name in ("format_version", "implementation", "sigma_rels", "gammas",
                     "reporting_index", "update_every"):
            expected = self.metadata()[name]
            if payload.get(name) != expected:
                raise RuntimeError(
                    f"FACM evaluation EMA checkpoint {name} mismatch: "
                    f"expected {expected!r}, got {payload.get(name)!r}"
                )
        for name in ("start_step", "num_updates", "training_step"):
            self._require_nonnegative_int(payload.get(name), name)
        if (payload["training_step"] != training_step
                or payload["start_step"] + payload["num_updates"] != training_step):
            raise RuntimeError(
                f"FACM evaluation EMA checkpoint clock does not match training step {training_step}"
            )
        self._ema.load_state_dict(payload["ema_state"], strict=True)
        self.start_step = payload["start_step"]
        self._validate_clock(training_step)
        return True


class TrainingEMA(torch.nn.Module):
    """Original fixed-decay average; independent of the reporting average."""

    def __init__(self, student, decay=0.999, device=None):
        super().__init__()
        self.decay = decay
        self.ema_model = deepcopy(student).to(device).requires_grad_(False)

    @torch.no_grad()
    def update(self, student):
        for target, source in zip(self.ema_model.parameters(), student.parameters()):
            target.detach().mul_(self.decay).add_(source, alpha=1 - self.decay)
        for (name, target), (_, source) in zip(
            self.ema_model.named_buffers(), student.named_buffers()
        ):
            if "num_batches_tracked" in name:
                target.copy_(source)
            else:
                target.detach().mul_(self.decay).add_(source, alpha=1 - self.decay)
