"""Import the original schema-2 training state without resetting its averages or Adam moments."""

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from safetensors.torch import load_file


ASSET_FILES = json.loads((Path(__file__).resolve().parents[1] / "configs/assets.json").read_text())["files"]
REQUIRED_FILES = ("trainer_state.json", "model.safetensors", "optimizer.bin", "scheduler.bin",
                  "multi_ema.pt", "evaluation_ema.pt", "random_states_0.pkl")
OBJECTIVE = {
    "checkpoint_schema": 2, "optimizer": "adamw", "ldit_model": "LightningDiT-XL/1",
    "dmf_depth": 10, "dmd_proxy_model": False, "evaluator_registered": False,
    "student_guidance_mode": "imf", "mf_student_cfg_target": False,
    "imf_cfg_beta": 2.0, "imf_cfg_s_max": 3.0,
    "teacher_cfg_scale": 1.75, "teacher_cfg_min": 0.125, "teacher_cfg_max": 1.0,
    "cfg_scale": 2.0, "t_r_ratio": 1.0, "t_logit_mean": -0.4, "t_logit_std": 1.0,
    "r_logit_mean": 1.2, "r_logit_std": 1.0, "norm_eps": 0.001, "norm_p": 1.0,
    "facm_anchor_curriculum": True, "facm_curriculum_start": 5000,
    "facm_curriculum_ramp": 5000, "xt_mixture_new_noise_prob": 0.5,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_assets(config, protocol):
    """Require the pinned teacher/decoder/statistics; accept the original FID container."""
    for key, filename in (("teacher_checkpoint", "800ep-stg1.pt"),
                          ("vae_checkpoint", "vavae-imagenet256-f16d32-dinov2.pt"),
                          ("latent_stats_path", "latents_stats.pt")):
        path, expected = Path(config[key]).expanduser(), ASSET_FILES[filename]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"Legacy import requires the pinned {filename}: {path}")
    reference = Path(config["ref_stats_path"]).expanduser()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    from utils.evaluation import reference_statistics

    current_mean, current_covariance = reference_statistics(reference)
    expected = ASSET_FILES["fid-50k-256.npz"]
    if reference.stat().st_size == expected["bytes"] and _sha256(reference) == expected["sha256"]:
        return
    original = Path(protocol["ref_stats_path"]).expanduser()
    if not original.is_file():
        raise ValueError("Cannot verify relocated FID statistics; use the pinned fid-50k-256.npz")
    original_mean, original_covariance = reference_statistics(original)
    if not (np.array_equal(current_mean, original_mean) and np.array_equal(current_covariance, original_covariance)):
        raise ValueError("FID statistics differ from the archived evaluation reference")


def metadata(path, config):
    """Validate an original full checkpoint and the continuation's fixed recipe.

    Asset checks hash the published teacher, VA-VAE and normalization statistics.
    Call this preflight once before a distributed launch when possible.
    """
    path = Path(path).expanduser().resolve()
    for filename in REQUIRED_FILES:
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Incomplete legacy checkpoint: {path / filename}")
    if (path / "model_1.safetensors").exists() and load_file(str(path / "model_1.safetensors")):
        raise ValueError("Unexpected additional model state in the legacy checkpoint")
    saved = json.loads((path / "trainer_state.json").read_text())
    for key, expected in OBJECTIVE.items():
        if saved.get(key) != expected:
            raise ValueError(f"Legacy checkpoint {key}: expected {expected!r}, got {saved.get(key)!r}")
    step = saved.get("step")
    if type(step) is not int or step <= 0:
        raise ValueError("Legacy checkpoint step must be a positive integer")
    if (type(saved.get("best_fid_step")) is not int or not 0 < saved["best_fid_step"] <= step
            or not isinstance(saved.get("best_fid"), (int, float))
            or not math.isfinite(saved["best_fid"]) or saved["best_fid"] <= 0):
        raise ValueError("Legacy best-FID record is invalid")
    for key, expected in {"seed": 42, "attn_func": "fa2", "bf16": True,
                          "learning_rate": 1e-4, "global_batch_size": 1024,
                          "eval_every": 500, "eval_samples": 50000,
                          "eval_batch_size": 32, "decode_batch_size": 8}.items():
        if config.get(key) != expected:
            raise ValueError(f"Legacy continuation config differs for {key}: expected {expected!r}")
    if saved.get("attn_func") != "fa2":
        raise ValueError("This legacy continuation requires the original FA2 checkpoint")
    if config["train_steps"] <= step:
        raise ValueError("train_steps must exceed the legacy checkpoint step")
    protocol = saved.get("evaluation_protocol", {})
    for key, expected in {"backend": "facm_torch_fidelity_v1", "weights": "posthoc",
                          "sigma_rel": 0.2, "seed": 0, "samples": 50000,
                          "guidance": 2.0, "bf16": True}.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Legacy evaluation protocol differs for {key}")
    ema = saved.get("evaluation_ema", {})
    for key, expected in {"format_version": 1, "implementation": "ema_pytorch.PostHocEMA",
                          "sigma_rels": [0.05, 0.2], "reporting_index": 1,
                          "gammas": [16.972198602303447, 1.8262137798958404],
                          "update_every": 1, "start_step": 0,
                          "num_updates": step, "training_step": step}.items():
        if ema.get(key) != expected:
            raise ValueError(f"Legacy reporting EMA metadata differs for {key}")
    _validate_assets(config, protocol)
    return saved


def _validate_model_state(model, state, label):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError(f"{label} tensor names do not match the public student")
    for name, value in state.items():
        if not torch.is_tensor(value) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"{label} tensor shape/dtype mismatch: {name}")


def _remap_optimizer(raw_student, optimizer, archived, ordered_model_state, step):
    """The archived optimizer excluded frozen position embeddings; public Adam includes them."""
    if len(archived.get("param_groups", [])) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError("Expected one AdamW parameter group")
    group = archived["param_groups"][0]
    for key, expected in {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
                          "weight_decay": 0.0, "amsgrad": False, "maximize": False,
                          "capturable": False, "differentiable": False,
                          "initial_lr": 1e-4}.items():
        if group.get(key) != expected:
            raise ValueError(f"Legacy optimizer differs for {key}")
    named = dict(raw_student.named_parameters())
    trainable = {name for name, parameter in named.items() if parameter.requires_grad}
    # multi_ema retains the original model registration order, unlike the
    # alphabetically ordered safetensors keys. Match optimizer slots by name.
    saved_names = [name for name in ordered_model_state if name in trainable]
    saved_ids = group["params"]
    current_names = [name for name, parameter in named.items() if parameter.requires_grad]
    if (saved_names != current_names or len(saved_ids) != len(saved_names)
            or len(set(saved_ids)) != len(saved_ids)):
        raise ValueError("Legacy optimizer parameter count/order is incompatible")
    if set(archived.get("state", {})) != set(saved_ids):
        raise ValueError("Legacy optimizer is missing or has unexpected moment slots")
    names_by_identity = {id(parameter): name for name, parameter in named.items()}
    current_ids = optimizer.state_dict()["param_groups"][0]["params"]
    current_parameters = optimizer.param_groups[0]["params"]
    try:
        target_ids = {names_by_identity[id(parameter)]: index
                      for parameter, index in zip(current_parameters, current_ids)}
    except KeyError as error:
        raise ValueError("Public optimizer contains parameters outside the student") from error
    if not trainable <= target_ids.keys() or len(target_ids) != len(current_parameters):
        raise ValueError("Public optimizer must contain every trainable student parameter once")
    remapped = {}
    for index, name in zip(saved_ids, saved_names):
        slot = archived["state"][index]
        if set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Legacy Adam moment fields mismatch: {name}")
        clock = slot["step"]
        if not torch.is_tensor(clock) or clock.numel() != 1 or clock.item() != step:
            raise ValueError(f"Legacy optimizer step mismatch: {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = slot[key]
            if not torch.is_tensor(value) or value.shape != named[name].shape or value.dtype != named[name].dtype:
                raise ValueError(f"Legacy optimizer moment shape/dtype mismatch: {name}/{key}")
        remapped[target_ids[name]] = slot
    return {"state": remapped, "param_groups": [{**group, "params": current_ids}]}


def load_legacy_training(path, raw_student, optimizer, scheduler, training_ema, reporting_ema, metadata):
    """Call after accelerator.prepare; retain existing parameter objects and teacher initialization."""
    path = Path(path).expanduser().resolve()
    inner_optimizer = getattr(optimizer, "optimizer", optimizer)
    inner_scheduler = getattr(scheduler, "scheduler", scheduler)
    raw_state = load_file(str(path / "model.safetensors"), device="cpu")
    _validate_model_state(raw_student, raw_state, "Raw student")
    averages = torch.load(path / "multi_ema.pt", map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(averages, dict) or set(averages) != {"0.999"} or training_ema.decay != 0.999:
        raise ValueError("Expected only the archived 0.999 training EMA")
    training_state = averages["0.999"]
    _validate_model_state(training_ema.ema_model, training_state, "Training EMA")
    archived_optimizer = torch.load(path / "optimizer.bin", map_location="cpu", weights_only=True, mmap=True)
    optimizer_state = _remap_optimizer(raw_student, inner_optimizer, archived_optimizer, training_state, metadata["step"])
    scheduler_state = torch.load(path / "scheduler.bin", map_location="cpu", weights_only=True)
    if scheduler_state.get("base_lrs") != [1e-4] or scheduler_state.get("_last_lr") != [1e-4]:
        raise ValueError("Legacy scheduler learning rate differs")
    if (type(scheduler_state.get("last_epoch")) is not int or scheduler_state["last_epoch"] < 0
            or scheduler_state.get("_step_count") != scheduler_state["last_epoch"] + 1
            or scheduler_state.get("lr_lambdas") != [None]):
        raise ValueError("Legacy constant scheduler state is invalid")
    raw_student.load_state_dict(raw_state, strict=True)
    training_ema.ema_model.load_state_dict(training_state, strict=True)
    inner_optimizer.load_state_dict(optimizer_state)
    inner_scheduler.load_state_dict(scheduler_state)
    del raw_state, training_state, averages, archived_optimizer, optimizer_state
    reporting_ema.load(path, training_step=metadata["step"])
    if json.loads(json.dumps(reporting_ema.metadata())) != metadata["evaluation_ema"]:
        raise ValueError("Restored reporting EMA differs from trainer_state.json")
    return {"source_checkpoint": str(path), "step": metadata["step"],
            "raw_tensors": len(raw_student.state_dict()),
            "optimizer_moment_slots": len(inner_optimizer.state),
            "optimizer_reset": False, "scheduler_last_epoch": inner_scheduler.last_epoch,
            "training_ema_decay": training_ema.decay,
            "reporting_ema": reporting_ema.metadata(),
            "rng_policy": "restore archived rank 0; seed other ranks with seed + rank"}


def restore_legacy_rng(path, rank, seed):
    """Restore the known rank-zero stream; never pretend its other devices are other ranks.

    This trusted local training archive contains NumPy's pickled RNG state.
    Call last, after model/optimizer/EMA/tracker initialization, before training.
    """
    if rank == 0:
        state = torch.load(Path(path) / "random_states_0.pkl", map_location="cpu", weights_only=False)
        required = {"random_state", "numpy_random_seed", "torch_manual_seed", "torch_cuda_manual_seed"}
        if not required <= state.keys():
            raise ValueError("Legacy rank-zero RNG state is incomplete")
        cuda_states = state["torch_cuda_manual_seed"]
        if not isinstance(cuda_states, (tuple, list)) or not cuda_states:
            raise ValueError("Legacy rank-zero CUDA RNG state is missing")
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_seed"])
        torch.set_rng_state(state["torch_manual_seed"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_states[0], device=torch.cuda.current_device())
        return {"rank": rank, "policy": "restored_archived_rank_zero"}
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.random.default_generator.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
    return {"rank": rank, "policy": "seeded_missing_archived_rank", "seed": rank_seed}
