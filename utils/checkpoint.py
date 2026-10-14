"""Strict loading of the compact public inference checkpoint."""

from collections.abc import Mapping

import torch

from ldit.model import create_student


MODEL_CONFIG = {
    "architecture": "LightningDiT-XL/1",
    "dmf_depth": 10,
    "fixed_guidance": 2.0,
    "latent_size": 16,
    "latent_channels": 32,
    "num_classes": 1000,
}


def public_metadata(metadata):
    """Keep documented inference provenance, excluding arbitrary local metadata."""
    integer = lambda value: type(value) is int and value >= 0
    positive_integer = lambda value: type(value) is int and value > 0

    def constant(expected):
        return lambda value: type(value) is type(expected) and value == expected

    def sequence(expected):
        return lambda value: (type(value) in (list, tuple) and len(value) == len(expected)
                              and all(type(item) is float and item == target
                                      for item, target in zip(value, expected)))

    def project(source, fields):
        if not isinstance(source, Mapping):
            raise ValueError("Public checkpoint metadata must be a mapping")
        selected = {}
        for key, value in source.items():
            if key not in fields:
                continue
            if not fields[key](value):
                raise ValueError(f"Invalid public checkpoint metadata field: {key}")
            selected[key] = value
        return selected

    ema_fields = {
        "format_version": constant(1), "implementation": constant("ema_pytorch.PostHocEMA"),
        "sigma_rels": sequence((0.05, 0.2)),
        "gammas": sequence((16.972198602303447, 1.8262137798958404)),
        "reporting_index": constant(1), "update_every": constant(1),
        "start_step": integer, "num_updates": integer, "training_step": integer,
    }
    protocol_fields = {
        "backend": constant("facm_torch_fidelity_v1"), "weights": constant("posthoc"),
        "sigma_rel": constant(0.2), "seed": integer, "samples": positive_integer,
        "guidance": constant(2.0), "bf16": constant(True),
    }
    result = project(metadata, {
        "step": positive_integer, "weights": constant("posthoc"),
        "sigma_rel": constant(0.2), "reporting_index": constant(1),
        "source_sha256": lambda value: (type(value) is str and len(value) == 64
                                         and all(char in "0123456789abcdef" for char in value)),
        "ema": lambda value: isinstance(value, Mapping),
        "evaluation_ema": lambda value: isinstance(value, Mapping),
        "evaluation_protocol": lambda value: isinstance(value, Mapping),
    })
    for key in ("ema", "evaluation_ema"):
        if key in result:
            result[key] = project(result[key], ema_fields)
    if "evaluation_protocol" in result:
        result["evaluation_protocol"] = project(result["evaluation_protocol"], protocol_fields)
    return result


def validate_state_dict(model, state):
    """Reject incompatible keys, shapes, or precision before loading tensors."""
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("model must be a string-to-tensor state dictionary")
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    invalid = [
        key for key in expected.keys() & state.keys()
        if state[key].shape != expected[key].shape
        or state[key].dtype != expected[key].dtype
        or state[key].layout != torch.strided
    ]
    if missing or unexpected or invalid:
        raise ValueError(
            "Incompatible model state: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape/dtype/layout mismatch={invalid[:8]}"
        )


def load_student(path, device="cpu"):
    """Return ``(student, metadata)`` from an exported checkpoint.

    Memory mapping and meta-device construction avoid allocating a second XL
    model on CPU. All checkpoint parameters, including the constant guidance
    conditioning modules, load without casting or transformation.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("Expected a public checkpoint with format_version=1")
    if payload.get("model_config") != MODEL_CONFIG:
        raise ValueError(f"Checkpoint model_config must equal {MODEL_CONFIG}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping")
    if (
        type(metadata.get("step")) is not int
        or metadata["step"] < 1
        or metadata.get("weights") != "posthoc"
        or metadata.get("sigma_rel") != 0.2
        or metadata.get("reporting_index") != 1
    ):
        raise ValueError("Expected positive step and PostHocEMA index 1 (sigma_rel=0.2)")
    with torch.device("meta"):
        model = create_student()
    state = payload.get("model")
    validate_state_dict(model, state)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device).eval(), dict(metadata)
