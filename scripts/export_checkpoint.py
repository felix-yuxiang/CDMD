"""Export the selected PostHocEMA model without changing a single tensor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldit.model import create_student
from utils.checkpoint import MODEL_CONFIG, load_student, public_metadata, validate_state_dict


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(ckpt_dir, output_path, *, expected_step=None, expected_source_sha256=None):
    """Extract reporting EMA 1 from an original or public-training checkpoint.

    Optional expected values verify a particular archived release. The output
    includes no optimizer state, other EMA model, or machine-specific paths.
    """
    ckpt_dir, output_path = Path(ckpt_dir), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    source = ckpt_dir / "evaluation_ema.pt"
    trainer = json.loads((ckpt_dir / "trainer_state.json").read_text())
    source_sha256 = sha256_file(source)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError("Source checkpoint SHA256 does not match the expected release")
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    step = trainer.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("trainer_state.json must contain a positive integer step")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"Expected step {expected_step}, found {step}")
    if trainer.get("ldit_model") != "LightningDiT-XL/1" or trainer.get("dmf_depth") != 10:
        raise ValueError("Expected LightningDiT-XL/1 with dmf_depth=10")
    protocol = trainer.get("evaluation_protocol", {})
    if protocol.get("guidance") != 2.0:
        raise ValueError("The public model requires evaluation guidance fixed at 2.0")
    expected_ema = {
        "format_version": 1,
        "implementation": "ema_pytorch.PostHocEMA",
        "reporting_index": 1,
        "update_every": 1,
        "start_step": 0,
        "num_updates": step,
        "training_step": step,
    }
    for key, expected in expected_ema.items():
        if payload.get(key) != expected:
            raise ValueError(f"Invalid EMA metadata: {key} must be {expected!r}")
    if tuple(payload.get("sigma_rels", ())) != (0.05, 0.2):
        raise ValueError("Expected EMA sigma_rels=(0.05, 0.2)")
    gammas = payload.get("gammas", ())
    if len(gammas) != 2 or not all(
        math.isclose(actual, expected, rel_tol=1e-12)
        for actual, expected in zip(gammas, (16.972198602303447, 1.8262137798958404))
    ):
        raise ValueError("EMA gammas do not match sigma_rels=(0.05, 0.2)")
    ema_metadata = public_metadata({
        "ema": {key: value for key, value in payload.items() if key != "ema_state"}
    })["ema"]
    trainer_ema = public_metadata({"ema": trainer.get("evaluation_ema")})["ema"]
    # JSON arrays and saved tuples describe the same EMA metadata.
    if json.loads(json.dumps(ema_metadata)) != json.loads(json.dumps(trainer_ema)):
        raise ValueError("Saved EMA metadata disagrees with trainer_state.json")
    ema_state = payload["ema_state"]
    if int(ema_state["ema_models.1.step"].item()) != step or not bool(
        ema_state["ema_models.1.initted"].item()
    ):
        raise ValueError("Reporting EMA is uninitialized or has an incorrect update count")
    prefix = "ema_models.1.ema_model."
    state = {key[len(prefix):]: value for key, value in ema_state.items() if key.startswith(prefix)}
    with torch.device("meta"):
        model = create_student()
    validate_state_dict(model, state)
    metadata = public_metadata({
        "step": step,
        "weights": "posthoc",
        "sigma_rel": 0.2,
        "reporting_index": 1,
        "ema": ema_metadata,
        "source_sha256": source_sha256,
        "evaluation_protocol": protocol,
    })
    result = {
        "format_version": 1,
        "model": state,
        "model_config": dict(MODEL_CONFIG),
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    try:
        torch.save(result, temporary)
        loaded, loaded_metadata = load_student(temporary)
        for key, value in loaded.state_dict().items():
            if not torch.equal(value, state[key]):
                raise RuntimeError(f"Export changed tensor {key}")
        if loaded_metadata != metadata:
            raise RuntimeError("Export changed checkpoint metadata")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n"
    )
    provenance = {
        "filename": output_path.name,
        "sha256": digest,
        "size_bytes": output_path.stat().st_size,
        "tensor_count": len(state),
        "all_tensors_exactly_equal_to_source": True,
        "model_config": MODEL_CONFIG,
        "metadata": metadata,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    result = export_checkpoint(
        args.checkpoint_dir, args.output,
        expected_step=args.expected_step,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
