"""Download the pinned teacher, VA-VAE, latent statistics, and FID reference."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--training", action="store_true", help="Also download the pretrained teacher")
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without network access")
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "configs/assets.json"
    manifest = json.loads(manifest_path.read_text())
    for name, info in manifest["files"].items():
        if info["training_only"] and not args.training:
            continue
        path = args.cache_dir / name
        if not path.is_file():
            if args.verify_only:
                raise FileNotFoundError(path)
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(
                repo_id=manifest["repository"], revision=manifest["revision"],
                filename=name, local_dir=args.cache_dir,
            ))
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {path}; remove this file and download it again")
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    main()
