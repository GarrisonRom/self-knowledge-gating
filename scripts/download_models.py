"""Download the fixed retrieval models from ModelScope.

This script only prepares local weights. Retrieval and evaluation must be run
with the resulting paths and offline mode enabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_IDS = (
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/modelscope"),
        help="ModelScope cache directory (ignored by Git).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional ModelScope revision or commit to pin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "ModelScope is not installed. Run: pip install -e \".[retrieval]\""
        ) from exc

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for model_id in MODEL_IDS:
        kwargs = {"model_id": model_id, "cache_dir": str(args.cache_dir)}
        if args.revision:
            kwargs["revision"] = args.revision
        path = snapshot_download(**kwargs)
        print(f"{model_id}\t{path}")


if __name__ == "__main__":
    main()
