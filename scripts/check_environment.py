"""Report runtime capabilities without enforcing exact package versions."""

from __future__ import annotations

import importlib
import platform
import sys


def _version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"missing ({type(exc).__name__})"
    return str(getattr(module, "__version__", "available"))


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {_version('torch')}")
    print(f"transformers: {_version('transformers')}")
    print(f"modelscope: {_version('modelscope')}")
    print(f"FlagEmbedding: {_version('FlagEmbedding')}")
    try:
        import torch

        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            print(f"cuda_runtime: {torch.version.cuda}")
    except Exception as exc:
        print(f"cuda_check: unavailable ({type(exc).__name__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
