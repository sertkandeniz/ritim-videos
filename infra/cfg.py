"""Tiny config reader for bootstrap.sh: prints install specs from config/pipeline.yaml.

  cfg.py pip <env>   -> space-separated pip requirement specs
  cfg.py git <env>   -> "<url> <commit>"
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"


def main(argv: list[str]) -> int:
    kind, env = argv[1], argv[2]
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["models"]["envs"][env]
    if kind == "pip":
        print(" ".join(spec.get("pip", [])))
    elif kind == "git":
        print(spec["git"], spec["commit"])
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
