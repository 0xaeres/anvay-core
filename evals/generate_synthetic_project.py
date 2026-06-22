from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

CODEBASE_JSONL = Path(__file__).parent / "synthetic_codebase.jsonl"


def generate_project(dest_path: Path) -> None:
    """Clear dest_path and write files defined in synthetic_codebase.jsonl."""
    log.info("Generating synthetic project under %s", dest_path)
    shutil.rmtree(dest_path, ignore_errors=True)
    dest_path.mkdir(parents=True, exist_ok=True)

    if not CODEBASE_JSONL.exists():
        raise FileNotFoundError(f"Synthetic codebase template not found at {CODEBASE_JSONL}")

    with open(CODEBASE_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            rel_path = item["path"]
            content = item["content"]

            file_path = dest_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            log.debug("Wrote synthetic file: %s", file_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic project files.")
    parser.add_argument("--out", type=Path, default=Path("evals/fixtures/synthetic_project"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_project(args.out)
    log.info("Successfully generated synthetic project at %s", args.out)
