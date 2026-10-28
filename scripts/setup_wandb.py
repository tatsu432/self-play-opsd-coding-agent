"""Run in a terminal on Yale to create a private compute-visible credentials file."""

import argparse
import getpass
import os
import re
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    entity = input("Authorized W&B entity (account or team): ").strip()
    key = getpass.getpass("W&B API key (hidden): ").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", entity) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise SystemExit("Invalid entity/key syntax; nothing written")
    path = args.path.expanduser().absolute()
    if path.exists():
        raise SystemExit("File already exists; inspect/rotate it explicitly")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"WANDB_API_KEY='{key}'\nWANDB_ENTITY='{entity}'\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-clobber publication, private from creation onward.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    print(f"Created private credentials file: {path}")


if __name__ == "__main__":
    main()
