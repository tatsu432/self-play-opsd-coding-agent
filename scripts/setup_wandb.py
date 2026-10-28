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
    parser.add_argument(
        "--from-login", action="store_true", help="Reuse the standard wandb login credential"
    )
    parser.add_argument(
        "--entity", help="Authorized entity; defaults to the authenticated account's default"
    )
    args = parser.parse_args()
    if args.from_login:
        import netrc

        import wandb

        credentials = netrc.netrc().authenticators("api.wandb.ai")
        if not credentials:
            raise SystemExit("Run uv run wandb login first")
        key = credentials[2]
        entity = args.entity or wandb.Api().default_entity
        if not entity:
            raise SystemExit("No default entity; specify --entity")
    else:
        entity = (
            args.entity or input("W&B entity (Enter to discover your authorized default): ").strip()
        )
        key = getpass.getpass("W&B API key (hidden): ").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise SystemExit("Invalid key syntax; nothing written")
    if not entity:
        import httpx

        try:
            response = httpx.post(
                "https://api.wandb.ai/graphql",
                auth=("api", key),
                json={"query": "query { viewer { username entity } }"},
                timeout=20,
            )
            response.raise_for_status()
            viewer = response.json()["data"]["viewer"]
            entity = viewer["entity"]
            print(f"Authenticated username: {viewer['username']}; default entity: {entity}")
        except Exception as error:
            raise SystemExit(
                f"Could not discover entity ({type(error).__name__}); nothing written"
            ) from None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", entity):
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
