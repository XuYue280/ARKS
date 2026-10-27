#!/usr/bin/env python3
"""Tiny stage-stamp helper used by arks_pipeline.sh for resume support.

A stage directory is considered complete when it contains
``.arks_stage.json`` with ``status == "complete"`` and the ``hash`` recorded
at completion equals the hash computed from the stage's current inputs and
configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STAMP = ".arks_stage.json"


def read(path: Path) -> dict:
    stamp = path / STAMP
    if not stamp.is_file():
        return {}
    try:
        return json.loads(stamp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write(path: Path, data: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / (STAMP + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path / STAMP)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    get = sub.add_parser("get", help="print one field of a stamp (empty if missing)")
    get.add_argument("dir", type=Path)
    get.add_argument("key")

    check = sub.add_parser("check", help="exit 0 when the stage is complete with this hash")
    check.add_argument("dir", type=Path)
    check.add_argument("hash")

    begin = sub.add_parser("begin")
    begin.add_argument("dir", type=Path)
    begin.add_argument("--stage", required=True)
    begin.add_argument("--hash", required=True)
    begin.add_argument("--inputs", default="")

    finish = sub.add_parser("finish")
    finish.add_argument("dir", type=Path)
    finish.add_argument("--extra", nargs="*", default=[], help="key=value pairs")

    setter = sub.add_parser("set")
    setter.add_argument("dir", type=Path)
    setter.add_argument("pairs", nargs="+", help="key=value pairs")

    args = parser.parse_args()
    if args.command == "get":
        data = read(args.dir)
        value = data.get(args.key, "")
        print(value if not isinstance(value, (dict, list)) else json.dumps(value))
    elif args.command == "check":
        data = read(args.dir)
        ok = data.get("status") == "complete" and data.get("hash") == args.hash
        sys.exit(0 if ok else 1)
    elif args.command == "begin":
        write(
            args.dir,
            {
                "stage": args.stage,
                "hash": args.hash,
                "status": "running",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_epoch": time.time(),
                "inputs": args.inputs,
            },
        )
    elif args.command == "finish":
        data = read(args.dir)
        data["status"] = "complete"
        data["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if "started_epoch" in data:
            data["elapsed_seconds"] = round(time.time() - float(data["started_epoch"]), 1)
        for pair in args.extra:
            key, _, value = pair.partition("=")
            data[key] = value
        write(args.dir, data)
    elif args.command == "set":
        data = read(args.dir)
        for pair in args.pairs:
            key, _, value = pair.partition("=")
            data[key] = value
        write(args.dir, data)


if __name__ == "__main__":
    main()
