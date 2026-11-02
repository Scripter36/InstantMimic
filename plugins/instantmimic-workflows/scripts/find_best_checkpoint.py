#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path


def _candidate_checkpoints(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    direct = path / "checkpoints" / "best_agent.pt"
    if direct.is_file():
        return [direct]
    if path.name == "checkpoints":
        best = path / "best_agent.pt"
        if best.is_file():
            return [best]
    return [p for p in path.rglob("checkpoints/best_agent.pt") if p.is_file()]


def find_candidates(queries: list[str]) -> list[Path]:
    if not queries:
        queries = ["logs/ppo", "logs"]
    candidates: list[Path] = []
    for query in queries:
        path = Path(query).expanduser()
        matches = _candidate_checkpoints(path)
        if not matches and any(ch in query for ch in "*?[]"):
            for globbed in glob.glob(query):
                matches.extend(_candidate_checkpoints(Path(globbed)))
        candidates.extend(matches)
    deduped = {p.resolve(): p for p in candidates}
    return sorted(deduped.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the newest best_agent.pt under logs or supplied paths."
    )
    parser.add_argument(
        "queries", nargs="*", help="Run dir, log root, checkpoint, or glob"
    )
    parser.add_argument(
        "--all", action="store_true", help="Print all matches newest first"
    )
    args = parser.parse_args()

    candidates = find_candidates(args.queries)
    if not candidates:
        return 1
    selected = candidates if args.all else candidates[:1]
    for checkpoint in selected:
        print(checkpoint.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
