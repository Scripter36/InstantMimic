#!/usr/bin/env python3
"""Build immutable migration review packets and derive mechanical verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn

SCHEMA_VERSION = 1
RESULTS = {"pass", "fail", "missing"}
ACCEPTANCE_CLASSES = {"exact", "tolerant", "statistical", "unsupported"}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")


def write_json(value: Any, output: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(f"{label} keys must be {sorted(expected)}; got {sorted(actual)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def packet_digest(packet: Any) -> str:
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_packet(value: Any) -> dict[str, Any]:
    packet = require_keys(
        value,
        {
            "schema_version",
            "original_user_request",
            "source_identity",
            "required_gates",
            "artifacts",
        },
        "packet",
    )
    validate_contract({key: packet[key] for key in packet if key != "artifacts"})
    artifacts = packet["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        fail("artifacts must be a non-empty list")
    seen: set[str] = set()
    for index, raw_artifact in enumerate(artifacts):
        artifact = require_keys(
            raw_artifact,
            {"id", "kind", "path", "sha256", "size_bytes"},
            f"artifact[{index}]",
        )
        if not all(
            isinstance(artifact[key], str) and artifact[key].strip()
            for key in ("id", "kind", "path")
        ):
            fail(f"artifact[{index}] id, kind, and path must be non-empty strings")
        if artifact["id"] in seen:
            fail(f"duplicate artifact id: {artifact['id']}")
        seen.add(artifact["id"])
        digest = artifact["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            fail(f"artifact[{index}] has an invalid sha256")
        if not isinstance(artifact["size_bytes"], int) or artifact["size_bytes"] < 0:
            fail(f"artifact[{index}] has an invalid size_bytes")
    return packet


def validate_contract(value: Any) -> dict[str, Any]:
    contract = require_keys(
        value,
        {
            "schema_version",
            "original_user_request",
            "source_identity",
            "required_gates",
        },
        "contract",
    )
    if contract["schema_version"] != SCHEMA_VERSION:
        fail(f"contract schema_version must be {SCHEMA_VERSION}")
    if (
        not isinstance(contract["original_user_request"], str)
        or not contract["original_user_request"].strip()
    ):
        fail("original_user_request must be a non-empty string")
    source = require_keys(
        contract["source_identity"], {"kind", "value"}, "source_identity"
    )
    if not all(isinstance(source[key], str) and source[key].strip() for key in source):
        fail("source_identity values must be non-empty strings")
    gates = contract["required_gates"]
    if not isinstance(gates, list) or not gates:
        fail("required_gates must be a non-empty list")
    seen: set[str] = set()
    for index, raw_gate in enumerate(gates):
        gate = require_keys(raw_gate, {"id", "claim", "acceptance"}, f"gate[{index}]")
        if not all(isinstance(gate[key], str) and gate[key].strip() for key in gate):
            fail(f"gate[{index}] values must be non-empty strings")
        if gate["acceptance"] not in ACCEPTANCE_CLASSES:
            fail(f"gate[{index}] has an invalid acceptance: {gate['acceptance']}")
        if gate["id"] in seen:
            fail(f"duplicate gate id: {gate['id']}")
        seen.add(gate["id"])
    return contract


def artifact_arg(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("artifact must be ID=KIND=PATH")
    return parts[0], parts[1], Path(parts[2])


def build_packet(args: argparse.Namespace) -> None:
    contract = validate_contract(read_json(args.contract))
    artifacts = []
    seen: set[str] = set()
    for artifact_id, kind, supplied_path in args.artifact:
        if artifact_id in seen:
            fail(f"duplicate artifact id: {artifact_id}")
        seen.add(artifact_id)
        path = supplied_path.resolve()
        if not path.is_file():
            fail(f"artifact is not a regular file: {path}")
        try:
            display_path = str(path.relative_to(args.repo_root.resolve()))
        except ValueError:
            display_path = str(path)
        artifacts.append(
            {
                "id": artifact_id,
                "kind": kind,
                "path": display_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not artifacts:
        fail("at least one artifact is required")
    packet = dict(contract)
    packet["artifacts"] = artifacts
    write_json(packet, args.output)


def show_digest(args: argparse.Namespace) -> None:
    packet = validate_packet(read_json(args.packet))
    print(packet_digest(packet))


def verify_artifacts(packet: dict[str, Any], repo_root: Path) -> list[str]:
    verified = []
    for artifact in packet["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = repo_root.resolve() / path
        if not path.is_file():
            fail(f"artifact is missing: {path}")
        if (
            path.stat().st_size != artifact["size_bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            fail(f"artifact changed after packet creation: {path}")
        verified.append(artifact["id"])
    return verified


def verify_packet(args: argparse.Namespace) -> None:
    packet = validate_packet(read_json(args.packet))
    verified = verify_artifacts(packet, args.repo_root)
    write_json(
        {"packet_sha256": packet_digest(packet), "verified_artifacts": verified}, None
    )


def derive_verdict(args: argparse.Namespace) -> None:
    packet = validate_packet(read_json(args.packet))
    verify_artifacts(packet, args.repo_root)
    result = require_keys(
        read_json(args.result),
        {"schema_version", "packet_sha256", "gates"},
        "review result",
    )
    digest = packet_digest(packet)
    if result["schema_version"] != SCHEMA_VERSION or result["packet_sha256"] != digest:
        fail("review result schema or packet digest mismatch")
    artifact_ids = {item["id"] for item in packet["artifacts"]}
    required_ids = [item["id"] for item in packet["required_gates"]]
    gates = result["gates"]
    if not isinstance(gates, list):
        fail("review result gates must be a list")
    reviewed: dict[str, dict[str, Any]] = {}
    for index, raw_gate in enumerate(gates):
        gate = require_keys(
            raw_gate, {"id", "result", "evidence", "finding"}, f"result gate[{index}]"
        )
        if gate["id"] in reviewed:
            fail(f"duplicate result gate id: {gate['id']}")
        if gate["id"] not in required_ids:
            fail(f"unknown result gate id: {gate['id']}")
        if gate["result"] not in RESULTS:
            fail(f"invalid result for gate {gate['id']}: {gate['result']}")
        if not isinstance(gate["evidence"], list) or not all(
            isinstance(item, str) and item in artifact_ids for item in gate["evidence"]
        ):
            fail(f"gate {gate['id']} references unknown evidence")
        if gate["result"] != "missing" and not gate["evidence"]:
            fail(f"gate {gate['id']} needs evidence for {gate['result']}")
        if not isinstance(gate["finding"], str) or not gate["finding"].strip():
            fail(f"gate {gate['id']} needs a finding")
        reviewed[gate["id"]] = gate
    missing_ids = [gate_id for gate_id in required_ids if gate_id not in reviewed]
    if missing_ids:
        fail(f"review omitted required gates: {missing_ids}")
    ordered = [reviewed[gate_id] for gate_id in required_ids]
    values = [gate["result"] for gate in ordered]
    verdict = (
        "changes-required"
        if "fail" in values
        else "unverifiable"
        if "missing" in values
        else "accept"
    )
    write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "packet_sha256": digest,
            "verdict": verdict,
            "counts": {value: values.count(value) for value in sorted(RESULTS)},
            "gates": ordered,
        },
        args.output,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    packet = commands.add_parser("packet", help="build a hashed raw-evidence packet")
    packet.add_argument("--contract", type=Path, required=True)
    packet.add_argument("--repo-root", type=Path, required=True)
    packet.add_argument(
        "--artifact",
        action="append",
        type=artifact_arg,
        default=[],
        metavar="ID=KIND=PATH",
    )
    packet.add_argument("--output", type=Path)
    packet.set_defaults(run=build_packet)
    digest = commands.add_parser("digest", help="print the canonical packet digest")
    digest.add_argument("packet", type=Path)
    digest.set_defaults(run=show_digest)
    verify = commands.add_parser(
        "verify", help="verify packet structure and artifact hashes"
    )
    verify.add_argument("--packet", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.set_defaults(run=verify_packet)
    verdict = commands.add_parser(
        "verdict", help="validate gate results and derive the verdict"
    )
    verdict.add_argument("--packet", type=Path, required=True)
    verdict.add_argument("--result", type=Path, required=True)
    verdict.add_argument("--repo-root", type=Path, default=Path.cwd())
    verdict.add_argument("--output", type=Path)
    verdict.set_defaults(run=derive_verdict)
    return root


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
