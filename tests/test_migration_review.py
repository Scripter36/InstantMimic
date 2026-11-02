"""The review boundary rejects invalid contracts and stale raw evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "plugins/instantmimic-workflows/scripts/migration_review.py"
)


def _run(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _packet(root, acceptance="exact"):
    contract = {
        "schema_version": 1,
        "original_user_request": "Port a task and verify its runtime.",
        "source_identity": {"kind": "git_commit", "value": "fixture"},
        "required_gates": [
            {
                "id": "runtime",
                "claim": "Runtime evidence establishes the contract.",
                "acceptance": acceptance,
            }
        ],
    }
    (root / "contract.json").write_text(json.dumps(contract))
    (root / "raw.txt").write_text("measured runtime evidence\n")
    return _run(
        root,
        "packet",
        "--contract",
        root / "contract.json",
        "--repo-root",
        root,
        "--artifact",
        f"runtime=test={root / 'raw.txt'}",
        "--output",
        root / "packet.json",
    )


@pytest.mark.parametrize(
    "acceptance", ["exact", "tolerant", "statistical", "unsupported"]
)
def test_supported_acceptance_class_round_trips(tmp_path, acceptance):
    assert _packet(tmp_path, acceptance).returncode == 0
    verify = _run(
        tmp_path,
        "verify",
        "--packet",
        tmp_path / "packet.json",
        "--repo-root",
        tmp_path,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["verified_artifacts"] == ["runtime"]


def test_unknown_acceptance_class_cannot_enter_packet(tmp_path):
    result = _packet(tmp_path, "approximately")
    assert result.returncode != 0
    assert "acceptance" in result.stderr
    assert not (tmp_path / "packet.json").exists()


def _review(root, result):
    digest = _run(root, "digest", root / "packet.json")
    assert digest.returncode == 0
    review = {
        "schema_version": 1,
        "packet_sha256": digest.stdout.strip(),
        "gates": [
            {
                "id": "runtime",
                "result": result,
                "evidence": [] if result == "missing" else ["runtime"],
                "finding": "Fixture finding.",
            }
        ],
    }
    (root / "review.json").write_text(json.dumps(review))


@pytest.mark.parametrize(
    "result,expected",
    [
        ("pass", "accept"),
        ("fail", "changes-required"),
        ("missing", "unverifiable"),
    ],
)
def test_verdict_derives_from_complete_results(tmp_path, result, expected):
    assert _packet(tmp_path).returncode == 0
    _review(tmp_path, result)
    verdict = _run(
        tmp_path,
        "verdict",
        "--packet",
        tmp_path / "packet.json",
        "--result",
        tmp_path / "review.json",
    )
    assert verdict.returncode == 0, verdict.stderr
    assert json.loads(verdict.stdout)["verdict"] == expected


@pytest.mark.parametrize("mutation", ["replace", "remove"])
def test_verdict_rechecks_evidence_before_accepting(tmp_path, mutation):
    assert _packet(tmp_path).returncode == 0
    _review(tmp_path, "pass")
    if mutation == "replace":
        (tmp_path / "raw.txt").write_text("different runtime evidence\n")
    else:
        (tmp_path / "raw.txt").unlink()
    verdict = _run(
        tmp_path,
        "verdict",
        "--packet",
        tmp_path / "packet.json",
        "--result",
        tmp_path / "review.json",
        "--output",
        tmp_path / "verdict.json",
    )
    assert verdict.returncode != 0
    assert not (tmp_path / "verdict.json").exists()


def test_verdict_supports_explicit_repo_root_from_another_directory(tmp_path):
    assert _packet(tmp_path).returncode == 0
    _review(tmp_path, "pass")
    other = tmp_path / "caller"
    other.mkdir()
    result = _run(
        other,
        "verdict",
        "--packet",
        tmp_path / "packet.json",
        "--result",
        tmp_path / "review.json",
        "--repo-root",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verdict"] == "accept"
