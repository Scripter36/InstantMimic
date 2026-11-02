#!/usr/bin/env python3
"""Audit runnable checkout configurations without loading the GPU runtime."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from instantmimic.config.composition import compose_config_file
from instantmimic.config.loader import load_semantic_config
from instantmimic.hpo.run_context import split_hpo_runtime_cfg
from instantmimic.hpo.schema import validate_task
from instantmimic.rl.build_specs import lower_agent_build_spec
from instantmimic.rl.models.musubi_actor_critic import MusubiUnavailableError

ROOT = Path(__file__).resolve().parents[3]


def _runtime_asset_declarations() -> dict[str, str]:
    path = ROOT / "data/runtime-assets.json"
    if not path.exists():
        return {}
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or set(values) != {"schema", "files"}:
        raise ValueError("Runtime assets must declare schema and files")
    if (
        type(values["schema"]) is not int
        or values["schema"] != 1
        or not isinstance(values["files"], dict)
    ):
        raise ValueError("Runtime assets must use schema 1 and a files mapping")
    for name, prepare in values["files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or relative.as_posix() != name
        ):
            raise ValueError(
                "Runtime asset paths must be canonical checkout-relative paths"
            )
        if (
            not relative.parts
            or relative.parts[0] != "data"
            or not isinstance(prepare, str)
            or not prepare.strip()
        ):
            raise ValueError(
                "Runtime assets need a data path and preparation instructions"
            )
    return values["files"]


def _checkout_asset(
    value: str,
    *,
    pattern: bool = False,
    declarations: dict[str, str] | None = None,
    deferred: set[str] | None = None,
) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"authored asset must be checkout-relative: {value}")
    matches = (
        [Path(p) for p in glob.glob(str(ROOT / path), recursive=True)]
        if pattern
        else [ROOT / path]
    )
    if any(not p.resolve().is_relative_to(ROOT) for p in matches):
        raise ValueError(f"checkout asset is outside the checkout: {value}")
    if not matches or any(not p.is_file() for p in matches):
        prepare = (declarations or {}).get(value)
        if prepare is not None and deferred is not None:
            deferred.add(value)
            return
        hint = f"; prepare with {prepare}" if prepare is not None else ""
        raise ValueError(f"checkout asset is missing: {value}{hint}")


def _validate_assets(
    cfg: DictConfig, *, declarations: dict[str, str], deferred: set[str] | None
) -> None:
    for key in (
        "env.sim.mjcf_path",
        "env.motion.file",
        "env.point_goal_marker_mesh_path",
        "env.point_goal_marker_texture_path",
        "env.action.policy_output.stable_pd.mjcf_path",
    ):
        value = OmegaConf.select(cfg, key)
        if value is not None:
            _checkout_asset(str(value), declarations=declarations, deferred=deferred)
    motion_dir = OmegaConf.select(cfg, "env.motion_data_dir")
    if motion_dir:
        clips = OmegaConf.select(cfg, "env.motion_clips") or [
            OmegaConf.select(cfg, "env.motion_clip")
        ]
        for clip in clips:
            if not clip:
                raise ValueError("motion_data_dir requires named motion clips")
            for array in ("qpos", "qvel", "xipos", "ximat"):
                _checkout_asset(
                    f"{motion_dir}/{array}/{clip}.npy",
                    declarations=declarations,
                    deferred=deferred,
                )
    source = OmegaConf.select(cfg, "env.motion.source")
    sources = [source] if isinstance(source, str) else list(source or ())
    for value in sources:
        if not str(value).strip():
            continue
        _checkout_asset(
            str(value), pattern=True, declarations=declarations, deferred=deferred
        )


def audit(*, allow_unprepared_assets: bool = False) -> dict[str, object]:
    declarations = _runtime_asset_declarations()
    deferred = set() if allow_unprepared_assets else None
    checked: list[str] = []
    placeholders: list[str] = []
    failures: list[dict[str, str]] = []
    for path in sorted((ROOT / "config").rglob("*.yaml")):
        relative = path.relative_to(ROOT / "config")
        if relative.parts[0] == "hpo":
            if len(relative.parts) != 2:
                continue  # Nested files are Hydra preset groups.
            hpo = True
        else:
            if "presets" in relative.parts or path.stem in {"base", "common"}:
                continue
            hpo = False
        name = relative.with_suffix("").as_posix()
        checked.append(name)
        try:
            cfg = compose_config_file(path)
            if hpo:
                task, runtime = split_hpo_runtime_cfg(cfg)
                errors = validate_task(path, task, runtime)
                if errors:
                    raise ValueError("; ".join(errors))
                for motion in task.batch.motions.values():
                    train_file = Path(str(motion.config_path)) / (
                        str(motion.config_name) + ".yaml"
                    )
                    _checkout_asset(train_file.as_posix())
            else:
                initialization = OmegaConf.select(cfg, "trainer.initialization")
                if initialization is not None and OmegaConf.is_missing(
                    initialization, "checkpoint"
                ):
                    initialization.checkpoint = "runtime-checkpoint-placeholder.pt"
                    placeholders.append(name)
                if OmegaConf.select(
                    cfg, "env.low_level_config_autoload"
                ) and not OmegaConf.select(cfg, "env.low_level_checkpoint"):
                    placeholders.append(name)
                _validate_assets(cfg, declarations=declarations, deferred=deferred)
                try:
                    semantic = load_semantic_config(cfg)
                    lower_agent_build_spec(semantic)
                except MusubiUnavailableError:
                    # Composition and assets are checked above; execution
                    # raises its own error for this method.
                    pass
        except Exception as exc:  # noqa: BLE001 - CLI reports all config failures.
            failures.append(
                {"config": name, "error_type": type(exc).__name__, "message": str(exc)}
            )
    if not checked:
        failures.append(
            {
                "config": "",
                "error_type": "ValueError",
                "message": "no runnable configurations found",
            }
        )
    return {
        "status": "failed" if failures else "passed",
        "validated": len(checked),
        "configurations": checked,
        "checkpoint_placeholders": placeholders,
        "asset_status": "validation-failed"
        if failures
        else ("requires-preparation" if deferred else "available"),
        "deferred_assets": [
            {"path": name, "prepare": declarations[name]}
            for name in sorted(deferred or ())
        ],
        "failures": failures,
        "scope": "Hydra composition and authored assets for every configuration; semantic/build contracts for runnable methods; no simulation or checkpoint loading",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Emit the full JSON report."
    )
    parser.add_argument(
        "--allow-unprepared-assets",
        action="store_true",
        help="Defer only explicitly declared runtime assets and report their preparation instructions.",
    )
    args = parser.parse_args()
    try:
        report = audit(allow_unprepared_assets=args.allow_unprepared_assets)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Configuration audit {report['status']}: {report['validated']} checked.")
        for failure in report["failures"]:
            print(f"{failure['config']}: {failure['message']}")
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
