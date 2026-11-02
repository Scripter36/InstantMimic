"""One way to say what an experiment is.

Both objective modes propose the same thing: a set of named candidates, each a
per-motion override set, each with a recorded parent. A scalar sweep is not a
different kind of proposal -- it is one axis that ``expand`` turns into N
candidates before anything runs.

The pieces:

* ``OverrideSet`` -- canonical ``path=value`` strings, merged right-wins. The
  ``+``/``++``/``~`` prefixes live inside this class and nowhere else.
* ``MotionRoot`` / ``RootSpec`` -- what a task trains, before any proposal.
* ``Axis`` -- one scalar sweep dimension, by value or by named variant.
* ``Candidate`` -- one node of the search: its delta, its resolved effective
  overrides, and the node it branched from.
* ``Proposal`` -- what one cycle asks the evaluator to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import DictConfig, ListConfig

# (cycle, candidate name). Replaces the scalar cycle index and the Pareto
# (experiment, eval_index) pair, both of which could collide across cycles.
NodeId = tuple[int, str]
RootRef = Literal["root", "recommended"]

_OVERRIDE_PREFIX_RE = re.compile(r"^[+~]{0,2}")


def _strip_prefix(override: str) -> tuple[str, str]:
    text = str(override).strip()
    match = _OVERRIDE_PREFIX_RE.match(text)
    prefix = match.group(0) if match else ""
    return prefix, text[len(prefix) :]


def override_path(override: str) -> str:
    """The Hydra path an override writes, without prefix or value."""

    _, remainder = _strip_prefix(override)
    return remainder.split("=", 1)[0].strip()


@dataclass(frozen=True)
class OverrideSet:
    """A candidate's Hydra overrides for one motion, in application order."""

    items: tuple[str, ...] = ()

    @classmethod
    def of(cls, values: Any) -> OverrideSet:
        if values is None:
            return cls(())
        if isinstance(values, OverrideSet):
            return values
        if isinstance(values, (str, bytes)):
            raise ValueError("An override set must be a sequence, not a single string.")
        if isinstance(values, (dict, DictConfig)):
            return cls(tuple(f"{key}={value}" for key, value in values.items()))
        return cls(tuple(str(item).strip() for item in values if str(item).strip()))

    def paths(self) -> tuple[str, ...]:
        return tuple(override_path(item) for item in self.items)

    def as_map(self) -> dict[str, str]:
        """Last write per path wins, matching Hydra's own override order."""

        mapping: dict[str, str] = {}
        for item in self.items:
            path, _, value = item.partition("=")
            mapping[override_path(path)] = value
        return mapping

    def merge(self, other: OverrideSet) -> OverrideSet:
        """Apply ``other`` on top of this set; a repeated path is replaced."""

        replaced = set(other.paths())
        kept = tuple(item for item in self.items if override_path(item) not in replaced)
        return OverrideSet(kept + other.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class MotionRoot:
    motion: str
    config_path: Path
    config_name: str


@dataclass(frozen=True)
class RootSpec:
    """The task's fixed frame: what trains, with what, under which seeds."""

    motions: dict[str, MotionRoot]
    fixed_overrides: OverrideSet
    seeds: tuple[int, ...]
    parameter_space: tuple[str, ...] | None
    forbidden_prefixes: tuple[str, ...]
    lock_motion_configs: bool = True

    def motion_names(self) -> tuple[str, ...]:
        return tuple(self.motions)


@dataclass(frozen=True)
class Axis:
    """One scalar sweep dimension.

    ``values`` sweeps a single path; ``variants`` sweeps named bundles, each of
    which sets several paths at once. Exactly one of the two is set.
    """

    name: str
    values: tuple[Any, ...] | None = None
    variants: tuple[tuple[str, dict[str, Any]], ...] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("An axis requires a non-empty name.")
        if (self.values is None) == (self.variants is None):
            raise ValueError(
                f"Axis '{self.name}' needs exactly one of `values` or `variants`."
            )
        if self.values is not None and not self.values:
            raise ValueError(f"Axis '{self.name}' has an empty value list.")
        if self.variants is not None:
            if not self.variants:
                raise ValueError(f"Axis '{self.name}' has an empty variant list.")
            labels = [label for label, _ in self.variants]
            if len(set(labels)) != len(labels):
                raise ValueError(f"Axis '{self.name}' repeats a variant label.")

    def labels(self) -> tuple[str, ...]:
        if self.variants is not None:
            return tuple(label for label, _ in self.variants)
        assert self.values is not None
        return tuple(str(value) for value in self.values)

    def overrides_for(self, label: str) -> OverrideSet:
        if self.variants is not None:
            for variant_label, overrides in self.variants:
                if variant_label == label:
                    return OverrideSet.of(overrides)
            raise ValueError(f"Axis '{self.name}' has no variant '{label}'.")
        return OverrideSet((f"{self.name}={label}",))


@dataclass(frozen=True)
class Candidate:
    """One node of the search tree, resolved against its base."""

    name: str
    rationale: str = ""
    parent: NodeId | RootRef = "root"
    delta: dict[str, OverrideSet] = field(default_factory=dict)
    effective: dict[str, OverrideSet] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "parent": list(self.parent)
            if isinstance(self.parent, tuple)
            else self.parent,
            "motions": {
                motion: {"overrides": list(overrides.items)}
                for motion, overrides in self.delta.items()
            },
            "effective_overrides": {
                motion: list(overrides.items)
                for motion, overrides in self.effective.items()
            },
        }


@dataclass(frozen=True)
class Proposal:
    """What one cycle runs, and what it branched from."""

    base: NodeId | RootRef = "root"
    axis: Axis | None = None
    candidates: tuple[Candidate, ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "base": list(self.base) if isinstance(self.base, tuple) else self.base,
            "experiments": [candidate.as_dict() for candidate in self.candidates],
        }
        if self.axis is not None:
            payload["axis"] = {
                "name": self.axis.name,
                **(
                    {"values": list(self.axis.values)}
                    if self.axis.values is not None
                    else {
                        "variants": [
                            {"label": label, "overrides": dict(overrides)}
                            for label, overrides in (self.axis.variants or ())
                        ]
                    }
                ),
            }
        return payload


@dataclass
class Node:
    """A candidate plus what running it produced."""

    id: NodeId
    candidate: Candidate
    status: Literal["completed", "failed"] = "completed"
    failure: str | None = None
    evidence_path: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    verdict: dict[str, Any] = field(default_factory=dict)


def _resolve_repo_path(root: Path, value: str) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def load_root(task_cfg: DictConfig, *, repo_root: Path) -> RootSpec:
    """Read the task's fixed frame out of the unified schema."""

    batch_cfg = task_cfg.get("batch", None)
    if batch_cfg is None:
        raise ValueError("Task is missing the `batch` section.")
    motions_cfg = batch_cfg.get("motions", None)
    if not motions_cfg:
        raise ValueError("`batch.motions` must define at least one motion.")

    motions: dict[str, MotionRoot] = {}
    for motion_name in motions_cfg:
        entry = motions_cfg[motion_name]
        motions[str(motion_name)] = MotionRoot(
            motion=str(motion_name),
            config_path=_resolve_repo_path(repo_root, entry["config_path"]),
            config_name=str(entry["config_name"]),
        )

    parameter_space_raw = task_cfg.get("parameter_space", None)
    parameter_space = (
        None
        if parameter_space_raw is None
        else tuple(str(item).strip() for item in parameter_space_raw)
    )
    forbidden_raw = task_cfg.get("forbidden_prefixes", None) or []
    return RootSpec(
        motions=motions,
        fixed_overrides=OverrideSet.of(
            batch_cfg.get("fixed_train_overrides", None) or []
        ),
        seeds=tuple(int(seed) for seed in (batch_cfg.get("seeds", None) or [])),
        parameter_space=parameter_space,
        forbidden_prefixes=tuple(
            str(item).strip() for item in forbidden_raw if str(item).strip()
        ),
        lock_motion_configs=bool(batch_cfg.get("lock_motion_configs", True)),
    )


def expand(
    axis: Axis,
    base: dict[str, OverrideSet],
    *,
    parent: NodeId | RootRef = "root",
) -> tuple[Candidate, ...]:
    """Turn one sweep axis into one candidate per value, over every motion.

    This is what makes a scalar cycle the same shape as a Pareto cycle: the
    planner names an axis, the controller enumerates it, and the evaluator only
    ever sees candidates.
    """

    candidates: list[Candidate] = []
    for label in axis.labels():
        delta_set = axis.overrides_for(label)
        delta = {motion: delta_set for motion in base}
        effective = {motion: base[motion].merge(delta_set) for motion in base}
        candidates.append(
            Candidate(
                name=f"{_slug(axis.name)}__{_slug(label)}",
                rationale=f"{axis.name}={label}",
                parent=parent,
                delta=delta,
                effective=effective,
            )
        )
    return tuple(candidates)


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return text or "value"


def validate_overrides(
    root: RootSpec,
    *,
    candidate_name: str,
    motion_name: str,
    overrides: OverrideSet,
) -> list[str]:
    """Check one candidate's overrides against the task's search boundary."""

    errors: list[str] = []
    for override in overrides.items:
        path = override_path(override)
        for prefix in root.forbidden_prefixes:
            if path == prefix or path.startswith(f"{prefix}."):
                errors.append(
                    f"Experiment '{candidate_name}' motion '{motion_name}' uses "
                    f"forbidden override '{override}'. Forbidden prefix: '{prefix}'."
                )
                break
        else:
            if root.parameter_space is not None and path not in root.parameter_space:
                errors.append(
                    f"Experiment '{candidate_name}' motion '{motion_name}' uses "
                    f"override '{override}', whose path '{path}' is not in the "
                    "exact parameter_space list."
                )
    return errors


def load_proposal(
    proposal_path: Path,
    root: RootSpec,
    *,
    min_candidates: int | None = None,
    max_candidates: int | None = None,
) -> Proposal:
    """Read ``cycles/NNN/proposal.yaml`` into a resolved ``Proposal``."""

    payload = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Proposal '{proposal_path}' must be a mapping.")
    experiments_raw = payload.get("experiments", None)
    if isinstance(experiments_raw, ListConfig):
        experiments_raw = list(experiments_raw)
    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise ValueError("Proposal must contain a non-empty 'experiments' list.")
    if max_candidates is not None and len(experiments_raw) > int(max_candidates):
        raise ValueError(
            f"Proposal has {len(experiments_raw)} experiments but task allows at "
            f"most {int(max_candidates)}."
        )
    if min_candidates is not None and len(experiments_raw) < int(min_candidates):
        raise ValueError(
            f"Proposal has {len(experiments_raw)} experiments but task requires at "
            f"least {int(min_candidates)}."
        )

    base = _parse_ref(payload.get("base", "root"))
    known_motions = set(root.motions)
    seen: set[str] = set()
    candidates: list[Candidate] = []
    errors: list[str] = []
    for item in experiments_raw:
        if not isinstance(item, dict):
            raise ValueError("Each proposal experiment must be a mapping.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each proposal experiment requires a non-empty name.")
        if name in seen:
            raise ValueError(f"Duplicate experiment name '{name}' in proposal.")
        seen.add(name)

        motions_raw = item.get("motions", None)
        if not isinstance(motions_raw, dict):
            raise ValueError(f"Experiment '{name}' must contain a motions mapping.")
        unknown = sorted(set(motions_raw) - known_motions)
        if unknown:
            raise ValueError(
                f"Experiment '{name}' has unknown motions: {', '.join(unknown)}"
            )
        missing = sorted(known_motions - set(motions_raw))
        if missing:
            raise ValueError(
                f"Experiment '{name}' is missing motions: {', '.join(missing)}"
            )

        delta: dict[str, OverrideSet] = {}
        for motion_name, motion_payload in motions_raw.items():
            if not isinstance(motion_payload, dict):
                raise ValueError(
                    f"Experiment '{name}' motion '{motion_name}' must be a mapping."
                )
            overrides = OverrideSet.of(motion_payload.get("overrides", None))
            errors.extend(
                validate_overrides(
                    root,
                    candidate_name=name,
                    motion_name=str(motion_name),
                    overrides=overrides,
                )
            )
            delta[str(motion_name)] = overrides

        candidates.append(
            Candidate(
                name=name,
                rationale=str(item.get("rationale", "")).strip(),
                parent=_parse_ref(item.get("parent", base)),
                delta=delta,
                effective=dict(delta),
            )
        )
    if errors:
        raise ValueError(errors[0])
    return Proposal(base=base, axis=None, candidates=tuple(candidates))


def _parse_ref(value: Any) -> NodeId | RootRef:
    if value in (None, "", "root"):
        return "root"
    if value == "recommended":
        return "recommended"
    if isinstance(value, (list, tuple, ListConfig)) and len(list(value)) == 2:
        cycle, name = list(value)
        return (int(cycle), str(name))
    if isinstance(value, dict):
        return (int(value["cycle"]), str(value["candidate"]))
    raise ValueError(f"Unsupported node reference: {value!r}")


def load_allowed_paths(task_cfg: DictConfig) -> tuple[str, ...] | None:
    """The exact Hydra paths a proposal may touch, or None when unrestricted."""

    raw = task_cfg.get("parameter_space", None)
    if raw is None:
        return None
    if not isinstance(raw, (list, ListConfig)):
        raise ValueError(
            "parameter_space must be a YAML list of exact Hydra parameter paths."
        )
    paths = tuple(str(item).strip() for item in raw)
    if not paths or any(not path for path in paths):
        raise ValueError("parameter_space must contain non-empty Hydra paths.")
    if len(set(paths)) != len(paths):
        raise ValueError("parameter_space contains duplicate Hydra paths.")
    if any("=" in path or path.startswith(("+", "~")) for path in paths):
        raise ValueError(
            "parameter_space entries must be paths, not Hydra override expressions."
        )
    return paths


def validate_candidate_configs(
    root: RootSpec, candidates: tuple[Candidate, ...]
) -> list[str]:
    """Compose and validate exactly what each candidate will train, without CUDA."""
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import HydraException
    from omegaconf.errors import OmegaConfBaseException

    from instantmimic.config.composition import compose_config_file
    from instantmimic.config.config import load_semantic_config

    errors: list[str] = []
    # The CLI may already own Hydra. Composition uses the motion's search root;
    # restore the CLI instance afterwards so its context and tracking stay intact.
    global_hydra = GlobalHydra.instance()
    previous = global_hydra.hydra
    global_hydra.clear()
    try:
        for candidate in candidates:
            for motion, motion_root in root.motions.items():
                settings = root.fixed_overrides.merge(candidate.effective[motion])
                settings = settings.merge(OverrideSet.of([f"seed={root.seeds[0]}"]))
                try:
                    cfg = compose_config_file(
                        motion_root.config_path / f"{motion_root.config_name}.yaml",
                        overrides=settings.items,
                    )
                    load_semantic_config(cfg)
                except (
                    HydraException,
                    OmegaConfBaseException,
                    ValueError,
                    OSError,
                ) as exc:
                    errors.append(
                        f"experiment '{candidate.name}' motion '{motion}': {exc}"
                    )
    finally:
        global_hydra.hydra = previous
    return errors
