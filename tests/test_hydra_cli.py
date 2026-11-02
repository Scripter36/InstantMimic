from hydra._internal.utils import get_args_parser

from instantmimic.utils.hydra_cli import hydra_no_output_args


def _overrides(argv: list[str]) -> list[str]:
    parsed, unknown = get_args_parser().parse_known_args(argv)
    assert unknown == []
    return parsed.overrides


def test_artifact_overrides_join_user_overrides_before_trailing_options(
    tmp_path,
) -> None:
    argv = ["--config-name", "task", "a=1", "+b=2", "--cfg", "job", "--resolve"]

    args = hydra_no_output_args(argv, runtime_dir=tmp_path)

    overrides = _overrides(args)
    assert overrides[:2] == ["a=1", "+b=2"]
    assert "hydra.output_subdir=null" in overrides
    assert args[-3:] == ["--cfg", "job", "--resolve"]


def test_artifact_overrides_append_without_user_overrides(tmp_path) -> None:
    args = hydra_no_output_args(
        ["--config-name", "task", "--cfg", "job"], runtime_dir=tmp_path
    )

    assert args[:4] == ["--config-name", "task", "--cfg", "job"]
    assert "hydra.job.chdir=false" in _overrides(args)


def test_user_hydra_artifact_overrides_are_replaced(tmp_path) -> None:
    args = hydra_no_output_args(
        ["hydra.run.dir=elsewhere", "a=1"], runtime_dir=tmp_path
    )

    overrides = _overrides(args)
    assert "hydra.run.dir=elsewhere" not in overrides
    assert overrides[0] == "a=1"
