"""Only supported trainer extensions cross the semantic configuration boundary."""

import pytest
from config_fixtures import valid_config

from instantmimic.config.loader import load_semantic_config
from instantmimic.config.validation import ConfigSchemaError


@pytest.mark.parametrize("field", ["curriculum", "unsupported_extension"])
def test_trainer_rejects_unavailable_extensions(field):
    cfg = valid_config()
    cfg["trainer"][field] = {}
    with pytest.raises(ConfigSchemaError, match=rf"trainer\.{field}: unknown key"):
        load_semantic_config(cfg)


def test_removed_fullgraph_update_compile_key_is_rejected():
    cfg = valid_config()
    cfg["runtime"]["compile_update_fullgraph"] = True
    with pytest.raises(
        ConfigSchemaError,
        match=r"runtime\.compile_update_fullgraph: was removed; full update "
        r"compilation is implied by runtime\.compile together with "
        r"runtime\.update_cuda_graph",
    ):
        load_semantic_config(cfg)
