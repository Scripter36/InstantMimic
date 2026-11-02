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
