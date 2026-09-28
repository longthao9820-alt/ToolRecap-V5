"""Schema loader and validator access."""

import json
from typing import Any, Dict

import jsonschema

from toolrecap_v4.runtime import find_resource

_SCHEMA_CACHE: Dict[str, Any] | None = None
_VALIDATOR_CACHE: jsonschema.protocols.Validator | None = None


def get_project_schema() -> Dict[str, Any]:
    """Load and return the Recap Project Schema v3 dictionary."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema_path = find_resource(
            "toolrecap_v4/schemas/recap_v3_schema.json",
            "schemas/recap_v3_schema.json",
        )
        with schema_path.open("r", encoding="utf-8") as handle:
            _SCHEMA_CACHE = json.load(handle)
    return _SCHEMA_CACHE


def get_schema_validator() -> jsonschema.protocols.Validator:
    """Get or create the jsonschema Draft 2020-12 validator instance."""
    global _VALIDATOR_CACHE
    if _VALIDATOR_CACHE is None:
        schema = get_project_schema()
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        _VALIDATOR_CACHE = validator_cls(schema)
    return _VALIDATOR_CACHE
