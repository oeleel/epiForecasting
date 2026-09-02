"""Unit tests for src.config dict-based helpers (Milestone 2 additions)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_GROUPS,
    get_default_config,
    get_config_value,
    set_config_value,
)


def test_default_config_has_all_sections():
    cfg = get_default_config()
    expected = {"xgboost", "features", "target", "floor",
                "quantiles", "clustering", "sample_weights", "model", "data"}
    assert set(cfg.keys()) == expected


def test_default_config_is_independent_per_call():
    a = get_default_config()
    b = get_default_config()
    a["xgboost"]["max_depth"] = 999
    assert b["xgboost"]["max_depth"] != 999


def test_default_config_is_json_serializable():
    cfg = get_default_config()
    s = json.dumps(cfg)
    back = json.loads(s)
    assert back["xgboost"]["max_depth"] == cfg["xgboost"]["max_depth"]


def test_get_config_value_dotted_path():
    cfg = get_default_config()
    assert get_config_value(cfg, "xgboost.max_depth") == cfg["xgboost"]["max_depth"]
    assert get_config_value(cfg, "features.groups_enabled.lag") is True
    assert get_config_value(cfg, "sample_weights.by_phase") == {}


def test_get_config_value_missing_raises():
    cfg = get_default_config()
    try:
        get_config_value(cfg, "xgboost.does_not_exist")
        assert False, "should have raised"
    except KeyError:
        pass


def test_set_config_value_returns_new_dict():
    cfg = get_default_config()
    new_cfg = set_config_value(cfg, "xgboost.max_depth", 7)
    assert new_cfg["xgboost"]["max_depth"] == 7
    assert cfg["xgboost"]["max_depth"] != 7  # original unchanged


def test_set_config_value_creates_leaf_keys():
    cfg = get_default_config()
    new_cfg = set_config_value(cfg, "sample_weights.by_phase", {"peak": 2.0})
    assert new_cfg["sample_weights"]["by_phase"] == {"peak": 2.0}
    assert cfg["sample_weights"]["by_phase"] == {}


def test_feature_groups_match_default_groups_enabled():
    cfg = get_default_config()
    assert set(cfg["features"]["groups_enabled"].keys()) == set(FEATURE_GROUPS.keys())


ALL = [
    test_default_config_has_all_sections,
    test_default_config_is_independent_per_call,
    test_default_config_is_json_serializable,
    test_get_config_value_dotted_path,
    test_get_config_value_missing_raises,
    test_set_config_value_returns_new_dict,
    test_set_config_value_creates_leaf_keys,
    test_feature_groups_match_default_groups_enabled,
]


def main():
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print()
    print("OK" if failed == 0 else f"{failed} FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
