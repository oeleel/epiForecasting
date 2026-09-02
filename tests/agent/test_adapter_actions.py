"""Unit tests for FluForecastAdapter.apply_action + action catalog."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import FluForecastAdapter
from src.config import get_default_config


def _adapter():
    return FluForecastAdapter()


def test_catalog_has_six_actions():
    a = _adapter()
    names = {x["name"] for x in a.get_available_actions()}
    assert names == {
        "adjust_hyperparameter", "reweight_training_samples",
        "toggle_feature", "adjust_floor_constraint",
        "change_target_transform", "stop",
    }


def test_apply_adjust_hyperparameter():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, desc = a.apply_action(
        {"name": "adjust_hyperparameter",
         "params": {"name": "max_depth", "value": 7}},
        cfg,
    )
    assert new_cfg["xgboost"]["max_depth"] == 7
    assert cfg["xgboost"]["max_depth"] != 7  # input unchanged


def test_apply_reweight_phase():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, _ = a.apply_action(
        {"name": "reweight_training_samples",
         "params": {"dimension": "phase", "value": "peak", "weight": 2.5}},
        cfg,
    )
    assert new_cfg["sample_weights"]["by_phase"] == {"peak": 2.5}


def test_apply_toggle_feature():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, _ = a.apply_action(
        {"name": "toggle_feature",
         "params": {"feature_group": "yoy", "enabled": False}},
        cfg,
    )
    assert new_cfg["features"]["groups_enabled"]["yoy"] is False


def test_apply_adjust_floor():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, _ = a.apply_action(
        {"name": "adjust_floor_constraint", "params": {"floor_pct": 0.45}},
        cfg,
    )
    assert new_cfg["floor"]["floor_pct"] == 0.45


def test_apply_change_target_transform():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, _ = a.apply_action(
        {"name": "change_target_transform", "params": {"transform": "raw"}},
        cfg,
    )
    assert new_cfg["target"]["mode"] == "raw"


def test_apply_stop_returns_unchanged_config():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, _ = a.apply_action({"name": "stop", "params": {}}, cfg)
    assert new_cfg == cfg


# ---- guardrail violations ----

def _expect_value_error(action: dict, label: str):
    a = _adapter()
    cfg = get_default_config()
    try:
        a.apply_action(action, cfg)
        raise AssertionError(f"{label} should have raised")
    except ValueError:
        pass


def test_unknown_action_name_raises():
    _expect_value_error({"name": "fly_to_mars", "params": {}}, "unknown name")


def test_max_depth_above_guardrail_raises():
    _expect_value_error(
        {"name": "adjust_hyperparameter",
         "params": {"name": "max_depth", "value": 50}},
        "max_depth=50",
    )


def test_missing_action_params_raises():
    _expect_value_error(
        {"name": "adjust_hyperparameter", "params": {"name": "max_depth"}},
        "missing value",
    )


def test_unknown_hp_name_raises():
    _expect_value_error(
        {"name": "adjust_hyperparameter",
         "params": {"name": "secret_param", "value": 1}},
        "unknown hp",
    )


def test_reweight_weight_above_guardrail_raises():
    _expect_value_error(
        {"name": "reweight_training_samples",
         "params": {"dimension": "phase", "value": "peak", "weight": 99}},
        "weight=99",
    )


def test_reweight_unknown_phase_raises():
    _expect_value_error(
        {"name": "reweight_training_samples",
         "params": {"dimension": "phase", "value": "autumn", "weight": 2}},
        "phase=autumn",
    )


def test_reweight_unknown_dimension_raises():
    _expect_value_error(
        {"name": "reweight_training_samples",
         "params": {"dimension": "galaxy", "value": "x", "weight": 2}},
        "dim=galaxy",
    )


def test_toggle_unknown_group_raises():
    _expect_value_error(
        {"name": "toggle_feature",
         "params": {"feature_group": "fake", "enabled": True}},
        "group=fake",
    )


def test_floor_pct_above_guardrail_raises():
    _expect_value_error(
        {"name": "adjust_floor_constraint", "params": {"floor_pct": 0.99}},
        "floor=0.99",
    )


def test_target_transform_unknown_value_raises():
    _expect_value_error(
        {"name": "change_target_transform", "params": {"transform": "cube"}},
        "cube",
    )



# ---- model-bank families (family-aware catalog + generic hyperparameter action) ----

def _bank_config(family: str = "persistence"):
    cfg = get_default_config()
    cfg["model"]["family"] = family
    return cfg


def test_catalog_for_bank_family_is_param_space_plus_stop():
    a = _adapter()
    catalog = a.get_available_actions(_bank_config("persistence"))
    names = [x["name"] for x in catalog]
    assert names == ["adjust_hyperparameter", "stop"]
    adjust = catalog[0]
    assert adjust["params_schema"]["name"]["enum"] == ["residual_window_weeks"]
    assert adjust["guardrails"]["residual_window_weeks"] == (8, 156)


def test_catalog_without_config_is_legacy_catalog():
    a = _adapter()
    assert len(a.get_available_actions()) == 6
    assert len(a.get_available_actions(get_default_config())) == 6


def test_apply_adjust_hyperparameter_bank_family_writes_model_params():
    a = _adapter()
    cfg = _bank_config("persistence")
    new_cfg, desc = a.apply_action(
        {"name": "adjust_hyperparameter",
         "params": {"name": "residual_window_weeks", "value": 52}},
        cfg,
    )
    assert new_cfg["model"]["params"] == {"residual_window_weeks": 52}
    assert cfg["model"]["params"] == {}  # input unchanged
    assert new_cfg["xgboost"] == cfg["xgboost"]  # legacy section untouched
    assert "model.params.residual_window_weeks" in desc


def test_apply_adjust_hyperparameter_bank_family_enforces_param_space():
    a = _adapter()
    cfg = _bank_config("persistence")
    try:
        a.apply_action({"name": "adjust_hyperparameter",
                        "params": {"name": "residual_window_weeks", "value": 1}}, cfg)
        assert False, "expected guardrail violation"
    except ValueError as e:
        assert "guardrail" in str(e)
    try:
        a.apply_action({"name": "adjust_hyperparameter",
                        "params": {"name": "max_depth", "value": 3}}, cfg)
        assert False, "expected unknown param"
    except ValueError as e:
        assert "not tunable" in str(e)


def test_legacy_only_actions_rejected_for_bank_family():
    a = _adapter()
    cfg = _bank_config("persistence")
    try:
        a.apply_action({"name": "toggle_feature",
                        "params": {"feature_group": "lag", "enabled": False}}, cfg)
        assert False, "expected rejection"
    except ValueError as e:
        assert "only available for the xgboost_direct family" in str(e)


def test_stop_is_a_no_op_for_bank_family():
    a = _adapter()
    cfg = _bank_config("persistence")
    new_cfg, desc = a.apply_action({"name": "stop", "params": {}}, cfg)
    assert new_cfg == cfg and "stop" in desc


def test_domain_context_describes_active_bank_family():
    a = _adapter()
    ctx = a.get_domain_context(_bank_config("persistence"))
    assert "'persistence'" in ctx
    assert "residual_window_weeks" in ctx
    assert "Epidemic phases" in ctx
    assert "XGBoost-based" not in ctx
    legacy_ctx = a.get_domain_context()
    assert "XGBoost-based" in legacy_ctx and "Epidemic phases" in legacy_ctx

ALL = [
    test_catalog_has_six_actions,
    test_apply_adjust_hyperparameter,
    test_apply_reweight_phase,
    test_apply_toggle_feature,
    test_apply_adjust_floor,
    test_apply_change_target_transform,
    test_apply_stop_returns_unchanged_config,
    test_unknown_action_name_raises,
    test_max_depth_above_guardrail_raises,
    test_missing_action_params_raises,
    test_unknown_hp_name_raises,
    test_reweight_weight_above_guardrail_raises,
    test_reweight_unknown_phase_raises,
    test_reweight_unknown_dimension_raises,
    test_toggle_unknown_group_raises,
    test_floor_pct_above_guardrail_raises,
    test_target_transform_unknown_value_raises,
    test_catalog_for_bank_family_is_param_space_plus_stop,
    test_catalog_without_config_is_legacy_catalog,
    test_apply_adjust_hyperparameter_bank_family_writes_model_params,
    test_apply_adjust_hyperparameter_bank_family_enforces_param_space,
    test_legacy_only_actions_rejected_for_bank_family,
    test_stop_is_a_no_op_for_bank_family,
    test_domain_context_describes_active_bank_family,
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
