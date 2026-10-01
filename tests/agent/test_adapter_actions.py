"""Unit tests for FluForecastAdapter.apply_action + action catalog."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import DEFAULT_MAX_TRAIN_WINDOW_WEEKS, FluForecastAdapter
from agent.knowledge import DEFAULT_CURATED_DIR, DEFAULT_HEADER, KnowledgeBank
from src.config import get_default_config
from src.pipeline import MIN_TRAIN_WINDOW_WEEKS


def _adapter():
    return FluForecastAdapter()


def test_catalog_has_seven_actions():
    a = _adapter()
    names = {x["name"] for x in a.get_available_actions()}
    assert names == {
        "adjust_hyperparameter", "reweight_training_samples",
        "toggle_feature", "adjust_floor_constraint",
        "change_target_transform", "set_training_window", "stop",
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


def test_apply_reweight_approaching_peak():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, desc = a.apply_action(
        {"name": "reweight_training_samples",
         "params": {"dimension": "approaching_peak", "value": 6, "weight": 2.0}},
        cfg,
    )
    assert new_cfg["sample_weights"]["approaching_peak"] == {"weeks_before": 6, "weight": 2.0}
    assert "approaching_peak" not in cfg["sample_weights"]  # input unchanged
    assert new_cfg["sample_weights"]["by_phase"] == {}  # other dimensions untouched
    assert "approaching_peak" in desc


def test_catalog_reweight_lists_approaching_peak_dimension():
    a = _adapter()
    spec = next(x for x in a.get_available_actions() if x["name"] == "reweight_training_samples")
    assert "approaching_peak" in spec["params_schema"]["dimension"]["enum"]
    assert spec["guardrails"]["approaching_peak_weeks"] == (2, 12)
    assert "approaching_peak" in spec["description"]


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


def test_reweight_approaching_peak_weeks_below_guardrail_raises():
    _expect_value_error(
        {"name": "reweight_training_samples",
         "params": {"dimension": "approaching_peak", "value": 1, "weight": 2.0}},
        "weeks_before=1",
    )


def test_reweight_approaching_peak_weeks_above_guardrail_raises():
    _expect_value_error(
        {"name": "reweight_training_samples",
         "params": {"dimension": "approaching_peak", "value": 13, "weight": 2.0}},
        "weeks_before=13",
    )


def test_reweight_approaching_peak_weight_above_guardrail_raises():
    _expect_value_error(
        {"name": "reweight_training_samples",
         "params": {"dimension": "approaching_peak", "value": 6, "weight": 6}},
        "weight=6",
    )


def test_reweight_approaching_peak_non_integer_weeks_raises():
    _expect_value_error(
        {"name": "reweight_training_samples",
         "params": {"dimension": "approaching_peak", "value": "six", "weight": 2.0}},
        "weeks_before=six",
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



# ---- set_training_window (data.train_window_weeks) ----

def test_catalog_set_training_window_guardrail_matches_pipeline():
    a = _adapter()
    entry = next(x for x in a.get_available_actions() if x["name"] == "set_training_window")
    assert entry["guardrails"]["weeks"] == (MIN_TRAIN_WINDOW_WEEKS, DEFAULT_MAX_TRAIN_WINDOW_WEEKS)
    assert entry["guardrails"]["weeks"] == (8, 104)
    assert "ROWS" in entry["description"]
    assert "12 weeks" in entry["description"]


def test_apply_set_training_window_writes_weeks():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, desc = a.apply_action(
        {"name": "set_training_window", "params": {"weeks": 12}}, cfg
    )
    assert new_cfg["data"]["train_window_weeks"] == 12
    assert cfg["data"].get("train_window_weeks") is None  # input unchanged
    assert desc == "data.train_window_weeks = 12"


def test_apply_set_training_window_none_means_all_rows():
    a = _adapter()
    cfg = get_default_config()
    cfg["data"]["train_window_weeks"] = 12
    new_cfg, desc = a.apply_action(
        {"name": "set_training_window", "params": {"weeks": None}}, cfg
    )
    assert new_cfg["data"]["train_window_weeks"] is None
    assert cfg["data"]["train_window_weeks"] == 12  # input unchanged
    assert "None" in desc


def test_set_training_window_below_minimum_raises():
    a = _adapter()
    try:
        a.apply_action({"name": "set_training_window", "params": {"weeks": 7}})
        assert False, "expected rejection"
    except ValueError as e:
        assert "weeks=7" in str(e)
        assert f"minimum {MIN_TRAIN_WINDOW_WEEKS}" in str(e)


def test_set_training_window_above_maximum_raises():
    a = _adapter()
    try:
        a.apply_action({"name": "set_training_window", "params": {"weeks": 105}})
        assert False, "expected rejection"
    except ValueError as e:
        assert "weeks=105" in str(e)
        assert str(DEFAULT_MAX_TRAIN_WINDOW_WEEKS) in str(e)


def test_set_training_window_accepts_integer_string():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, desc = a.apply_action(
        {"name": "set_training_window", "params": {"weeks": "12"}}, cfg
    )
    assert new_cfg["data"]["train_window_weeks"] == 12
    assert type(new_cfg["data"]["train_window_weeks"]) is int
    assert desc == "data.train_window_weeks = 12"


def test_set_training_window_accepts_whole_float():
    a = _adapter()
    cfg = get_default_config()
    new_cfg, desc = a.apply_action(
        {"name": "set_training_window", "params": {"weeks": 12.0}}, cfg
    )
    assert new_cfg["data"]["train_window_weeks"] == 12
    assert type(new_cfg["data"]["train_window_weeks"]) is int
    assert desc == "data.train_window_weeks = 12"


def test_set_training_window_fractional_float_raises():
    a = _adapter()
    try:
        a.apply_action({"name": "set_training_window", "params": {"weeks": "12.5"}})
        assert False, "expected rejection"
    except ValueError as e:
        assert "whole number" in str(e)
        assert "12.5" in str(e)
    try:
        a.apply_action({"name": "set_training_window", "params": {"weeks": 12.5}})
        assert False, "expected rejection"
    except ValueError as e:
        assert "12.5" in str(e)


def test_set_training_window_non_numeric_string_raises():
    a = _adapter()
    try:
        a.apply_action({"name": "set_training_window", "params": {"weeks": "abc"}})
        assert False, "expected rejection"
    except ValueError as e:
        assert "must be an int" in str(e)
        assert "'abc'" in str(e)


def test_set_training_window_bool_raises():
    a = _adapter()
    try:
        a.apply_action({"name": "set_training_window", "params": {"weeks": True}})
        assert False, "expected rejection"
    except ValueError as e:
        assert "must be an int" in str(e)
        assert "True" in str(e)


def test_set_training_window_missing_weeks_raises():
    a = _adapter()
    try:
        a.apply_action({"name": "set_training_window", "params": {}})
        assert False, "expected rejection"
    except ValueError as e:
        assert "weeks" in str(e)


def test_set_training_window_rejected_for_bank_family():
    a = _adapter()
    cfg = _bank_config("persistence")
    try:
        a.apply_action({"name": "set_training_window", "params": {"weeks": 12}}, cfg)
        assert False, "expected rejection"
    except ValueError as e:
        assert "only available for the xgboost_direct family" in str(e)


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
    assert len(a.get_available_actions()) == 7
    assert len(a.get_available_actions(get_default_config())) == 7


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


def test_get_available_actions_does_not_mutate_class_catalog():
    before = copy.deepcopy(FluForecastAdapter.ACTION_CATALOG)

    catalog = _adapter().get_available_actions()

    resolved = next(x for x in catalog if x["name"] == "set_training_window")
    assert resolved["guardrails"]["weeks"] == (MIN_TRAIN_WINDOW_WEEKS, DEFAULT_MAX_TRAIN_WINDOW_WEEKS)
    assert FluForecastAdapter.ACTION_CATALOG == before
    class_entry = next(
        x for x in FluForecastAdapter.ACTION_CATALOG if x["name"] == "set_training_window"
    )
    assert class_entry["guardrails"]["weeks"] is None


def test_importing_adapter_does_not_load_xgboost():
    # The adapter keeps src/* lazy so the CLI and the knowledge tooling stay
    # light (and macOS avoids the libomp clash). A fresh interpreter proves it.
    code = (
        "import sys\n"
        "import agent.adapters.flu_forecast\n"
        "loaded = sorted(m for m in sys.modules if m == 'xgboost' or m.startswith('xgboost.'))\n"
        "print(loaded)\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}

    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )

    assert proc.returncode == 0, f"xgboost loaded on import: {proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip() == "[]"


def test_knowledge_disabled_returns_model_paragraph_only_and_never_opens_bank():
    a = FluForecastAdapter(knowledge_enabled=False)

    legacy_ctx = a.get_domain_context()
    bank_ctx = a.get_domain_context(_bank_config("persistence"))

    assert "XGBoost-based" in legacy_ctx
    assert "KNOWN FACTS" not in legacy_ctx
    assert DEFAULT_HEADER not in legacy_ctx
    assert "'persistence'" in bank_ctx
    assert "KNOWN FACTS" not in bank_ctx
    assert a._knowledge_bank is None  # the lazy property was never triggered


def test_knowledge_enabled_rejects_non_bool():
    try:
        FluForecastAdapter(knowledge_enabled="no")
        assert False, "expected rejection"
    except ValueError as e:
        assert "knowledge_enabled" in str(e)


def test_domain_context_describes_active_bank_family():
    # The bank is built in a temp dir from the real curated files so the test
    # never writes knowledge/knowledge.db.
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank.open(DEFAULT_CURATED_DIR, Path(tmp) / "knowledge.db")
        a = FluForecastAdapter(knowledge_bank=bank)
        migrated = "Onset phase (Oct-Nov): flu activity begins rising."

        ctx = a.get_domain_context(_bank_config("persistence"))
        assert "'persistence'" in ctx
        assert "residual_window_weeks" in ctx
        assert DEFAULT_HEADER in ctx
        assert migrated in ctx
        assert "XGBoost-based" not in ctx
        legacy_ctx = a.get_domain_context()
        assert "XGBoost-based" in legacy_ctx
        assert DEFAULT_HEADER in legacy_ctx and migrated in legacy_ctx

ALL = [
    test_catalog_has_seven_actions,
    test_apply_adjust_hyperparameter,
    test_apply_reweight_phase,
    test_apply_reweight_approaching_peak,
    test_catalog_reweight_lists_approaching_peak_dimension,
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
    test_reweight_approaching_peak_weeks_below_guardrail_raises,
    test_reweight_approaching_peak_weeks_above_guardrail_raises,
    test_reweight_approaching_peak_weight_above_guardrail_raises,
    test_reweight_approaching_peak_non_integer_weeks_raises,
    test_reweight_unknown_dimension_raises,
    test_toggle_unknown_group_raises,
    test_floor_pct_above_guardrail_raises,
    test_target_transform_unknown_value_raises,
    test_catalog_set_training_window_guardrail_matches_pipeline,
    test_apply_set_training_window_writes_weeks,
    test_apply_set_training_window_none_means_all_rows,
    test_set_training_window_below_minimum_raises,
    test_set_training_window_above_maximum_raises,
    test_set_training_window_accepts_integer_string,
    test_set_training_window_accepts_whole_float,
    test_set_training_window_fractional_float_raises,
    test_set_training_window_non_numeric_string_raises,
    test_set_training_window_bool_raises,
    test_set_training_window_missing_weeks_raises,
    test_set_training_window_rejected_for_bank_family,
    test_catalog_for_bank_family_is_param_space_plus_stop,
    test_catalog_without_config_is_legacy_catalog,
    test_apply_adjust_hyperparameter_bank_family_writes_model_params,
    test_apply_adjust_hyperparameter_bank_family_enforces_param_space,
    test_legacy_only_actions_rejected_for_bank_family,
    test_stop_is_a_no_op_for_bank_family,
    test_get_available_actions_does_not_mutate_class_catalog,
    test_importing_adapter_does_not_load_xgboost,
    test_knowledge_disabled_returns_model_paragraph_only_and_never_opens_bank,
    test_knowledge_enabled_rejects_non_bool,
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
