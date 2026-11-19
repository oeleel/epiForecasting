# Model bank: plugging any forecasting model into the agent framework

The agent loop used to be hardwired to one model (the in-repo XGBoost
ensemble). The model bank (`src/model_bank/`) puts **one small contract** in
front of the loop so any model, including the lab's in-house LSTM/GNN models,
drops in without touching the orchestrator, the evaluator, or the prompts.

The Nixtla wrappers shipped here are *examples* that prove the contract handles
classical, boosted, and neural models. They are not the point. The point is
that the team's own models plug in the same way.

## What a model has to provide

Subclass `src.model_bank.ForecastModel` and implement two methods:

```python
from src.model_bank import ForecastModel, ParamSpec, quantile_column

class FluLSTM(ForecastModel):
    family = "lab_lstm"                       # the name used in configs / CLI
    description = "In-house LSTM, one network per location"
    supports_warm_start = False               # True once get_state/set_state exist

    @classmethod
    def param_space(cls):                     # what Agent 2 is allowed to tune
        return {
            "hidden": ParamSpec("hidden", "int", low=8, high=512),
            "lr": ParamSpec("lr", "float", low=1e-5, high=1e-1),
        }

    @classmethod
    def default_params(cls):
        return {"hidden": 64, "lr": 1e-3}

    def fit(self, history):                   # long frame: unique_id, ds, y (+ extras)
        ...                                   # rows are already <= the cutoff
        self.is_fitted = True

    def predict(self, history):               # -> unique_id, ds, horizon, q05..q95
        ...
```

Data contract, both directions, uses the Nixtla long layout:

| direction | columns | notes |
|---|---|---|
| input to `fit` / `predict` | `unique_id` (str), `ds` (datetime), `y` (float), extras | weekly grid, sorted, no NaN, rows `<=` cutoff |
| output of `predict` | `unique_id`, `ds`, `horizon` (1..H), one column per quantile level | names from `quantile_column(level)`: `q05 q25 q50 q75 q95`; non-negative; non-decreasing |

`self.params` holds the validated params (defaults merged with the config's
overrides). `self.quantile_levels` and `self.horizon` are set by the runner.

The framework validates every prediction frame at the seam
(`validate_forecast_frame`): missing columns, NaN, negative values, bad
horizons, or a missing series raise a `ModelBankError` naming the problem.
Quantile crossing is repaired by sorting and counted, not raised.

## How the model reaches the framework

Two options. Neither requires editing this repo.

1. **Dotted path** (recommended for in-house code kept in its own package):

   ```bash
   python -m agent select-model --families persistence my_lab.models:FluLSTM
   python -m agent improve --cutoff-date 2025-12-06 --model-family my_lab.models:FluLSTM
   ```
   The package just has to be importable (installed, or on `PYTHONPATH`).

2. **Registered name**: add `@register` from `src.model_bank.registry` to the
   class and import the module from `src/model_bank/registry.py`'s
   `_LAZY_MODULES` list. Then `family = "lab_lstm"` works everywhere.

Check what is available in the current environment:

```bash
python -m agent list-models
```

## What you get for free once a model is in

- **Selection stage** (`python -m agent select-model`): rolling-origin warm-up
  of every candidate over the pinned evaluation window, phase-aware WIS /
  coverage / bias per family, and an incumbent chosen by the goal metric
  (`--metric wis --phase peak` etc.).
- **Improvement loop** (`python -m agent improve --model-family X`): Agent 2's
  `adjust_hyperparameter` action is generated from your `param_space()`, so
  the LLM can only propose names and values you declared. Config edits land in
  `config["model"]["params"]`; the pipeline retrains through the same runner.
- **Evaluation and tracking**: the runner writes the repo-standard forecast CSV
  (`predicted_q05..q95`, `forecast`, `horizon`, `model_family`), so
  `summarize`, `history`, `status`, `compare`, and the SQLite run tracker work
  unchanged.
- **Error isolation**: a family that crashes is reported with its error next
  to the others; it never aborts a comparison.

## Data split (pinned, decided 2026-09-02)

`src/config.py` pins the split every comparison uses:

| constant | value |
|---|---|
| `TRAIN_START_DATE` | 2022-02-05 (first FluSight target week) |
| `EVAL_START_DATE` .. `EVAL_END_DATE` | 2025-10-01 .. 2026-05-31 |
| `generate_eval_cutoffs(stride_weeks)` | Saturday cutoffs in the eval window |

`load_history(cutoff)` returns rows in `[TRAIN_START_DATE, cutoff]`; the runner
refuses a history that extends past the cutoff.

## Families shipped in this repo

| family | source | deps | why it is here |
|---|---|---|---|
| `persistence`, `seasonal_naive` | `baselines.py` | none | floor every model must beat; reference implementation of the contract |
| `xgboost_direct` | `legacy.py` | xgboost | the lab's incumbent, wrapped so selection compares it fairly |
| `nn_quantile` | `legacy.py` | torch | proves an in-repo PyTorch multi-horizon model plugs in |
| `sf_autoarima`, `sf_autoets`, `sf_autotheta` | `nixtla.py` | statsforecast | classical examples (`sf_autoarima` is slow: ~1 min per 3 series) |
| `mlf_lightgbm` | `nixtla.py` | mlforecast, lightgbm | global boosted example |
| `mlf_linear`, `mlf_random_forest` | `nixtla.py` | mlforecast, scikit-learn | TSci's linear regression and random forest, as global models on lag features |
| `nf_nhits` | `nixtla.py` | neuralforecast | neural example |
| `nf_lstm`, `nf_transformer` | `nixtla.py` | neuralforecast | TSci's LSTM and transformer, with a multi-quantile loss |

Nixtla extras: `pip install -r documentation/requirements-nixtla.txt`.

### Coverage of the TimeSeriesScientist model list

The comparison study (2026-11) runs the models TSci offers inside our framework. The seven the advisor
named, and the family that stands for each:

| TSci model | Our family | Tunable parameters | Note |
|---|---|---|---|
| ARIMA | `sf_autoarima` | `season_length` | AutoARIMA searches the orders itself, so the agent has almost nothing to tune |
| Exponential smoothing | `sf_autoets` | `season_length` | Same: the library selects the ETS form |
| XGBoost | `xgboost_direct` | the repo's XGBoost params, plus weights, features, window | The lab model: 59 features, one model per horizon and quantile. TSci's XGBoost is 10 lags of one series, recursive |
| LSTM | `nf_lstm` | `input_size`, `max_steps`, `learning_rate`, `encoder_hidden_size`, `encoder_n_layers` | Added 2026-11-19 |
| Transformer | `nf_transformer` | `input_size`, `max_steps`, `learning_rate`, `hidden_size`, `n_head` | Added 2026-11-19. neuralforecast `VanillaTransformer` |
| Linear regression | `mlf_linear` | `alpha`, `max_lag` | Added 2026-11-19. `alpha = 0` is ordinary least squares |
| Random forest | `mlf_random_forest` | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_lag` | Added 2026-11-19 |

Models where agent tuning is close to trivial (the advisor's point about model complexity): ARIMA and
exponential smoothing, because the library already searches their structure, and linear regression,
which has one penalty and one lag length. The loop has real choices to make only for XGBoost, random
forest, LSTM and the transformer.

Selection on the pinned split (Rivanna job 23167385, 2026-11-19: 9 stride-4 cutoffs, 52 locations, default
parameters, no agent). AutoARIMA is absent because its run on 9 cutoffs takes hours.

| Family | WIS | Growth | Peak | Decline | Plateau | 90% coverage | Fit, s |
|---|---|---|---|---|---|---|---|
| `xgboost_direct` (incumbent) | 50.9 | 63.6 | 109.9 | 32.6 | 7.5 | 0.84 | 280 |
| `nf_lstm` | 70.7 | 79.7 | 160.8 | 35.6 | 16.1 | 0.88 | 189 |
| `sf_autoets` | 81.9 | 102.6 | 166.7 | 56.3 | 18.1 | 0.86 | 78 |
| `persistence` | 90.4 | 131.6 | 153.8 | 79.1 | 22.2 | 0.73 | 0.1 |
| `mlf_random_forest` | 115.0 | 92.2 | 306.1 | 55.1 | 27.1 | 0.65 | 103 |
| `mlf_linear` | 143.9 | 104.7 | 360.8 | 83.9 | 55.3 | 0.64 | 1.2 |
| `nf_transformer` | 168.4 | 191.5 | 387.1 | 98.3 | 25.5 | 0.13 | 1321 |

`mlf_random_forest`, `mlf_linear` and `nf_transformer` do not beat `persistence` at their defaults. The
transformer's interval covers 13% of outcomes; 300 training steps on CPU are not enough for it.

First run of the loop on each family (Rivanna, 2026-11-19, cutoff 2025-11-29, `qwen3:8b`, up to 5
iterations, WIS over 52 locations). These are in-sample numbers: the loop keeps the iteration that scores
best on the scored weeks, and the held-out check in
`documentation/experiments/2026-11-tsci-comparison-results.md` shows such gains need not transfer.

| Family | WIS baseline | WIS best | Change | Stop reason |
|---|---|---|---|---|
| `xgboost_direct` | 124.4 | 100.6 | -19.2% | max iterations |
| `nf_lstm` | 159.8 | 153.7 | -3.8% | two regressions |
| `mlf_linear` | 206.7 | 204.3 | -1.1% | two regressions |
| `nf_transformer` | 366.6 | 363.8 | -0.8% | no improvement twice |
| `mlf_random_forest` | 180.8 | 180.8 | none | two regressions |
| `sf_autoarima` | 185.9 | 185.9 | none | two regressions |
| `sf_autoets` | 206.0 | 206.0 | none | agent chose to stop |

The loop has the most to work with on `xgboost_direct`, the only family that honours every action
(reweighting, feature toggles, window, floor). For the bank families it can only change the parameters in
`param_space()`, and for the two auto-selected statistical models that is one number.

TSci's released code offers 21 model names. Of the other 14, `RandomWalk` is our `persistence`, `Theta` is
`sf_autotheta` and `LightGBM` is `mlf_lightgbm`; the rest (moving average, polynomial, ridge, lasso, elastic
net, SVR, gradient boosting, neural network, Prophet, TBATS, Croston) have no family here. Every one of our
families is a global model over all locations with five quantiles; every TSci model is fitted to one series
and returns a point forecast, so a shared name does not mean the same model.

## Where the pieces live

```
src/model_bank/
  contract.py     ForecastModel, ParamSpec, validators, column names
  registry.py     @register, resolve_family(name | "pkg.mod:Class"), list_families
  data_bridge.py  CDC <-> long frame; prediction -> repo forecast CSV
  runner.py       load_history, fit_predict, run_bank_model (pipeline entry)
  baselines.py    persistence, seasonal_naive
  legacy.py       xgboost_direct, nn_quantile
  nixtla.py       statsforecast / mlforecast / neuralforecast examples
agent/model_selection.py   evaluate_candidates, select_incumbent, SelectionGoal
```

`src/pipeline.run_pipeline(config)` dispatches on `config["model"]["family"]`:
`xgboost_direct` keeps the legacy code path (it honors every legacy action:
feature toggles, reweighting, floor, target transform); every other family
goes through `run_bank_model`.

## Warm start (roadmap workstream 4)

Models that can resume from last week's weights set `supports_warm_start =
True` and implement `get_state()` / `set_state(state)`. The runner does not
call these yet; the warm-start research layer will.
