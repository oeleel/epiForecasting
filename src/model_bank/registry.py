"""Family registry: turn config["model"]["family"] into a ForecastModel class.

Two ways a family becomes resolvable, in lookup order:

1. **Registered name** - modules inside `src.model_bank` decorate their classes
   with `@register`. Optional-dependency families (Nixtla, torch) register
   lazily: their modules are imported on first lookup and an ImportError is
   reported as "family X needs package Y", never as a crash at import time.
2. **Dotted path** - `"my_lab.models:FluLSTM"` imports the module and takes the
   attribute. This is the plug-and-play path for in-house models: no edit to
   this repo is required, the class just has to subclass ForecastModel.

`list_families()` reports every known family with an `available` flag so the
CLI can show what is runnable in the current environment.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from src.model_bank.contract import ForecastModel, ModelBankError

__all__ = [
    "FamilyInfo",
    "get_family",
    "list_families",
    "register",
    "resolve_family",
]

# Modules that register families when imported. Order = display order.
# Each entry is (module, human-readable requirement) so a missing optional
# dependency yields an actionable message.
_LAZY_MODULES: list[tuple[str, str]] = [
    ("src.model_bank.baselines", "no extra dependencies"),
    ("src.model_bank.legacy", "documentation/requirements.txt (xgboost, torch)"),
    ("src.model_bank.nixtla", "documentation/requirements-nixtla.txt"),
]

_REGISTRY: dict[str, type[ForecastModel]] = {}
_IMPORT_ERRORS: dict[str, str] = {}
_loaded = False


@dataclass(frozen=True, slots=True)
class FamilyInfo:
    family: str
    description: str
    available: bool
    reason: str = ""
    supports_warm_start: bool = False


def register(cls: type[ForecastModel]) -> type[ForecastModel]:
    """Class decorator: add a ForecastModel subclass to the registry by its `family`."""
    if not issubclass(cls, ForecastModel):
        raise ModelBankError(f"@register expects a ForecastModel subclass, got {cls!r}")
    if not cls.family:
        raise ModelBankError(f"{cls.__name__} must set a non-empty `family` before registering")
    existing = _REGISTRY.get(cls.family)
    if existing is not None and existing is not cls:
        raise ModelBankError(
            f"family {cls.family!r} already registered by {existing.__name__}; "
            f"refusing to overwrite with {cls.__name__}"
        )
    _REGISTRY[cls.family] = cls
    return cls


def _load_all() -> None:
    """Import every lazy module once, recording (not raising) import failures."""
    global _loaded
    if _loaded:
        return
    for module_name, requirement in _LAZY_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:  # optional dependency missing
            _IMPORT_ERRORS[module_name] = f"{exc} (install: {requirement})"
    _loaded = True


def resolve_family(name: str) -> type[ForecastModel]:
    """Return the ForecastModel class for a registered name or a dotted path."""
    if not name or not isinstance(name, str):
        raise ModelBankError(f"model family must be a non-empty string, got {name!r}")
    _load_all()
    if name in _REGISTRY:
        return _REGISTRY[name]
    if ":" in name:
        return _import_dotted(name)
    known = sorted(_REGISTRY)
    hint = ""
    if _IMPORT_ERRORS:
        hint = " Some families failed to import: " + "; ".join(
            f"{m}: {e}" for m, e in _IMPORT_ERRORS.items()
        )
    raise ModelBankError(
        f"unknown model family {name!r}. Registered: {known}. "
        f"For an in-house model use a dotted path 'package.module:ClassName'.{hint}"
    )


def get_family(name: str) -> Callable[..., ForecastModel]:
    """Alias of resolve_family for call sites that read better as a factory."""
    return resolve_family(name)


def list_families() -> list[FamilyInfo]:
    """Every family known to this environment, with availability."""
    _load_all()
    infos = [
        FamilyInfo(
            family=cls.family,
            description=cls.description,
            available=True,
            supports_warm_start=cls.supports_warm_start,
        )
        for cls in _REGISTRY.values()
    ]
    for module_name, error in _IMPORT_ERRORS.items():
        infos.append(
            FamilyInfo(
                family=f"({module_name.rsplit('.', 1)[-1]} families)",
                description="not importable",
                available=False,
                reason=error,
            )
        )
    return sorted(infos, key=lambda i: (not i.available, i.family))


def _import_dotted(path: str) -> type[ForecastModel]:
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ModelBankError(f"dotted family path must look like 'pkg.module:Class', got {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelBankError(
            f"cannot import module {module_name!r} for family {path!r}: {exc}"
        ) from exc
    cls: type | None = getattr(module, attr, None)
    if cls is None:
        raise ModelBankError(f"module {module_name!r} has no attribute {attr!r}")
    if not (isinstance(cls, type) and issubclass(cls, ForecastModel)):
        raise ModelBankError(
            f"{path!r} is not a ForecastModel subclass (got {cls!r}); "
            "see src/model_bank/contract.py for the interface"
        )
    return cls
