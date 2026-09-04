"""Log adapters.

Each adapter translates one raw log into the canonical event frame described in
`riskradar.logspec`, so everything downstream is log-agnostic.
"""

from __future__ import annotations

from types import ModuleType

_REGISTRY = {"uci_servicenow", "bpi2013"}


def get_adapter(name: str) -> ModuleType:
    """Return the adapter module for `name`."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown log {name!r}; expected one of {sorted(_REGISTRY)}")
    if name == "uci_servicenow":
        from . import uci_servicenow as mod
    else:
        from . import bpi2013 as mod
    return mod


__all__ = ["get_adapter"]
