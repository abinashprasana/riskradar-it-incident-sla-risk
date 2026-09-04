"""Shared fixtures.

The UCI log is 45 MB and takes a few seconds to parse, so it is loaded once per
session.  Tests that assert exact counts are the point of this suite -- they are
what would have caught the back-filled closure columns -- so they run against
the real file rather than a synthetic stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UCI_CSV = ROOT / "incident_event_log.csv"
BPI_CSV = ROOT / "csv_files" / "VINST cases incidents.csv"


@pytest.fixture(scope="session")
def uci_events():
    pytest.importorskip("pandas")
    if not UCI_CSV.exists():
        pytest.skip(f"missing {UCI_CSV}")
    from riskradar.adapters import uci_servicenow as ad

    return ad.load_events(str(UCI_CSV))


@pytest.fixture(scope="session")
def uci_labels(uci_events):
    from riskradar.adapters import uci_servicenow as ad

    return ad.make_labels(uci_events)


@pytest.fixture(scope="session")
def uci_prefix(uci_events, uci_labels):
    from riskradar.adapters import uci_servicenow as ad
    from riskradar.prefix_log import build_prefix_log

    return build_prefix_log(uci_events, uci_labels, ad.SPEC)
