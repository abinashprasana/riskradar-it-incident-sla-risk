"""Retired.

This app was version 1 of the project. It retrained a model at startup on
whole-trace features -- including resolution time, computed by subtracting the
moment a ticket opened from the moment it closed -- and its evaluation tab
reported metrics on the same rows the model had been fitted on. The 0.9674 it
displayed was not a prediction result.

The corrected study, and a presentation layer built on its actual artifacts,
live elsewhere in this repository:

    docs/MODEL_CARD.md      what the model does, and what it cannot claim
    web/                    KAIROS, the current front end
    python scripts/run_all.py

The previous implementation is preserved at `app_v1_legacy.py.bak` for
reference. It is deliberately not runnable from here: a live deployment that
contradicts its own repository is worse than no deployment.
"""

import sys

MESSAGE = """
RiskRadar v1 has been retired.

  Why:  it trained on features that do not exist at prediction time and
        reported in-sample metrics as if they were held out.

  Now:  docs/MODEL_CARD.md   the corrected study
        web/                 the current front end (KAIROS)
        scripts/run_all.py   reproduce every result

  The honest earliness curve runs 0.619 at one event to 0.887 at eight,
  against the 0.967 this app used to display.
"""

try:
    import streamlit as st

    st.set_page_config(page_title="RiskRadar — retired", page_icon="🗄️")
    st.title("RiskRadar v1 has been retired")
    st.error(
        "This app trained on features unavailable at prediction time and "
        "reported in-sample metrics. Its numbers were not valid."
    )
    st.markdown(
        "The corrected study lives in `docs/MODEL_CARD.md`, the current front "
        "end in `web/`, and every result reproduces with "
        "`python scripts/run_all.py`.\n\n"
        "Honest earliness curve: **0.619** at one event rising to **0.887** at "
        "eight, against the **0.967** this app used to display."
    )
except ImportError:
    print(MESSAGE, file=sys.stderr)
