"""Feedback: read-only developer summary of Stage 8 ratings."""

import streamlit as st

from ui import services
from ui.config import FEEDBACK_PAGE_SIZES


def render() -> None:
    st.title("Feedback")
    st.caption("Ratings users gave to answers. Read-only; answers and document "
               "excerpts are not shown here.")

    limit = st.selectbox("Recent ratings to show", FEEDBACK_PAGE_SIZES)
    try:
        summary, rows = services.feedback_overview(limit)
    except services.AssistantError as error:
        st.error(str(error), icon=":material/error:")
        return

    cols = st.columns(4)
    cols[0].metric("Total ratings", summary["total"])
    cols[1].metric("👍 Helpful", summary["thumbs_up"])
    cols[2].metric("👎 Not helpful", summary["thumbs_down"])
    cols[3].metric("Helpful", f"{summary['positive_percentage']}%"
                   if summary["total"] else "n/a")

    if not rows:
        st.info("No feedback yet. Rate an answer on the Ask Documentation page.",
                icon=":material/inbox:")
        return
    st.subheader("Recent feedback")
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Inspect everything from the command line with "
               "`scripts/view_feedback.py`.")
