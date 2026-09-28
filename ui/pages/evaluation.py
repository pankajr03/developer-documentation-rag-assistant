"""Evaluation: read-only view of Stage 10 reports. Never runs an evaluation."""

import streamlit as st

from ui import reports
from ui.formatting import format_rate, format_score

CLI_COMMAND = (
    r".\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --top-k 5")


def _metric_grid(metrics: dict) -> None:
    rows = [
        [("Total cases", str(metrics.get("total_cases", "n/a"))),
         ("Errors", str(metrics.get("failed_cases", "n/a"))),
         ("Source-hit rate", format_rate(metrics.get("source_hit_rate"))),
         ("MRR", format_score(metrics.get("mrr")))],
        [("Hit@1", format_rate(metrics.get("hit_at_1"))),
         ("Hit@3", format_rate(metrics.get("hit_at_3"))),
         ("Hit@5", format_rate(metrics.get("hit_at_5"))),
         ("Keyword coverage", format_rate(metrics.get("average_keyword_coverage")))],
        [("Citation precision", format_rate(metrics.get("citation_precision"))),
         ("Correct abstention", format_rate(metrics.get("correct_abstention_rate"))),
         ("Incorrect abstentions",
          "n/a" if metrics.get("incorrect_abstention_count") is None
          else str(metrics["incorrect_abstention_count"])),
         ("Avg latency", "n/a" if metrics.get("average_total_latency_ms") is None
          else f"{metrics['average_total_latency_ms'] / 1000:.1f} s")],
    ]
    for row in rows:
        for column, (label, value) in zip(st.columns(4), row):
            column.metric(label, value)


def _run_details(summary: dict) -> None:
    config = summary.get("config") or {}
    models = summary.get("models") or {}
    dataset = summary.get("dataset") or {}
    parts = [
        f"Run {summary.get('timestamp', 'unknown time')}",
        f"{dataset.get('selected_case_count', '?')} of "
        f"{dataset.get('case_count', '?')} cases",
        f"top_k {config.get('top_k', '?')}",
        f"generation `{models.get('generation_model') or 'skipped'}`",
    ]
    st.caption(" · ".join(parts))
    if "passed" in summary:
        st.badge("PASS" if summary["passed"] else "FAIL",
                 color="green" if summary["passed"] else "red")


def render() -> None:
    st.title("Evaluation")
    st.caption("Stage 10 evaluation reports. This page only reads reports; "
               "run new evaluations from the command line.")

    refs = reports.list_reports()
    if not refs:
        st.info("No evaluation reports found yet. Run one with:",
                icon=":material/inbox:")
        st.code(CLI_COMMAND, language="powershell")
        return

    labels = {ref.report_id: ref.label for ref in refs}
    selected = st.selectbox("Report", list(labels), format_func=labels.get,
                            help="Newest first.")
    try:
        ref = reports.resolve_report(selected, refs)
        summary = reports.load_summary(ref)
    except reports.ReportError as error:
        st.error(str(error), icon=":material/error:")
        return

    _run_details(summary)
    _metric_grid(summary["metrics"])

    st.subheader("Failed or weak cases")
    try:
        records, bad_lines = reports.load_failures(ref)
    except reports.ReportError as error:
        st.error(str(error))
    else:
        if records:
            st.dataframe(reports.failure_rows(records), hide_index=True,
                         width="stretch")
        else:
            st.success("No failed or weak cases in this report.")
        if bad_lines:
            st.warning(f"{bad_lines} line(s) of failures.jsonl could not be read.")

    categories = summary.get("by_category")
    if isinstance(categories, dict) and categories:
        with st.expander("Metrics by category"):
            st.dataframe(
                [{"Category": name, "Cases": m.get("total_cases"),
                  "Source hit": format_rate(m.get("source_hit_rate")),
                  "MRR": format_score(m.get("mrr")),
                  "Keyword coverage": format_rate(m.get("average_keyword_coverage"))}
                 for name, m in categories.items() if isinstance(m, dict)],
                hide_index=True, width="stretch")

    st.subheader("Download")
    columns = st.columns(len(reports.DOWNLOADABLE_FILES))
    for column, filename in zip(columns, reports.DOWNLOADABLE_FILES):
        data = reports.read_download(ref, filename)
        column.download_button(filename, data or b"", file_name=filename,
                               disabled=data is None, icon=":material/download:",
                               key=f"download_{selected}_{filename}")

    st.caption("Run a new evaluation (retrieval + answers):")
    st.code(CLI_COMMAND, language="powershell")
    st.caption("Add `--skip-generation` for a cheaper retrieval-only run.")
