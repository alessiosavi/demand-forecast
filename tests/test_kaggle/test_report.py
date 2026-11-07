"""Run report rendering tests (pure: text files only)."""

from demand_forecast.kaggle.report import write_report


def test_write_report_labels_mirror_and_pending_overlap(tmp_path):
    path = write_report(tmp_path, {"baseline": 0.5, "zero_shot": 0.4, "finetune": 0.3})

    assert path == tmp_path / "REPORT.md"
    text = path.read_text(encoding="utf-8")
    assert "mirror (non-official)" in text
    assert "corpus_overlap: pending" in text
    assert "finetune" in text
    # The table renders each score with its model, in insertion order.
    assert "| baseline | 0.500000 |" in text
    assert "| zero_shot | 0.400000 |" in text
    assert "| finetune | 0.300000 |" in text
    assert text.index("| baseline |") < text.index("| zero_shot |") < text.index("| finetune |")
    # No submission has been recorded yet.
    assert "submission_ref:" not in text


def test_write_report_includes_submission_ref_when_present(tmp_path):
    text = write_report(tmp_path, {"finetune": 0.3}, submission_ref="4242").read_text(
        encoding="utf-8"
    )
    assert "submission_ref: 4242" in text
    assert "No leaderboard submission recorded yet." not in text


def test_write_report_creates_nested_run_dir(tmp_path):
    run_dir = tmp_path / "runs" / "store-sales"
    path = write_report(run_dir, {"baseline": 1.0})
    assert path == run_dir / "REPORT.md"
    assert path.is_file()
