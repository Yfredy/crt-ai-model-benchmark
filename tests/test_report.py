import pandas as pd

from analysis.report import write_report

summary = pd.DataFrame({
    "model": ["TabNet", "Logistic_LASSO_baseline"],
    "mean_auc": [0.85, 0.80], "sd_auc": [0.05, 0.06],
    "min_auc": [0.75, 0.70], "max_auc": [0.92, 0.88],
})
comp = pd.DataFrame({
    "model": ["TabNet"], "mean_auc_diff": [0.05],
    "p_wilcoxon": [0.03], "p_holm": [0.03],
})

def test_write_report(tmp_path):
    out = tmp_path / "report.md"
    write_report(summary, comp, "TabNet", "TreeExplainer", out,
                 extra={"n_repeats": 5, "n_folds": 5, "n_events": 93, "n_total": 163},
                 quality_notes=["Age=165 at row 33 outside plausible range"])
    text = out.read_text(encoding="utf-8")
    assert "TabNet" in text and "0.850" in text
    assert "TreeExplainer" in text
    assert "93" in text and "57.1%" in text
    assert "Age=165" in text
    assert "nested repeated stratified cross-validation" in text

def test_write_report_defaults(tmp_path):
    out = tmp_path / "report2.md"
    write_report(summary, comp, "TabNet", "KernelExplainer", out)
    text = out.read_text(encoding="utf-8")
    assert "KernelExplainer" in text
    assert "Data Quality Notes" not in text and "数据质量备注" not in text
