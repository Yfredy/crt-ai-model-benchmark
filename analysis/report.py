import pandas as pd


def write_report(summary_df, comp_df, best_model, shap_method, out_path,
                 extra=None, quality_notes=None, shap_top=None) -> str:
    extra = extra or {}
    quality_notes = quality_notes or []
    n_folds = extra.get("n_folds", 5)
    n_repeats = extra.get("n_repeats", 5)
    inner_folds = extra.get("inner_folds", 3)
    n_events = extra.get("n_events")
    n_total = extra.get("n_total")
    lines = [
        "# CRT 应答预测：传统基线 vs AI 模型对比报告",
        "",
        "## 设计",
        "",
        f"- 交叉验证方案：外层 {n_folds} 折 × {n_repeats} 次重复分层 CV，"
        f"内层 {inner_folds} 折网格调参（嵌套，无数据泄漏）",
        f"- 队列：{n_total} 例 CRT 患者"
        + (f"（应答 {n_events} 例，应答率 {n_events / n_total * 100:.1f}%）" if n_total else ""),
        f"- 随机种子：{extra.get('seed', 2025)}（所有模型与 CV 分割统一）",
        f"- 设备：{extra.get('device', 'cpu')}"
        + ("（TabNet 使用 GPU 加速）" if extra.get("device") == "cuda" else ""),
        "",
        f"## 最优模型：**{best_model}**（SHAP 方法：{shap_method}）",
        "",
        "## 模型汇总（外层折 AUC，mean±sd）",
        "",
        summary_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 与 LASSO 基线的配对比较",
        "",
        "同一折配对 Wilcoxon 符号秩检验 + Holm 多重比较校正（比较单位：每个外层折的 AUC）：",
        "",
        comp_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 图形",
        "",
        "- ![ROC](figures/roc_curves.png)",
        "- ![Calibration](figures/calibration_curves.png)",
        "- ![DCA](figures/dca_curves.png)",
        "- ![SHAP summary](figures/shap_summary.png)",
        "- ![SHAP bar](figures/shap_bar.png)",
        "",
    ]
    if quality_notes:
        lines += ["## 数据质量备注", ""] + [f"- {n}" for n in quality_notes] + [""]
    if shap_top is not None and len(shap_top):
        lines += [
            "## 最优模型 SHAP 特征重要性（前 10）",
            "",
            shap_top.to_frame("mean_abs_shap").to_markdown(floatfmt=".4f"),
            "",
        ]
    lines += [
        "## 方法说明（供论文 Methods 复用）",
        "",
        f"All models were embedded in leak-free preprocessing pipelines and evaluated "
        f"with nested repeated stratified cross-validation (outer {n_folds}-fold × "
        f"{n_repeats} repeats; inner {inner_folds}-fold grid search optimizing AUC). "
        "Discrimination (AUC), calibration (Brier score, calibration slope), "
        "decision-curve net benefit, and Youden sensitivity/specificity were computed "
        "per outer fold on out-of-fold predictions. Pairwise differences versus the "
        "LASSO baseline (univariate selection + L1-penalized logistic regression, "
        "replicating the published nomogram method) were tested with Wilcoxon "
        "signed-rank tests and Holm correction. SHAP values were computed for the "
        "best-performing model on the full cohort for interpretation.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
