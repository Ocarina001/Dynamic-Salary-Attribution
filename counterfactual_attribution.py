# -*- coding: utf-8 -*-
"""
counterfactual_attribution.py —— 反事实动态归因模块
====================================================

在整条数据管线中的位置：

  path_search.py --export（path_features_specific.csv）
  build_job_salary.py（job_salary.csv）
  train_xgb_shap.py（models/xgb_salary.json）
        └─> [本脚本] counterfactual_attribution.py
              ├─> reports/counterfactual/pair_shap_waterfall.png         # 单配对 SHAP 瀑布图
              ├─> reports/counterfactual/pair_shap_all.csv                # 1276 配对 SHAP 分解
              ├─> reports/counterfactual/same_job_counterfactual.csv       # 同岗位跨专业反事实
              ├─> reports/counterfactual/same_major_counterfactual.csv     # 同专业跨岗位反事实
              ├─> reports/counterfactual/intervention_scan_{feature}.png   # 特征干预扫描图
              ├─> reports/counterfactual/intervention_matrix.png            # 多特征干预矩阵
              └─> reports/counterfactual/summary_report.md                 # 自动生成的归因报告

核心思想：
  XGBoost + SHAP 可以把每一次预测拆解到"每个特征贡献了多少"，
  但它本身是关联模型，不能直接做因果陈述。
  反事实归因的策略是：在「苹果对苹果」的组内对比，
  避免跨岗位/跨专业的选择偏差（Selection Bias）。

四种反事实模式（可信度从高到低）：

  ① pair_explain（配对内解释）— 最可信
     对每个（source_major → target_job）配对，
     用配对级模型算 SHAP：预测薪资 $Y，其中 skill_gap 贡献 +$A，
     semantic_score 贡献 -$B，……，每个特征各占多少。

  ② same_job_counterfactual（同岗位跨专业对比）— 可信
     固定 target_job（比如 Data Scientist），
     比较不同 source_major 的特征值和预测薪资：
     "英语专业 skill_gap=0.63 时预测 $95k，
      计算机专业 skill_gap=0.35 时预测 $142k；
      差距 $47k 中，skill_gap 贡献了 $31k（按 SHAP 分解）"

  ③ same_major_counterfactual（同专业跨岗位对比）— 可信
     固定 source_major（比如英语专业），
     比较它在不同 target_job 上的薪资和特征：
     "英语专业做 Software Developer $144k，
      做 HR Specialist $73k；
      差距 $71k 中，哪些特征贡献最大？"

  ④ intervention_scan（单特征干预扫描）— 次可信，标注"模型假设"
     对某个配对，手动把 skill_gap 从 0.6 降到 0.3，
     看模型预测涨多少。
     ⚠️ 这是模型内的干预，忽略选择偏差——
     因为"skill_gap 降了"这件事在现实中是"换了一个更容易覆盖的岗位"，
     不是"在同一个岗位上补了技能"。

评估协议：
  用 train_xgb_shap.py 的配对级模型 xgb_salary.json
  （不是岗位级 xgb_salary_job.json，岗位级模型受选择偏差影响严重）
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

FEATURES = [
    "path_length", "skill_gap", "semantic_score", "baseline_gap",
    "baseline_best_gap", "gap_reduction", "gap_reduction_vs_best", "n_paths",
]
PAIR_KEY = ["source_major", "target_job"]
LABEL_COL = "median_wage"

SALARY_PATH = ROOT / "job_salary.csv"
MODEL_PATH = ROOT / "models" / "xgb_salary.json"
OUT_DIR = ROOT / "reports" / "counterfactual"


def load_data() -> tuple[pd.DataFrame, object]:
    import xgboost as xgb

    pf = pd.read_csv(ROOT / "path_features_specific.csv", encoding="utf-8")
    sal = pd.read_csv(SALARY_PATH, encoding="utf-8")
    df = pf.merge(
        sal[["onet_job", LABEL_COL]].drop_duplicates("onet_job"),
        left_on="target_job", right_on="onet_job", how="left",
    )
    missing = df[LABEL_COL].isna().sum()
    if missing:
        bad = sorted(df.loc[df[LABEL_COL].isna(), "target_job"].unique())
        print(f"[警告] {missing} 行缺少工资标签，涉及岗位: {bad[:5]}")
        df = df.dropna(subset=[LABEL_COL])

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    print(f"[数据] {len(df)} 配对 | 岗位 {df.target_job.nunique()} | 专业 {df.source_major.nunique()}")
    print(f"[模型] 已加载 {MODEL_PATH.name}（配对级，{len(FEATURES)} 特征）")
    return df, model


def predict_and_explain(model, X: pd.DataFrame):
    import shap

    pred = model.predict(X[FEATURES])
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X[FEATURES])
    expected = float(explainer.expected_value)
    return pred, shap_vals, expected


# ──────────────────────────────────────────────
# ① 配对级 SHAP 分解（最可信）
# ──────────────────────────────────────────────
def run_pair_explain(df: pd.DataFrame, model) -> pd.DataFrame:
    print("\n========== ① 配对级 SHAP 分解 ==========")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred, shap_vals, expected = predict_and_explain(model, df)
    df = df.copy()
    df["pred_wage"] = pred
    df["pred_vs_actual"] = df["pred_wage"] - df[LABEL_COL]

    shap_df = df[PAIR_KEY + [LABEL_COL, "pred_wage", "pred_vs_actual"]].copy()
    for i, f in enumerate(FEATURES):
        shap_df[f"shap_{f}"] = shap_vals[:, i]
        shap_df[f"feat_{f}"] = df[f].values
    shap_df["expected_value"] = expected

    out_csv = OUT_DIR / "pair_shap_all.csv"
    shap_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] {out_csv}（{len(shap_df)} 配对 × {2 + 8 + 8 + 1} 列）")

    top_residual = shap_df.reindex(shap_df["pred_vs_actual"].abs().sort_values(ascending=False).index).head(5)
    print("\n[预测偏差最大的 5 个配对]")
    for _, r in top_residual.iterrows():
        print(f"  {r.source_major[:18]:18s} → {r.target_job[:22]:22s} "
              f"真实=${r[LABEL_COL]:.0f}  预测=${r.pred_wage:.0f}  "
              f"偏差=${r.pred_vs_actual:+.0f}")

    _plot_top_shap_waterfall(shap_df, expected)
    return shap_df, expected


def _draw_waterfall_barh(ax, base_val: float, feat_names: list[str],
                         shap_vals: list[float], total_pred: float,
                         title: str = ""):
    cum = base_val
    y_positions = list(range(len(feat_names) + 2))
    names = ["基线"] + feat_names + ["预测值"]
    starts = [0.0]
    bar_widths = [base_val]
    colors = ["gray"]
    texts = [f"${base_val:.0f}"]

    for fname, sv in zip(feat_names, shap_vals):
        starts.append(cum)
        bar_widths.append(sv)
        colors.append("#2ecc71" if sv >= 0 else "#e74c3c")
        texts.append(f"${sv:+.0f}")
        cum += sv

    starts.append(0.0)
    bar_widths.append(total_pred)
    colors.append("steelblue")
    texts.append(f"${total_pred:.0f}")

    ax.barh(y_positions, bar_widths, left=starts, height=0.55,
            color=colors, edgecolor="white")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("薪资（美元/年）", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axvline(base_val, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)

    for i, (s, w, t) in enumerate(zip(starts, bar_widths, texts)):
        x = s + w if w >= 0 else s
        ax.text(x + (300 if w >= 0 else -300), i, t,
                va="center", ha="left" if w >= 0 else "right", fontsize=8)


def _plot_top_shap_waterfall(shap_df: pd.DataFrame, expected: float, top_n: int = 6):
    biggest = shap_df.reindex(
        shap_df["pred_vs_actual"].abs().sort_values(ascending=False).index
    ).head(top_n)

    fig, axes = plt.subplots(top_n, 1, figsize=(12, 4.2 * top_n))
    if top_n == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(biggest.iterrows()):
        feat_names_with_val = [
            f"{f}={row[f'feat_{f}']:.3f}" for f in FEATURES
        ]
        shap_vals = [row[f"shap_{f}"] for f in FEATURES]

        title = (
            f"{row.source_major[:16]} → {row.target_job[:22]}\n"
            f"真实=${row[LABEL_COL]:.0f}  预测=${row.pred_wage:.0f}  "
            f"偏差=${row.pred_vs_actual:+.0f}"
        )
        _draw_waterfall_barh(
            axes[idx], expected, feat_names_with_val, shap_vals,
            row["pred_wage"], title=title,
        )

    fig.suptitle("配对级 SHAP 瀑布图（预测偏差最大的 6 个配对）\n"
                 "绿色=该特征拉高预测  红色=该特征拉低预测",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_png = OUT_DIR / "pair_shap_waterfall.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] 已保存 {out_png}")


# ──────────────────────────────────────────────
# ② 同岗位跨专业反事实
# ──────────────────────────────────────────────
def run_same_job_counterfactual(shap_df: pd.DataFrame, expected: float) -> pd.DataFrame:
    print("\n========== ② 同岗位跨专业反事实 ==========")

    rows = []
    for job, grp in shap_df.groupby("target_job"):
        if len(grp) < 2:
            continue

        best_idx = grp["pred_wage"].idxmax()
        worst_idx = grp["pred_wage"].idxmin()
        best = grp.loc[best_idx]
        worst = grp.loc[worst_idx]

        total_gap = best["pred_wage"] - worst["pred_wage"]
        if total_gap <= 0:
            continue

        shap_delta = {}
        for f in FEATURES:
            shap_delta[f] = best[f"shap_{f}"] - worst[f"shap_{f}"]

        rows.append({
            "target_job": job,
            "actual_wage": best[LABEL_COL],
            "best_major": best["source_major"],
            "best_pred": best["pred_wage"],
            "worst_major": worst["source_major"],
            "worst_pred": worst["pred_wage"],
            "pred_gap": round(total_gap, 1),
            "pred_gap_pct": round(total_gap / worst["pred_wage"] * 100, 1),
            **{f"delta_{f}": round(shap_delta[f], 1) for f in FEATURES},
        })

    result = pd.DataFrame(rows).sort_values("pred_gap", ascending=False)
    out_csv = OUT_DIR / "same_job_counterfactual.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] {out_csv}（{len(result)} 岗位）")

    print("\n[同岗位预测薪资差距 Top 10]")
    show_cols = ["target_job", "best_major", "worst_major", "pred_gap", "pred_gap_pct"]
    for _, r in result.head(10).iterrows():
        print(f"  {r.target_job[:22]:22s}  "
              f"best={r.best_major[:12]:12s}(${r.best_pred:.0f})  "
              f"worst={r.worst_major[:12]:12s}(${r.worst_pred:.0f})  "
              f"gap=${r.pred_gap:.0f}({r.pred_gap_pct:.1f}%)")

    result["top_contributor"] = result.apply(
        lambda r: max(FEATURES, key=lambda f: abs(r[f"delta_{f}"])), axis=1
    )
    result["top_contribution"] = result.apply(
        lambda r: r[f"delta_{r.top_contributor}"], axis=1
    )
    print("\n[最大贡献特征（SHAP 分解）]")
    contrib_counts = result["top_contributor"].value_counts()
    for feat, cnt in contrib_counts.items():
        print(f"  {feat:24s}: {cnt} 个岗位的主导特征")

    _plot_job_counterfactual_heatmap(result)
    return result


def _plot_job_counterfactual_heatmap(result: pd.DataFrame, top_n: int = 15):
    top = result.head(top_n).copy()
    feat_contrib = top[[f"delta_{f}" for f in FEATURES]].values
    feat_contrib = feat_contrib / (np.abs(feat_contrib).sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.55)))
    im = ax.imshow(feat_contrib, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{r.target_job[:18]}\n(${r.pred_gap:.0f})" for _, r in top.iterrows()],
        fontsize=8,
    )
    ax.set_title(
        f"同岗位跨专业反事实：特征对预测薪资差距的贡献（Top {top_n} 岗位）\n"
        "红色=best 专业该特征更优  蓝色=worst 专业该特征更优",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="SHAP 贡献占比")
    plt.tight_layout()
    out_png = OUT_DIR / "same_job_contribution_heatmap.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[热力图] 已保存 {out_png}")


# ──────────────────────────────────────────────
# ③ 同专业跨岗位反事实
# ──────────────────────────────────────────────
def run_same_major_counterfactual(shap_df: pd.DataFrame, expected: float) -> pd.DataFrame:
    print("\n========== ③ 同专业跨岗位反事实 ==========")

    rows = []
    for major, grp in shap_df.groupby("source_major"):
        if len(grp) < 2:
            continue

        high_idx = grp["pred_wage"].idxmax()
        low_idx = grp["pred_wage"].idxmin()
        h = grp.loc[high_idx]
        l = grp.loc[low_idx]

        total_gap = h["pred_wage"] - l["pred_wage"]
        if total_gap <= 0:
            continue

        shap_delta = {}
        for f in FEATURES:
            shap_delta[f] = h[f"shap_{f}"] - l[f"shap_{f}"]

        rows.append({
            "source_major": major,
            "high_job": h["target_job"],
            "high_actual": h[LABEL_COL],
            "high_pred": h["pred_wage"],
            "low_job": l["target_job"],
            "low_actual": l[LABEL_COL],
            "low_pred": l["pred_wage"],
            "pred_gap": round(total_gap, 1),
            "pred_gap_pct": round(total_gap / l["pred_wage"] * 100, 1),
            **{f"delta_{f}": round(shap_delta[f], 1) for f in FEATURES},
        })

    result = pd.DataFrame(rows).sort_values("pred_gap", ascending=False)
    out_csv = OUT_DIR / "same_major_counterfactual.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] {out_csv}（{len(result)} 专业）")

    print("\n[同专业预测薪资差距 Top 10]")
    for _, r in result.head(10).iterrows():
        print(f"  {r.source_major[:18]:18s}  "
              f"→ {r.high_job[:16]:16s}(${r.high_pred:.0f})  "
              f"vs {r.low_job[:16]:16s}(${r.low_pred:.0f})  "
              f"gap=${r.pred_gap:.0f}({r.pred_gap_pct:.1f}%)")

    return result


# ──────────────────────────────────────────────
# ④ 单特征干预扫描 + 多特征干预矩阵
# ──────────────────────────────────────────────
def run_intervention_scan(df: pd.DataFrame, model):
    print("\n========== ④ 特征干预扫描（模型内假设）==========")
    print("[警告] 这是模型内干预，忽略选择偏差——真实世界中改变 skill_gap 可能伴随岗位变化")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = df.copy()
    pred_orig = model.predict(ref[FEATURES])
    ref["pred_orig"] = pred_orig

    scan_features = ["skill_gap", "semantic_score", "path_length", "gap_reduction"]
    ranges = {
        "skill_gap": np.linspace(0.1, 0.9, 41),
        "semantic_score": np.linspace(0.2, 0.8, 31),
        "path_length": np.linspace(1, 5, 41),
        "gap_reduction": np.linspace(0.0, 0.6, 31),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    intervention_rows = []

    for ax, feat in zip(axes, scan_features):
        feat_vals = ranges.get(feat, np.linspace(df[feat].min(), df[feat].max(), 31))
        agg = []

        for v in feat_vals:
            test = ref.copy()
            test[feat] = v
            new_pred = model.predict(test[FEATURES])
            agg.append({
                "feat_val": v,
                "mean_pred": new_pred.mean(),
                "p25_pred": np.percentile(new_pred, 25),
                "p75_pred": np.percentile(new_pred, 75),
                "delta_mean": (new_pred - ref["pred_orig"]).mean(),
            })
            intervention_rows.append({"feature": feat, "feat_val": v, "mean_pred": new_pred.mean(),
                                       "delta_mean": (new_pred - ref["pred_orig"]).mean()})

        agg_df = pd.DataFrame(agg)
        ax.plot(agg_df["feat_val"], agg_df["mean_pred"], "b-", linewidth=2, label="均值预测")
        ax.fill_between(agg_df["feat_val"], agg_df["p25_pred"], agg_df["p75_pred"],
                        alpha=0.2, color="b", label="25-75% 区间")

        orig = df[feat].mean()
        orig_pred = ref["pred_orig"].mean()
        ax.axvline(orig, color="r", linestyle="--", alpha=0.7, label=f"当前均值={orig:.3f}")
        ax.axhline(orig_pred, color="r", linestyle=":", alpha=0.5)

        delta = agg_df.iloc[-1]["mean_pred"] - agg_df.iloc[0]["mean_pred"]
        direction = "↑正" if delta > 0 else "↓负"
        ax.set_title(f"{feat} 干预扫描  (Δ={delta:+.0f}美元, {direction})", fontsize=11)
        ax.set_xlabel(f"{feat} 值")
        ax.set_ylabel("模型预测薪资（均值）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("单特征干预扫描：特征值变化 → 预测薪资变化\n"
                "（⚠️ 模型内假设，忽略选择偏差）", fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "intervention_scan.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[干预] 已保存 {out_png}")

    interv_df = pd.DataFrame(intervention_rows)
    interv_df.to_csv(OUT_DIR / "intervention_scan_data.csv", index=False, encoding="utf-8-sig")

    _plot_intervention_matrix(df, model)
    return interv_df


def _plot_intervention_matrix(df: pd.DataFrame, model):
    import shap

    ref = df.copy()
    ref["pred_orig"] = model.predict(ref[FEATURES])
    explainer = shap.TreeExplainer(model)
    shap_orig = explainer.shap_values(ref[FEATURES])

    target_features = ["skill_gap", "semantic_score", "gap_reduction", "n_paths"]
    interventions = {
        "skill_gap": [("改善 -0.1", -0.1), ("改善 -0.2", -0.2), ("改善 -0.3", -0.3)],
        "semantic_score": [("提高 +0.1", 0.1), ("提高 +0.2", 0.2)],
        "gap_reduction": [("提高 +0.1", 0.1), ("提高 +0.2", 0.2)],
        "n_paths": [("翻倍 ×2", None)],
    }

    labels = []
    effects = []
    for feat in target_features:
        for label, delta in interventions[feat]:
            test = ref.copy()
            if feat == "n_paths":
                test[feat] = test[feat] * 2
            else:
                test[feat] = test[feat] + delta
                test[feat] = test[feat].clip(lower=0, upper=1)

            new_pred = model.predict(test[FEATURES])
            delta_wage = (new_pred - ref["pred_orig"]).mean()

            new_shap = explainer.shap_values(test[FEATURES])
            shap_delta = (new_shap - shap_orig).mean(axis=0)

            labels.append(f"{feat}\n{label}")
            effects.append({
                "delta_wage": delta_wage,
                **dict(zip(FEATURES, shap_delta)),
            })

    eff_df = pd.DataFrame(effects, index=labels)
    shap_cols = [c for c in eff_df.columns if c != "delta_wage"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(labels) * 0.55)))

    bars = ax1.barh(range(len(labels)), eff_df["delta_wage"].values, color="steelblue")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("预测薪资变化（美元/年，均值）")
    ax1.set_title("干预 → 预测薪资变化")
    ax1.axvline(0, color="k", linewidth=0.5)
    for bar, val in zip(bars, eff_df["delta_wage"].values):
        ax1.text(val + (100 if val >= 0 else -100), bar.get_y() + bar.get_height() / 2,
                 f"{val:+.0f}", va="center", fontsize=8)

    shap_vals = eff_df[shap_cols].values
    im = ax2.imshow(shap_vals, cmap="RdBu_r", aspect="auto")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xticks(range(len(shap_cols)))
    ax2.set_xticklabels(shap_cols, rotation=35, ha="right", fontsize=8)
    ax2.set_title("各特征 SHAP 值变化（红色=正贡献，蓝色=负贡献）")
    plt.colorbar(im, ax=ax2, label="Δ SHAP")

    fig.suptitle("多特征干预矩阵\n（⚠️ 模型内假设）", fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "intervention_matrix.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[干预矩阵] 已保存 {out_png}")


# ──────────────────────────────────────────────
# ⑤ 自动生成归因报告
# ──────────────────────────────────────────────
def generate_report(shap_df, job_cf, major_cf, interv_df):
    print("\n========== 生成归因报告 ==========")

    lines = []
    lines.append("# 反事实动态归因报告\n")
    lines.append(f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"数据：{len(shap_df)} 配对 / {shap_df.target_job.nunique()} 岗位 / "
                 f"{shap_df.source_major.nunique()} 专业\n")
    lines.append("---\n")

    lines.append("## 模型预测 vs 真实工资（配对级）\n")
    mae = (shap_df["pred_wage"] - shap_df["median_wage"]).abs().mean()
    r2 = 1 - ((shap_df["pred_wage"] - shap_df["median_wage"]) ** 2).sum() / \
           ((shap_df["median_wage"] - shap_df["median_wage"].mean()) ** 2).sum()
    lines.append(f"- MAE = **${mae:,.0f}**")
    lines.append(f"- R² = **{r2:.4f}**")
    lines.append(f"- 真实工资范围：${shap_df.median_wage.min():,.0f} ~ ${shap_df.median_wage.max():,.0f}")
    lines.append(f"- 预测工资范围：${shap_df.pred_wage.min():,.0f} ~ ${shap_df.pred_wage.max():,.0f}\n")

    lines.append("---\n")
    lines.append("## 同岗位跨专业反事实（固定岗位，看不同专业的预测薪资差距）\n")
    lines.append(f"共 {len(job_cf)} 个岗位有跨专业差异。以下是预测薪资差距最大的 10 个岗位：\n")
    top = job_cf.head(10)
    lines.append("| 岗位 | 最优专业（预测） | 最差专业（预测） | 预测差距 | 主导特征 |")
    lines.append("|---|---|---|---|---|")
    for _, r in top.iterrows():
        top_f = max(FEATURES, key=lambda f: abs(r[f"delta_{f}"]))
        lines.append(f"| {r.target_job} | {r.best_major} (${r.best_pred:,.0f}) | "
                     f"{r.worst_major} (${r.worst_pred:,.0f}) | ${r.pred_gap:,.0f} "
                     f"({r.pred_gap_pct:.1f}%) | {top_f} |")
    lines.append("")

    lines.append("---\n")
    lines.append("## 同专业跨岗位反事实（固定专业，看不同岗位的预测薪资差距）\n")
    lines.append(f"共 {len(major_cf)} 个专业跨了多个岗位。以下是预测薪资差距最大的 10 个专业：\n")
    top_m = major_cf.head(10)
    lines.append("| 专业 | 最高薪岗位（预测） | 最低薪岗位（预测） | 预测差距 |")
    lines.append("|---|---|---|---|")
    for _, r in top_m.iterrows():
        lines.append(f"| {r.source_major} | {r.high_job} (${r.high_pred:,.0f}) | "
                     f"{r.low_job} (${r.low_pred:,.0f}) | ${r.pred_gap:,.0f} ({r.pred_gap_pct:.1f}%) |")
    lines.append("")

    lines.append("---\n")
    lines.append("## 单特征干预扫描（模型内假设）\n")
    lines.append("⚠️ 以下结论是**模型内的关联外推**，忽略选择偏差，仅供方向参考：\n")

    for feat in ["skill_gap", "semantic_score", "path_length", "gap_reduction"]:
        sub = interv_df[interv_df.feature == feat]
        delta = sub.iloc[-1]["mean_pred"] - sub.iloc[0]["mean_pred"]
        direction = "↑涨" if delta > 0 else "↓跌"
        lines.append(f"- 干预 `{feat}` 从 {sub.iloc[0].feat_val:.2f} → {sub.iloc[-1].feat_val:.2f}："
                     f"预测薪资 {direction} ${abs(delta):,.0f}")
    lines.append("")

    lines.append("---\n")
    lines.append("## 重要提醒（归因≠因果）\n")
    lines.append("""
1. **选择偏差（Selection Bias）**：高薪岗位（如 Financial Manager）要求的技能天然更复杂、更难覆盖，
   所以这些岗位的 skill_gap 反而**更大**。这不是"skill_gap 大→高薪"，而是"高薪岗位→skill_gap 难降"。

2. **反事实干预是模型内假设**：如果把某配对的 skill_gap 从 0.6 降到 0.3，
   现实中大概率意味着你**换了一个更容易覆盖的岗位**（比如从 Data Scientist 换成 Data Analyst），
   不是"在同一个岗位上补了技能就涨薪"。

3. **跨切片对比最可信**：同岗位跨专业、同专业跨岗位的对比，
   才是"苹果对苹果"的归因——它们控制了岗位或专业一个维度的影响。

4. **29 个唯一工资值的局限**：BLS 只有 29 个岗位的中位数工资，
   模型训练信号有限，归因结论是**方向级**的，不是精确数值级的。
""")

    report_path = OUT_DIR / "summary_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[报告] 已保存 {report_path}")
    return report_path


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="反事实动态归因（配对级 SHAP + 同岗位/同专业对比 + 特征干预）")
    ap.add_argument("--mode", choices=["all", "explain", "job_cf", "major_cf", "intervene"], default="all",
                    help="只跑某一种模式（默认 all）")
    args = ap.parse_args()

    df, model = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "explain", "job_cf", "major_cf"):
        shap_df, expected = run_pair_explain(df, model)

    if args.mode in ("all", "job_cf"):
        job_cf = run_same_job_counterfactual(shap_df, expected)
    elif args.mode in ("all", "explain"):
        shap_df, expected = run_pair_explain(df, model)
        job_cf = pd.DataFrame()

    if args.mode in ("all", "major_cf"):
        major_cf = run_same_major_counterfactual(shap_df, expected)
    elif args.mode == "all":
        major_cf = pd.DataFrame()

    if args.mode in ("all", "intervene"):
        interv_df = run_intervention_scan(df, model)
    else:
        interv_df = pd.DataFrame()

    if args.mode == "all":
        generate_report(shap_df, job_cf, major_cf, interv_df)

    print(f"\n[完成] 所有产出在 {OUT_DIR}/")


if __name__ == "__main__":
    main()