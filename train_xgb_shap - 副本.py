# -*- coding: utf-8 -*-
"""
train_xgb_shap.py —— 用路径特征训练 XGBoost + SHAP 解释（模型解释阶段）
=================================================================
在整条数据管线中的位置：

  path_search.py --export（产出 path_features_specific.csv）
        └─> [本脚本] train_xgb_shap.py
              ├─> reports/xgb_metrics.csv              交叉验证与留出集指标
              ├─> reports/xgb_feature_importance.csv   增益重要性（gain）
              ├─> reports/shap_summary_*.png           SHAP 汇总图（点图 + 条形图）
              └─> models/xgb_*.json                    训练好的模型（XGBoost 原生格式）

      前置依赖：job_salary.csv（build_job_salary.py 生成，提供 BLS 工资标签）

为什么要做这一步（与"薪资查表"的区别）：
  build_job_salary.py 只是把 BLS 工资"贴"到岗位上（查表，不是模型，且不参与排名）。
  本脚本要回答的是：**哪些路径特征与高薪岗位相关、各自贡献多大**——
  用 XGBoost 学一个「路径特征 -> 外部标签」的映射，再用 SHAP 把预测逐条拆解。
  它是"验证 + 归因"工具，不替代薪资表，也不替代 path_search 的排序。

标签选择（防循环论证，务必看）：
  不用 cost / high_fit 当标签——它们是本脚本输入特征（skill_gap 等）的确定性函数，
  拿它们训练等于让模型把公式重新拟合一遍，SHAP 图会循环论证。
  改用**外部信号**：岗位的 BLS 中位工资（median_wage），来自 labor market，与特征独立。
    - task=salary       回归：预测岗位中位工资（美元/年）
    - task=high_salary  分类：是否高薪岗位（工资 >= 全体 29 岗位中位数）

评估协议（防泄漏）：
  1,276 行来自 44 专业 × 29 岗位的交叉结构，同一岗位/专业的样本高度相关，
  随机切分会把"同一岗位"泄漏到训练集和测试集。因此全部评估按 target_job
  分组（GroupKFold / GroupShuffleSplit），训练与测试永不共享岗位。

特征（8 个，全部来自 path_features_specific.csv）：
  path_length / skill_gap / semantic_score / baseline_gap / baseline_best_gap /
  gap_reduction / gap_reduction_vs_best / n_paths
  刻意**排除 cost**：它是上述特征的加权和，进模型只会重复计权、淹没真实贡献。

输出解读提醒：
  本数据只有 29 个唯一工资值、样本相关性高，结论是**相关性/归因**，不是因果；
  指标仅供参考，重点看 SHAP 的方向与排序是否与领域直觉一致。

用法：
  python train_xgb_shap.py                    # 三个任务都跑（默认）
  python train_xgb_shap.py --task salary      # 只跑配对级薪资回归
  python train_xgb_shap.py --task high_salary # 只跑配对级高薪分类
  python train_xgb_shap.py --task salary_job  # 岗位级聚合（29 行，留一法 LOOCV）

  task=salary_job 说明：同一岗位的工资是常量，配对级预测工资等价于"猜岗位"，
  因此额外提供岗位级聚合模式：每岗位取 8 个特征的跨专业均值，工资取唯一值，
  用留一法（LOOCV, n=29）评估，SHAP 在全部 29 个岗位上解释。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
P = {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in CFG["paths"].items()}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURES = [
    "path_length",
    "skill_gap",
    "semantic_score",
    "baseline_gap",
    "baseline_best_gap",
    "gap_reduction",
    "gap_reduction_vs_best",
    "n_paths",
]
GROUP_COL = "target_job"
LABEL_COL = "median_wage"

SALARY_PATH = ROOT / "job_salary.csv"
OUT_METRICS = ROOT / "reports" / "xgb_metrics.csv"
OUT_IMPORTANCE = ROOT / "reports" / "xgb_feature_importance.csv"
MODEL_DIR = ROOT / "models"


def load_data() -> pd.DataFrame:
    """读路径特征表并合并 BLS 工资标签（按岗位）。"""
    pf_path = P["path_features"]
    if not pf_path.exists():
        raise SystemExit(f"[错误] 找不到路径特征表: {pf_path}，请先运行 path_search.py --export")
    if not SALARY_PATH.exists():
        raise SystemExit(f"[错误] 找不到薪资表: {SALARY_PATH}，请先运行 build_job_salary.py")

    pf = pd.read_csv(pf_path, encoding="utf-8")
    sal = pd.read_csv(SALARY_PATH, encoding="utf-8")

    missing = [c for c in FEATURES + [GROUP_COL] if c not in pf.columns]
    if missing:
        raise SystemExit(f"[错误] 路径特征表缺少列: {missing}")
    if "onet_job" not in sal.columns or LABEL_COL not in sal.columns:
        raise SystemExit(f"[错误] 薪资表缺少 onet_job / {LABEL_COL} 列")

    df = pf.merge(
        sal[["onet_job", LABEL_COL]].drop_duplicates("onet_job"),
        left_on=GROUP_COL, right_on="onet_job", how="left",
    )
    if df[LABEL_COL].isna().any():
        bad = sorted(df.loc[df[LABEL_COL].isna(), GROUP_COL].unique())
        raise SystemExit(f"[错误] 以下岗位缺少工资标签: {bad}")

    df = df.dropna(subset=FEATURES + [LABEL_COL]).reset_index(drop=True)
    return df


def group_folds(df: pd.DataFrame, n_splits: int, seed: int):
    """按岗位分组的 K 折划分（训练/测试不共享岗位，防泄漏）。"""
    from sklearn.model_selection import GroupKFold
    groups = df[GROUP_COL].to_numpy()
    n_groups = len(np.unique(groups))
    k = max(2, min(int(n_splits), n_groups))
    gkf = GroupKFold(n_splits=k)
    return list(gkf.split(df, groups=groups)), k


def build_model(task: str):
    """按 config.model.xgb_params 构建模型；objective 由任务决定。"""
    import xgboost as xgb
    params = dict(CFG.get("model", {}).get("xgb_params", {}))
    seed = int(CFG.get("model", {}).get("random_seed", 42))
    params.update({"random_state": seed, "n_jobs": 4})
    if task == "salary":
        params.setdefault("objective", "reg:squarederror")
        return xgb.XGBRegressor(**params)
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    return xgb.XGBClassifier(**params)


def cv_evaluate(df: pd.DataFrame, task: str) -> dict:
    """按岗位分组做 K 折交叉验证，返回指标均值。"""
    from sklearn.metrics import (accuracy_score, mean_absolute_error,
                                 r2_score, roc_auc_score)
    folds, k = group_folds(df, CFG.get("model", {}).get("n_splits", 5),
                           CFG.get("model", {}).get("random_seed", 42))
    X = df[FEATURES]
    y = df[LABEL_COL] if task == "salary" else (df[LABEL_COL] >= df[LABEL_COL].median()).astype(int)

    metrics = {"r2": [], "mae": [], "auc": [], "acc": []}
    for tr, te in folds:
        model = build_model(task)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        if task == "salary":
            metrics["r2"].append(r2_score(y.iloc[te], pred))
            metrics["mae"].append(mean_absolute_error(y.iloc[te], pred))
        else:
            prob = model.predict_proba(X.iloc[te])[:, 1]
            metrics["auc"].append(roc_auc_score(y.iloc[te], prob))
            metrics["acc"].append(accuracy_score(y.iloc[te], (prob >= 0.5).astype(int)))

    out = {"task": task, "n_splits": k}
    for name, vals in metrics.items():
        if vals:
            out[f"cv_{name}_mean"] = round(float(np.mean(vals)), 4)
            out[f"cv_{name}_std"] = round(float(np.std(vals)), 4)
    return out


def fit_and_explain(df: pd.DataFrame, task: str, do_plots: bool) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """按岗位分组切一次留出集 -> 训练 -> 指标 + SHAP 归因（解释在测试集上）。"""
    import shap
    from sklearn.metrics import (accuracy_score, mean_absolute_error,
                                 r2_score, roc_auc_score)
    from sklearn.model_selection import GroupShuffleSplit

    seed = int(CFG.get("model", {}).get("random_seed", 42))
    test_size = float(CFG.get("model", {}).get("test_size", 0.2))

    X = df[FEATURES]
    y = df[LABEL_COL] if task == "salary" else (df[LABEL_COL] >= df[LABEL_COL].median()).astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups=df[GROUP_COL]))

    model = build_model(task)
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])

    holdout = {"task": task, "n_train": int(len(tr)), "n_test": int(len(te)),
               "n_test_jobs": int(df.iloc[te][GROUP_COL].nunique())}
    if task == "salary":
        holdout["holdout_r2"] = round(float(r2_score(y.iloc[te], pred)), 4)
        holdout["holdout_mae"] = round(float(mean_absolute_error(y.iloc[te], pred)), 1)
    else:
        prob = model.predict_proba(X.iloc[te])[:, 1]
        holdout["holdout_auc"] = round(float(roc_auc_score(y.iloc[te], prob)), 4)
        holdout["holdout_acc"] = round(float(accuracy_score(y.iloc[te], (prob >= 0.5).astype(int))), 4)

    # 增益重要性（模型自带，不依赖 SHAP 包）
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.DataFrame({
        "task": task,
        "feature": FEATURES,
        "gain": [round(float(gain.get(f, 0.0)), 4) for f in FEATURES],
    })
    imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    imp = imp.sort_values("gain", ascending=False)

    # SHAP 解释（在留出测试集上，避免解释训练集上的过拟合噪声）
    explainer = shap.TreeExplainer(model)
    sv = explainer(X.iloc[te])
    sv.feature_names = FEATURES

    if do_plots:
        shap_dir = P.get("shap_dir", ROOT / "reports")
        shap_dir.mkdir(parents=True, exist_ok=True)
        for plot_type, suffix in (("dot", "summary"), ("bar", "bar")):
            plt.figure()
            shap.summary_plot(sv, X.iloc[te], plot_type=plot_type,
                              show=False, feature_names=FEATURES)
            plt.title(f"SHAP {suffix} — {task}")
            plt.tight_layout()
            out_png = shap_dir / f"shap_{suffix}_{task}.png"
            plt.savefig(out_png, dpi=180, bbox_inches="tight")
            plt.close()
            print(f"[SHAP] 已保存 {out_png}")

    # 平均 |SHAP| 排序（比 gain 更贴近"对预测的平均影响"）
    shap_abs = pd.DataFrame(np.abs(sv.values), columns=FEATURES).mean().sort_values(ascending=False)
    imp_shap = pd.DataFrame({
        "task": task,
        "feature": shap_abs.index,
        "mean_abs_shap": shap_abs.values.round(4),
    }).reset_index(drop=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgb_{task}.json"
    model.get_booster().save_model(str(model_path))
    print(f"[模型] 已保存 {model_path}")

    return holdout, imp, imp_shap


def run_job_level(df: pd.DataFrame, do_plots: bool) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """岗位级聚合模式：29 个岗位 × 平均路径特征 -> BLS 工资（留一法评估）。

    为什么单独做：同一岗位的工资是常量，配对级样本（44 专业一个岗位）的
    特征差异来自专业侧；用它们预测"岗位工资"等价于让模型猜岗位。
    聚合到岗位级后，问题变成"岗位的平均技能缺口/代价是否与该岗位工资相关"，
    这才是可解释、可评估的问题。
    """
    import shap
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import LeaveOneOut

    jl = df.groupby(GROUP_COL)[FEATURES].mean().reset_index()
    wage = df.groupby(GROUP_COL)[LABEL_COL].first().reset_index()
    jl = jl.merge(wage, on=GROUP_COL)
    X, y = jl[FEATURES], jl[LABEL_COL]

    # 留一法交叉验证（n=29，用普通 KFold 会有乐观偏差）
    preds = np.zeros(len(jl))
    for tr, te in LeaveOneOut().split(X):
        model = build_model("salary")
        model.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = model.predict(X.iloc[te])
    metrics = {
        "task": "salary_job",
        "n_jobs": int(len(jl)),
        "loocv_r2": round(float(r2_score(y, preds)), 4),
        "loocv_mae": round(float(mean_absolute_error(y, preds)), 1),
    }

    # 用全部 29 个岗位训练最终模型并解释
    model = build_model("salary")
    model.fit(X, y)
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.DataFrame({
        "task": "salary_job",
        "feature": FEATURES,
        "gain": [round(float(gain.get(f, 0.0)), 4) for f in FEATURES],
    })
    imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    imp = imp.sort_values("gain", ascending=False)

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    sv.feature_names = FEATURES
    if do_plots:
        shap_dir = P.get("shap_dir", ROOT / "reports")
        shap_dir.mkdir(parents=True, exist_ok=True)
        for plot_type, suffix in (("dot", "summary"), ("bar", "bar")):
            plt.figure()
            shap.summary_plot(sv, X, plot_type=plot_type, show=False, feature_names=FEATURES)
            plt.title(f"SHAP {suffix} — salary_job（29 个岗位）")
            plt.tight_layout()
            out_png = shap_dir / f"shap_{suffix}_salary_job.png"
            plt.savefig(out_png, dpi=180, bbox_inches="tight")
            plt.close()
            print(f"[SHAP] 已保存 {out_png}")

    shap_abs = pd.DataFrame(np.abs(sv.values), columns=FEATURES).mean().sort_values(ascending=False)
    imp_shap = pd.DataFrame({
        "task": "salary_job",
        "feature": shap_abs.index,
        "mean_abs_shap": shap_abs.values.round(4),
    }).reset_index(drop=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgb_salary_job.json"
    model.get_booster().save_model(str(model_path))
    print(f"[模型] 已保存 {model_path}")
    return metrics, imp, imp_shap


def main():
    ap = argparse.ArgumentParser(description="路径特征 -> XGBoost + SHAP（外部标签：BLS 工资）")
    ap.add_argument("--task", choices=["salary", "high_salary", "salary_job", "all"], default="all")
    ap.add_argument("--no-plots", action="store_true", help="不生成 SHAP 图")
    args = ap.parse_args()

    df = load_data()
    jobs = df[GROUP_COL].nunique()
    unique_wage = df[LABEL_COL].nunique()
    print(f"[数据] {len(df)} 行 | 岗位 {jobs} | 唯一工资值 {unique_wage} | 特征 {len(FEATURES)} 个")
    print("[注意] 同一岗位的样本高度相关，所有评估按岗位分组（GroupKFold），结论是相关性/归因而非因果")

    tasks = ["salary", "high_salary", "salary_job"] if args.task == "all" else [args.task]

    rows_metrics, rows_imp, rows_shap = [], [], []
    for task in tasks:
        print(f"\n========== 任务: {task} ==========")
        if task == "salary_job":
            m, imp, imp_shap = run_job_level(df, do_plots=not args.no_plots)
            rows_metrics.append(m)
            rows_imp.append(imp)
            rows_shap.append(imp_shap)
            print(f"[留一法] R2={m['loocv_r2']} | MAE={m['loocv_mae']} 美元 | 岗位数={m['n_jobs']}")
            print("[增益重要性 Top5]")
            for _, r in imp.head(5).iterrows():
                print(f"  {r['feature']:24s} gain={r['gain']:.0f} ({r['gain_share']*100:.1f}%)")
            print("[平均|SHAP| Top5]")
            for _, r in imp_shap.head(5).iterrows():
                print(f"  {r['feature']:24s} mean|SHAP|={r['mean_abs_shap']:.1f}")
            continue
        m = cv_evaluate(df, task)
        h, imp, imp_shap = fit_and_explain(df, task, do_plots=not args.no_plots)
        m.update(h)
        rows_metrics.append(m)
        rows_imp.append(imp)
        rows_shap.append(imp_shap)

        if task == "salary":
            print(f"[交叉验证] R2={m.get('cv_r2_mean')}±{m.get('cv_r2_std')} | "
                  f"MAE={m.get('cv_mae_mean')}±{m.get('cv_mae_std')} 美元")
            print(f"[留出集]   R2={m.get('holdout_r2')} | MAE={m.get('holdout_mae')} 美元")
        else:
            print(f"[交叉验证] AUC={m.get('cv_auc_mean')}±{m.get('cv_auc_std')} | "
                  f"ACC={m.get('cv_acc_mean')}±{m.get('cv_acc_std')}")
            print(f"[留出集]   AUC={m.get('holdout_auc')} | ACC={m.get('holdout_acc')}")

        print("[增益重要性 Top5]")
        for _, r in imp.head(5).iterrows():
            print(f"  {r['feature']:24s} gain={r['gain']:.0f} ({r['gain_share']*100:.1f}%)")
        print("[平均|SHAP| Top5]")
        for _, r in imp_shap.head(5).iterrows():
            print(f"  {r['feature']:24s} mean|SHAP|={r['mean_abs_shap']:.3f}")

    metrics_df = pd.DataFrame(rows_metrics)
    imp_df = pd.concat(rows_imp, ignore_index=True)
    shap_df = pd.concat(rows_shap, ignore_index=True)
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8")
    imp_df.to_csv(OUT_IMPORTANCE, index=False, encoding="utf-8")
    shap_df.to_csv(ROOT / "reports" / "xgb_shap_importance.csv", index=False, encoding="utf-8")
    print(f"\n[输出] {OUT_METRICS}")
    print(f"[输出] {OUT_IMPORTANCE}")
    print(f"[输出] {ROOT / 'reports' / 'xgb_shap_importance.csv'}")
    print("\n[解读提醒] 只有 29 个唯一工资值，样本按岗位分组评估；"
          "SHAP 图说明'特征与预测的关联方向/强度'，用于解释与验证，不宣称因果。")


if __name__ == "__main__":
    main()
