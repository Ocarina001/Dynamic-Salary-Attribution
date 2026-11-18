# -*- coding: utf-8 -*-
"""
recommend_major_jobs.py —— 专业 → 岗位技能匹配 + 路径 + 薪资参考
================================================================
核心原则:
  排名只使用技能匹配与路径指标，薪资只作为附加参考列，不参与匹配分/排名。

输入:
  - path_features_specific.csv   44 专业 × 29 岗位的路径特征
  - job_salary.csv               29 个岗位的 BLS 薪资（仅参考）
  - job_skills_unified_specific.csv + major_skills_specific.csv
                                 用于计算岗位核心技能覆盖率

输出:
  - major_job_specific_ranking.csv   44 × 29 全量匹配结果 + 薪资参考
  - major_job_specific_top5.csv      每个专业技能匹配最高的 TopN
  - major_job_specific_summary.csv   每个专业一行摘要
  - reports/major_job_specific_heatmap.png
  - reports/major_job_specific_quadrant.png（匹配 vs 薪资参考）
  - reports/major_job_specific_<专业>.png

用法:
  python recommend_major_jobs.py
  python recommend_major_jobs.py --major 人工智能
  python recommend_major_jobs.py --top-n 5

说明:
  fit_score      = 1 - skill_gap
  core_coverage  = 专业技能 ∩ 岗位 Top10 核心技能 / 岗位 Top10 核心技能数
  high_fit       = fit_score >= 0.50 且 core_coverage >= 0.50
  match_rank     = 先高匹配，再按 fit_score、core_coverage、path_length 排序
  salary_*       = BLS 参考信息，不参与 match_rank 计算
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
P = {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in CFG["paths"].items()}

REPORTS_DIR = ROOT / "reports"


def _default_side_files(path_features: Path):
    pf = str(path_features)
    if "specific" in pf:
        return (
            P["onet_dir"] / "job_skills_unified_specific.csv",
            P["onet_dir"] / "major_skills_specific.csv",
        )
    return P["unified_skills"], P["onet_dir"] / "major_skills_llm.csv"


def load_core_coverage(path_features: Path, ranking: pd.DataFrame) -> pd.DataFrame:
    job_skills_path, major_skills_path = _default_side_files(path_features)
    if not job_skills_path.exists() or not major_skills_path.exists():
        ranking["core_coverage"] = np.nan
        return ranking

    job_skills = pd.read_csv(job_skills_path, encoding="utf-8")
    major_skills = pd.read_csv(major_skills_path, encoding="utf-8")

    # 岗位核心技能：只取 seed / onet_tool 来源的具体技能，排除纯 O*NET 通用能力与旧领域技能
    if "source" in job_skills.columns:
        specific_job = job_skills[
            job_skills["source"].astype(str).str.contains("seed|onet_tool", regex=True)
        ].copy()
    else:
        specific_job = job_skills.copy()
    core_map = {}
    for job, g in specific_job.groupby("job"):
        core = g.sort_values("weight", ascending=False).head(10)["skill"].tolist()
        core_map[job] = set(core)

    # 专业侧核心覆盖只使用培养方案直接命中的具体技能，排除低权重通用背景技能
    if "source" in major_skills.columns:
        major_skills = major_skills[major_skills["source"] == "curriculum"].copy()
    major_map = {
        major: set(g["skill"])
        for major, g in major_skills.groupby("major")
    }

    def _cov(row):
        core = core_map.get(row["target_job"], set())
        if not core:
            return np.nan
        return len(major_map.get(row["source_major"], set()) & core) / len(core)

    ranking["core_coverage"] = ranking.apply(_cov, axis=1)
    return ranking


def load_inputs(path_features: Path) -> pd.DataFrame:
    path = pd.read_csv(path_features, encoding="utf-8")
    salary = pd.read_csv(ROOT / "job_salary.csv", encoding="utf-8")

    need = ["onet_job", "median_wage", "p25_wage", "p75_wage", "salary_source"]
    missing = [c for c in need if c not in salary.columns]
    if missing:
        raise SystemExit(f"[错误] job_salary.csv 缺少列: {missing}")

    salary = salary[need].copy()
    df = path.merge(salary, left_on="target_job", right_on="onet_job", how="left")
    for c in ("median_wage", "p25_wage", "p75_wage"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.empty:
        raise SystemExit("[错误] 没有可用的专业×岗位记录")
    missing_salary = int(df["median_wage"].isna().sum())
    if missing_salary:
        print(f"[WARN] {missing_salary} 行缺少岗位薪资（匹配不受影响，仅作为参考列）")
    return df


def build_ranking(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fit_score"] = 1.0 - out["skill_gap"]

    # 绝对匹配：fit >= 0.50 且岗位核心技能覆盖率 >= 50%
    out["high_fit"] = (out["fit_score"] >= 0.50) & \
        (out["core_coverage"].fillna(0) >= 0.50)

    # 薪资只作为参考：全局中位数二分，不参与 match_rank
    out["high_salary"] = out["median_wage"] >= out["median_wage"].median()
    out["salary_rank"] = out["median_wage"].rank(ascending=False, method="min")

    out["quadrant"] = np.select(
        [
            out["high_fit"] & out["high_salary"],
            out["high_fit"] & ~out["high_salary"],
            ~out["high_fit"] & out["high_salary"],
        ],
        ["高匹配高薪", "高匹配低薪", "低匹配高薪"],
        default="低匹配低薪",
    )

    # match_rank 只按技能匹配与路径排序，薪资不参与
    out["match_rank"] = np.nan
    for _, g in out.groupby("source_major", sort=False):
        high = g[g["high_fit"]].sort_values(
            ["fit_score", "core_coverage", "path_length"],
            ascending=[False, False, True],
        )
        low = g[~g["high_fit"]].sort_values(
            ["fit_score", "core_coverage", "path_length"],
            ascending=[False, False, True],
        )
        ordered = pd.concat([high, low])
        out.loc[ordered.index, "match_rank"] = range(1, len(ordered) + 1)

    return out.sort_values(
        ["source_major", "match_rank", "target_job"]
    ).reset_index(drop=True)


def build_outputs(ranking: pd.DataFrame, top_n: int):
    top = ranking[ranking["high_fit"]].sort_values(
        ["source_major", "match_rank"]
    ).groupby("source_major", sort=False).head(top_n).copy()

    rows = []
    for major, g in ranking.groupby("source_major", sort=False):
        high_fit = g[g["high_fit"]].sort_values(
            ["fit_score", "core_coverage", "path_length"],
            ascending=[False, False, True],
        )
        best = high_fit.iloc[0] if not high_fit.empty else None

        high_salary_high_fit = high_fit[high_fit["high_salary"]].sort_values(
            ["median_wage", "fit_score"], ascending=[False, False]
        )
        best_high_salary = high_salary_high_fit.iloc[0] if not high_salary_high_fit.empty else None
        top3 = high_fit.head(3)

        rows.append({
            "source_major": major,
            "n_jobs": len(g),
            "n_high_fit": len(high_fit),
            "n_high_fit_high_salary": len(high_salary_high_fit),
            "best_match_job": best["target_job"] if best is not None else "",
            "best_match_fit": round(float(best["fit_score"]), 4) if best is not None else np.nan,
            "best_match_core_coverage": round(float(best["core_coverage"]), 4) if best is not None else np.nan,
            "best_match_salary": round(float(best["median_wage"]), 0) if best is not None and pd.notna(best["median_wage"]) else np.nan,
            "best_high_salary_job": best_high_salary["target_job"] if best_high_salary is not None else "",
            "best_high_salary_wage": round(float(best_high_salary["median_wage"]), 0) if best_high_salary is not None and pd.notna(best_high_salary["median_wage"]) else np.nan,
            "top3_match_jobs": "; ".join(
                f"{r['target_job']} (fit={r['fit_score']:.3f}, core={r['core_coverage']:.2f}, ${r['median_wage']:,.0f})"
                for _, r in top3.iterrows()
                if pd.notna(r["median_wage"])
            ),
        })

    summary = pd.DataFrame(rows)
    return top, summary


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Arial Unicode MS", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def plot_heatmap(ranking: pd.DataFrame, out_path: Path):
    plt = setup_matplotlib()
    pivot = ranking.pivot_table(
        index="source_major", columns="target_job",
        values="fit_score", aggfunc="mean",
    )
    salary_order = (
        ranking.groupby("target_job")["median_wage"].first()
        .sort_values(ascending=False).index
    )
    pivot = pivot[salary_order]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(18, 12))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("专业 × 岗位技能匹配度（1 - skill_gap；列按薪资从高到低）", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 已保存 {out_path}")


def plot_quadrant(ranking: pd.DataFrame, out_path: Path):
    plt = setup_matplotlib()
    colors = {
        "高匹配高薪": "#2e7d32",
        "高匹配低薪": "#1565c0",
        "低匹配高薪": "#ef6c00",
        "低匹配低薪": "#9e9e9e",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, g in ranking.groupby("quadrant"):
        ax.scatter(
            g["fit_score"], g["median_wage"],
            s=18, alpha=0.55, label=f"{label} ({len(g)})",
            c=colors.get(label, "#333333"),
        )
    salary_cut = ranking["median_wage"].median()
    ax.axvline(0.50, color="#444444", lw=1, ls="--")
    ax.axhline(salary_cut, color="#444444", lw=1, ls="--")
    ax.set_xlabel("技能匹配度（fit_score；虚线 = 高匹配绝对阈值 0.50）")
    ax.set_ylabel("岗位年薪中位数（USD，仅参考；虚线 = 全部岗位薪资中位数）")
    ax.set_title("专业 × 岗位：技能匹配 vs 薪资参考（薪资不参与匹配排名）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 已保存 {out_path}")


def safe_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", text)[:80]


def plot_major(ranking: pd.DataFrame, major: str, out_dir: Path, top_n: int = 12,
               file_prefix: str = "major"):
    g = ranking[ranking["source_major"] == major].copy()
    if g.empty:
        print(f"[WARN] 找不到专业: {major}")
        return

    high_fit = g[g["high_fit"]].sort_values(
        ["fit_score", "core_coverage", "path_length"],
        ascending=[False, False, True],
    )
    if high_fit.empty:
        print(f"[WARN] {major} 没有 fit >= 0.50 的高匹配岗位，跳过条形图")
        return
    g = high_fit.head(top_n).iloc[::-1]

    plt = setup_matplotlib()
    colors = {
        "高匹配高薪": "#2e7d32",
        "高匹配低薪": "#1565c0",
        "低匹配高薪": "#ef6c00",
        "低匹配低薪": "#9e9e9e",
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        g["target_job"], g["median_wage"],
        color=[colors.get(x, "#333333") for x in g["quadrant"]],
    )
    for y, (_, row) in enumerate(g.iterrows()):
        ax.text(
            row["median_wage"] + 1500, y,
            f"${row['median_wage']:,.0f} | fit={row['fit_score']:.3f}",
            va="center", fontsize=8,
        )
    ax.set_xlabel("BLS OEWS 岗位年薪中位数（USD，仅参考；排序按技能匹配）")
    ax.set_title(f"{major}：高匹配岗位（按技能匹配排序，薪资为参考）")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_prefix}_{safe_filename(major)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 已保存 {out_path}")


def main():
    parser = argparse.ArgumentParser(description="专业 → 岗位技能匹配 + 路径 + 薪资参考")
    parser.add_argument("--top-n", type=int, default=5, help="每个专业输出 Top N，默认 5")
    parser.add_argument("--major", type=str, default=None,
                        help="额外生成某个专业的 Top 岗位条形图")
    parser.add_argument("--path-features", type=str, default=None,
                        help="替代 path_features.csv，例如 path_features_specific.csv")
    parser.add_argument("--out-prefix", type=str, default="major_job",
                        help="输出文件名前缀，默认 major_job")
    parser.add_argument("--no-plots", action="store_true", help="不生成图")
    args = parser.parse_args()

    path_features = Path(args.path_features) if args.path_features else P["path_features"]
    prefix = args.out_prefix
    if prefix == "major_job" and "specific" in path_features.name:
        prefix = "major_job_specific"
    df = load_inputs(path_features)
    df = load_core_coverage(path_features, df)
    ranking = build_ranking(df)
    top, summary = build_outputs(ranking, args.top_n)

    ranking.to_csv(ROOT / f"{prefix}_ranking.csv", index=False, encoding="utf-8-sig")
    top.to_csv(ROOT / f"{prefix}_top5.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(ROOT / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")

    print(f"[输入] {path_features} | {len(df)} 行 | {df.source_major.nunique()} 专业 × "
          f"{df.target_job.nunique()} 岗位 | 薪资来源 "
          f"{df.salary_source.value_counts().to_dict()}")
    print(f"[输出] {prefix}_ranking.csv / {prefix}_top5.csv / {prefix}_summary.csv")

    if not args.no_plots:
        plot_heatmap(ranking, REPORTS_DIR / f"{prefix}_heatmap.png")
        plot_quadrant(ranking, REPORTS_DIR / f"{prefix}_quadrant.png")
        if args.major:
            plot_major(ranking, args.major, REPORTS_DIR, top_n=12,
                       file_prefix=prefix)

    n_no_fit = int((summary["n_high_fit"] == 0).sum())
    print(f"\n无 fit>=0.50 高匹配岗位的专业数: {n_no_fit}/{len(summary)}")
    print("各专业技能匹配最高的首选岗位（薪资仅参考，不参与排名）：")
    show = summary.dropna(subset=["best_match_fit"]).sort_values(
        "best_match_fit", ascending=False
    ).head(10)
    for _, r in show.iterrows():
        print(f"  {r['source_major']}: {r['best_match_job']} | "
              f"fit={r['best_match_fit']:.3f} | core={r['best_match_core_coverage']:.2f} | "
              f"salary=${r['best_match_salary']:,.0f} | "
              f"高匹配岗位数={r['n_high_fit']} | 高匹配高薪岗位数={r['n_high_fit_high_salary']}")


if __name__ == "__main__":
    main()
