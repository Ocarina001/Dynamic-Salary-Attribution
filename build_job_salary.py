# -*- coding: utf-8 -*-
"""
build_job_salary.py —— 生成 29 个岗位的岗位级薪资表
=====================================================
输入:
  - config.yaml 的 jobs: 显示岗位名 -> O*NET 标准职业名
  - onet/occupation_data.csv: O*NET 职业名 -> O*NET-SOC Code
  - data/bls/us_wages_by_occupation_national_2024.csv：BLS OEWS 全国薪资（自动扫描）

输出:
  - job_soc.csv                 29 行岗位 -> SOC6 映射
  - job_salary.csv              岗位级薪资表（BLS OEWS）
  - job_aggregated_features.csv 按岗位聚合的路径特征
  - job_features_salary.csv     路径特征 + 岗位薪资
  - reports/job_salary_coverage.csv
  - reports/job_salary_summary.csv
  - reports/job_salary_scatter.png

用法:
  # 1) 有 BLS OEWS 文件（推荐）
  python build_job_salary.py --bls data/bls/oesm25nat.zip

  # 2) 自动扫描 data/bls/
  python build_job_salary.py

  # 3) 只生成 29 行空模板
  python build_job_salary.py --template-only

BLS OEWS 下载:
  https://www.bls.gov/oes/
  下载 National 的最新 ZIP，放到 data/bls/ 后重跑。
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
P = {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in CFG["paths"].items()}

BLS_DIR = ROOT / "data" / "bls"
REPORTS_DIR = ROOT / "reports"

SALARY_FIELDS = [
    "occ_title",
    "employment",
    "n_salary_rows",
    "mean_wage",
    "median_wage",
    "p25_wage",
    "p75_wage",
    "year",
    "area",
    "source",
]


def build_job_soc() -> pd.DataFrame:
    occ = pd.read_csv(P["occupation"])
    title2code = dict(zip(occ["Title"], occ["O*NET-SOC Code"]))

    rows = []
    missing = []
    for display_job, onet_title in CFG["jobs"].items():
        code = title2code.get(onet_title)
        if code is None:
            missing.append((display_job, onet_title))
            continue
        rows.append({
            "onet_job": display_job,
            "onet_title": onet_title,
            "onet_soc": code,
            "soc6": code.split(".")[0],
        })

    if missing:
        print("[WARN] 以下岗位在 occupation_data.csv 中找不到 O*NET-SOC Code:")
        for display_job, onet_title in missing:
            print(f"       {display_job} -> {onet_title}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[错误] job_soc 为空，请检查 config.yaml 和 occupation_data.csv")

    group_size = df.groupby("soc6")["onet_job"].transform("size")
    df["soc_group_size"] = group_size.astype(int)
    df["shared_soc"] = df["soc_group_size"] > 1
    return df


def infer_year(path: Path) -> int | None:
    m = re.search(r"(20\d{2})", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"oesm(\d{2})", path.name, re.I)
    if m:
        return 2000 + int(m.group(1))
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip()).upper() for c in df.columns]
    return df


def _to_number(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(",", "", regex=False).str.strip()
    text = text.replace({"#": np.nan, "**": np.nan, "*": np.nan,
                         "-": np.nan, "—": np.nan, "": np.nan, "NAN": np.nan})
    return pd.to_numeric(text, errors="coerce")


def read_table(path: Path, depth: int = 0) -> pd.DataFrame:
    if depth > 2:
        raise ValueError(f"ZIP 嵌套层级过深: {path}")
    suffix = path.suffix.lower()

    if suffix == ".csv":
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError as e:
                last_err = e
        raise last_err
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".zip":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path) as z:
                z.extractall(td)
            candidates = [
                p for p in Path(td).rglob("*")
                if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".zip"}
            ]
            if not candidates:
                raise FileNotFoundError(f"ZIP 中没有 CSV/XLSX: {path}")
            candidates.sort(key=lambda p: (
                0 if "national" in p.name.lower() else 1,
                0 if p.suffix.lower() == ".csv" else 1,
                p.name,
            ))
            return read_table(candidates[0], depth + 1)

    raise ValueError(f"不支持的文件类型: {path}")


def find_bls_candidates(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".zip"}
        and ("national" in p.name.lower() or "oesm" in p.name.lower())
    ]
    paths.sort(key=lambda p: (infer_year(p) or 0, p.stat().st_size), reverse=True)
    return paths


def standardize_oews(df: pd.DataFrame, path: Path, year: int | None = None) -> pd.DataFrame:
    df = _normalize_columns(df)

    # 兼容 BLS 原始列名和常见的清洗版列名（如 Occupation Code / Annual Median）
    alias = {
        "OCCUPATION_CODE": "OCC_CODE",
        "OCCUPATION_TITLE": "OCC_TITLE",
        "EMPLOYMENT": "TOT_EMP",
        "ANNUAL_MEDIAN": "A_MEDIAN",
        "ANNUAL_MEAN": "A_MEAN",
        "ANNUAL_25TH_PCTL": "A_PCT25",
        "ANNUAL_75TH_PCTL": "A_PCT75",
        "ANNUAL_10TH_PCTL": "A_PCT10",
        "ANNUAL_90TH_PCTL": "A_PCT90",
        "HOURLY_MEDIAN": "H_MEDIAN",
    }
    df = df.rename(columns={
        src: dst for src, dst in alias.items()
        if src in df.columns and dst not in df.columns
    })

    required = {"OCC_CODE", "OCC_TITLE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OEWS 缺少列 {sorted(missing)}；实际列: {list(df.columns)[:20]}")

    for c in ("OCC_CODE", "OCC_TITLE", "O_GROUP", "I_GROUP"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "O_GROUP" in df.columns:
        mask = df["O_GROUP"].str.lower().eq("detailed")
        if mask.any():
            df = df[mask].copy()
        else:
            print("[WARN] O_GROUP 中没有 detailed，保留全部行")

    if "I_GROUP" in df.columns:
        mask = df["I_GROUP"].str.lower().eq("cross-industry")
        if mask.any():
            df = df[mask].copy()
        else:
            print("[WARN] I_GROUP 中没有 Cross-industry，保留全部行")

    num_cols = ["TOT_EMP", "A_MEAN", "A_MEDIAN", "A_PCT25", "A_PCT75",
                "H_MEAN", "H_MEDIAN", "H_PCT25", "H_PCT75"]
    for c in num_cols:
        if c in df.columns:
            df[c] = _to_number(df[c])

    for annual, hourly in (("A_MEAN", "H_MEAN"), ("A_MEDIAN", "H_MEDIAN"),
                           ("A_PCT25", "H_PCT25"), ("A_PCT75", "H_PCT75")):
        if annual not in df.columns:
            df[annual] = np.nan
        if hourly in df.columns:
            df[annual] = df[annual].fillna(df[hourly] * 2080.0)

    def col(name):
        return df[name] if name in df.columns else np.nan

    out = pd.DataFrame({
        "soc6": df["OCC_CODE"].astype(str).str.strip(),
        "occ_title": df["OCC_TITLE"],
        "employment": col("TOT_EMP"),
        "mean_wage": col("A_MEAN"),
        "median_wage": col("A_MEDIAN"),
        "p25_wage": col("A_PCT25"),
        "p75_wage": col("A_PCT75"),
    })
    out["n_salary_rows"] = np.nan
    out["year"] = year if year is not None else infer_year(path)
    out["area"] = "US national"
    out["source"] = "BLS OEWS"

    out = out.sort_values(
        ["soc6", "employment"], ascending=[True, False], na_position="last"
    ).drop_duplicates("soc6", keep="first")
    return out.reset_index(drop=True)


def load_manual_salary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "onet_job" not in df.columns:
        raise ValueError(f"手工薪资表缺少 onet_job 列: {path}")

    out = df.copy()
    for c in SALARY_FIELDS:
        if c not in out.columns:
            out[c] = None

    out["median_wage"] = pd.to_numeric(out["median_wage"], errors="coerce")
    out["mean_wage"] = pd.to_numeric(out["mean_wage"], errors="coerce")
    out["median_wage"] = out["median_wage"].fillna(out["mean_wage"])
    for c in ("p25_wage", "p75_wage", "employment", "n_salary_rows"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["source"] = out["source"].fillna("manual")
    out["area"] = out["area"].fillna("manual")
    out["year"] = out["year"].fillna("manual")
    return out[out["median_wage"].notna()].copy()


def load_job_features() -> pd.DataFrame | None:
    agg_path = ROOT / "job_aggregated_features.csv"
    if agg_path.exists():
        return pd.read_csv(agg_path)

    path = P["path_features"]
    if not path.exists():
        return None

    df = pd.read_csv(path)
    spec = {
        "avg_path_length": ("path_length", "mean"),
        "avg_skill_gap": ("skill_gap", "mean"),
        "std_skill_gap": ("skill_gap", "std"),
        "min_skill_gap": ("skill_gap", "min"),
        "max_skill_gap": ("skill_gap", "max"),
        "avg_semantic_score": ("semantic_score", "mean"),
        "std_semantic_score": ("semantic_score", "std"),
        "avg_baseline_gap": ("baseline_gap", "mean"),
        "avg_baseline_best_gap": ("baseline_best_gap", "mean"),
        "avg_gap_reduction": ("gap_reduction", "mean"),
        "avg_gap_reduction_vs_best": ("gap_reduction_vs_best", "mean"),
        "avg_cost": ("cost", "mean"),
    }
    spec = {k: v for k, v in spec.items() if v[0] in df.columns}
    agg = df.groupby("target_job").agg(**spec).reset_index()
    for c in ("std_skill_gap", "std_semantic_score"):
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)

    agg.to_csv(agg_path, index=False, encoding="utf-8")
    print(f"[特征] 从 {path.name} 聚合出 {len(agg)} 行 -> {agg_path.name}")
    return agg


def init_salary_base(job_soc: pd.DataFrame) -> pd.DataFrame:
    base = job_soc.copy()
    for c in SALARY_FIELDS:
        if c not in base.columns:
            base[c] = pd.Series([None] * len(base), index=base.index, dtype="object")
    base["salary_source"] = "missing"
    return base


def attach_salary(base: pd.DataFrame, sal: pd.DataFrame, source_label: str) -> pd.DataFrame:
    sal = sal.copy()
    keep = ["onet_job"] + SALARY_FIELDS
    sal = sal[[c for c in keep if c in sal.columns]].drop_duplicates("onet_job")
    sal = sal.set_index("onet_job")

    mask = base["onet_job"].isin(sal.index) & base["salary_source"].eq("missing")
    if mask.any():
        for c in SALARY_FIELDS:
            if c in sal.columns:
                base.loc[mask, c] = base.loc[mask, "onet_job"].map(sal[c])
        base.loc[mask, "salary_source"] = source_label

    return base


def build_template(job_soc: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "onet_job", "onet_title", "onet_soc", "soc6", "shared_soc", "soc_group_size",
        "occ_title", "employment", "n_salary_rows", "mean_wage", "median_wage",
        "p25_wage", "p75_wage", "year", "area", "source", "salary_source",
        "manual_note",
    ]
    t = job_soc.copy()
    for c in cols:
        if c not in t.columns:
            t[c] = np.nan
    t["source"] = "manual"
    t["salary_source"] = "manual"
    t["manual_note"] = ""
    return t[cols]


def build_coverage(job_salary: pd.DataFrame, merged: pd.DataFrame | None):
    cols = [
        "onet_job", "onet_title", "onet_soc", "soc6", "shared_soc", "soc_group_size",
        "occ_title", "employment", "n_salary_rows", "median_wage", "p25_wage",
        "p75_wage", "year", "area", "salary_source",
    ]
    out = job_salary[[c for c in cols if c in job_salary.columns]].copy()
    out["has_salary"] = out["median_wage"].notna()

    n_jobs = len(out)
    n_with = int(out["has_salary"].sum())
    summary = {
        "n_jobs": n_jobs,
        "n_with_salary": n_with,
        "coverage": round(n_with / n_jobs, 4) if n_jobs else 0.0,
        "n_missing": n_jobs - n_with,
        "n_shared_soc_groups": int(
            (job_salary.groupby("soc6")["onet_job"].nunique() > 1).sum()
        ),
        "salary_sources": "; ".join(
            f"{k}:{v}" for k, v in out["salary_source"].value_counts().items()
        ),
    }

    if merged is not None and not merged.empty:
        for feature, key in (("avg_skill_gap", "spearman_gap_wage"),
                             ("avg_cost", "spearman_cost_wage")):
            if feature in merged.columns and "median_wage" in merged.columns:
                sub = merged.dropna(subset=[feature, "median_wage"])
                summary[key] = (
                    round(float(sub[feature].corr(sub["median_wage"], method="spearman")), 4)
                    if len(sub) >= 3 else np.nan
                )

    return out, pd.DataFrame([summary])


def plot_scatter(merged: pd.DataFrame, out_path: Path):
    sub = merged.dropna(subset=["median_wage"]).copy()
    if len(sub) < 2:
        print("[图] 有薪资的岗位少于 2 个，跳过散点图")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import StrMethodFormatter
    except ImportError as e:
        print(f"[WARN] matplotlib 不可用，跳过散点图: {e}")
        return

    label_col = "target_job" if "target_job" in sub.columns else "onet_job"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    specs = [
        ("avg_skill_gap", "Skill gap vs median wage"),
        ("avg_cost", "Path cost vs median wage"),
    ]
    for ax, (xcol, title) in zip(axes, specs):
        if xcol not in sub.columns:
            ax.axis("off")
            continue
        ax.scatter(sub[xcol], sub["median_wage"], s=60, alpha=0.85)
        for _, row in sub.iterrows():
            ax.annotate(
                str(row[label_col]), (row[xcol], row["median_wage"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points",
            )
        ax.set_xlabel(xcol)
        ax.set_ylabel("Median annual wage (USD)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    fig.suptitle("Job-level salary vs path features")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 已保存 {out_path}")


def main():
    parser = argparse.ArgumentParser(description="生成 29 个岗位的岗位级薪资表")
    parser.add_argument("--bls", type=str, default=None,
                        help="BLS OEWS 全国 CSV/XLSX/ZIP 路径；不填则自动扫描 data/bls/")
    parser.add_argument("--salary-csv", type=str, default=None,
                        help="手工薪资表 CSV（至少含 onet_job, median_wage）")
    parser.add_argument("--template-only", action="store_true",
                        help="只生成 job_soc.csv 和 job_salary_template.csv")
    parser.add_argument("--year", type=int, default=None, help="覆盖薪资数据年份")
    parser.add_argument("--outdir", type=str, default=str(ROOT), help="输出目录")
    parser.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR), help="报告目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成散点图")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    reports_dir = Path(args.reports_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    BLS_DIR.mkdir(parents=True, exist_ok=True)

    job_soc = build_job_soc()
    job_soc.to_csv(outdir / "job_soc.csv", index=False, encoding="utf-8")

    print(f"[SOC] {len(job_soc)} 个岗位 | {job_soc['soc6'].nunique()} 个唯一 SOC6 | "
          f"{int((job_soc.groupby('soc6')['onet_job'].nunique() > 1).sum())} 个共享 SOC 组")

    if args.template_only:
        build_template(job_soc).to_csv(
            outdir / "job_salary_template.csv", index=False, encoding="utf-8"
        )
        print(f"[模板] 已生成 {outdir / 'job_salary_template.csv'}")
        print("[提示] 把 BLS OEWS 文件放到 data/bls/，或手动填写模板后另存为 job_salary.csv")
        return

    bls_path = Path(args.bls) if args.bls else None
    if bls_path is not None and not bls_path.exists():
        raise SystemExit(f"[错误] BLS 文件不存在: {bls_path}")
    if bls_path is None:
        candidates = find_bls_candidates(BLS_DIR)
        if candidates:
            bls_path = candidates[0]

    job_salary = init_salary_base(job_soc)

    if args.salary_csv is not None:
        manual_path = Path(args.salary_csv)
        manual = load_manual_salary(manual_path)
        job_salary = attach_salary(job_salary, manual, "manual")
        print(f"[手工] {manual_path.name} | 覆盖 {manual['onet_job'].nunique()} 个岗位")

    if bls_path is not None:
        try:
            raw = read_table(bls_path)
            oews = standardize_oews(raw, bls_path, args.year)
            bls_job = job_soc[["onet_job", "soc6"]].merge(oews, on="soc6", how="left")
            job_salary = attach_salary(job_salary, bls_job, "bls_oews")
            print(f"[BLS] {bls_path.name} | 原始 {len(raw)} 行 -> {len(oews)} 个 SOC")
        except Exception as e:
            print(f"[WARN] BLS 读取失败，跳过 BLS: {e}")
            bls_path = None

    for c in ("employment", "n_salary_rows", "mean_wage", "median_wage", "p25_wage", "p75_wage"):
        job_salary[c] = pd.to_numeric(job_salary[c], errors="coerce")

    job_salary["source"] = job_salary["source"].fillna("missing")
    job_salary.to_csv(outdir / "job_salary.csv", index=False, encoding="utf-8")

    features = load_job_features()
    merged = None
    if features is not None:
        merged = features.merge(
            job_salary, left_on="target_job", right_on="onet_job",
            how="left", suffixes=("", "_salary"),
        )
        merged.to_csv(outdir / "job_features_salary.csv", index=False, encoding="utf-8")
        print(f"[合并] {len(merged)} 行 -> {outdir / 'job_features_salary.csv'}")
    else:
        print("[WARN] 找不到 job_aggregated_features.csv 或 path_features.csv，跳过合并")

    coverage, summary = build_coverage(job_salary, merged)
    coverage.to_csv(reports_dir / "job_salary_coverage.csv", index=False, encoding="utf-8")
    summary.to_csv(reports_dir / "job_salary_summary.csv", index=False, encoding="utf-8")

    if not args.no_plot and merged is not None:
        plot_scatter(merged, reports_dir / "job_salary_scatter.png")

    n_jobs = len(job_salary)
    n_with = int(job_salary["median_wage"].notna().sum())
    print("=" * 70)
    print(f"岗位数: {n_jobs} | 有薪资: {n_with} | 覆盖率: {n_with / n_jobs:.1%}")
    print(f"薪资来源: {summary.loc[0, 'salary_sources']}")
    missing = job_salary.loc[job_salary["median_wage"].isna(), "onet_job"].tolist()
    if missing:
        print(f"缺少薪资: {', '.join(missing)}")
    print(f"输出: {outdir / 'job_salary.csv'}")
    print(f"报告: {reports_dir / 'job_salary_coverage.csv'}")
    if bls_path is None and n_with < n_jobs:
        print("[提示] 当前不是 29 岗位全覆盖；把 BLS OEWS 文件放到 data/bls/ 后重跑即可。")


if __name__ == "__main__":
    main()
