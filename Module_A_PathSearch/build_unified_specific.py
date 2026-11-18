# -*- coding: utf-8 -*-
"""
build_unified_specific.py —— 基于具体技能重建统一岗位技能表 + 岗位相似度矩阵
=================================================================
在整条数据管线中的位置：

  build_specific_skills.py（产出岗位侧具体技能）
        └─> [本脚本] build_unified_specific.py
              ├─> onet/job_skills_unified_specific.csv         （最终岗位技能表）
              └─> Module_B_SemanticMatch/job_similarity_matrix_specific.csv（岗位相似度）
                        └─> path_search.py 建图（岗位跳板边按相似度阈值生成）

做了什么（两步）：

  1) 统一技能表：
     把「旧版通用技能」（O*NET IM 35 项 + 5 项领域技能）与「新版具体技能」
     按 岗位×技能 合并：拼接 -> 去重 -> 同一 (job, skill) 取最大权重。
     取 max 而非求和，是为了避免通用与具体重名技能（如 Statistics）被双重计权，
     同时保持权重尺度不膨胀。

  2) 岗位相似度矩阵：
     把每个岗位表示成「技能 -> 权重」的向量（缺失技能补 0），
     对每个技能列做中心化（减去跨岗位均值），再算行向量之间的余弦相似度。
     中心化余弦 = 皮尔逊相关系数，能把「所有岗位都有的通用技能」的影响去掉，
     让相似度反映技能组合的共性；相关系数 [-1,1] 再线性映射到 [0,1]。

输入:
  onet/job_skills_unified.csv        原始 40 技能（O*NET 官方 IM + 领域技能，静态输入）
  onet/job_skills_specific.csv       新版具体技能（build_specific_skills.py 输出）

输出:
  onet/job_skills_unified_specific.csv   岗位 × 技能，去重取最大权重
  Module_B_SemanticMatch/job_similarity_matrix_specific.csv
    29 岗位两两相似度（406 对，Similarity_Score ∈ [0,1]）

用法:
  python build_unified_specific.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
P = {k: (ROOT / v if not Path(v).is_absolute() else Path(v)) for k, v in CFG["paths"].items()}

# 注意：config.yaml 的 unified_skills 指向的是本脚本的输出文件（最终表），
# 所以这里不用 P["unified_skills"]，而是显式拼出「原始通用技能」的输入路径。
IN_GENERIC = P["onet_dir"] / "job_skills_unified.csv"
IN_SPECIFIC = P["onet_dir"] / "job_skills_specific.csv"
OUT_UNIFIED = P["onet_dir"] / "job_skills_unified_specific.csv"
OUT_SIM_SPECIFIC = ROOT / "Module_B_SemanticMatch" / "job_similarity_matrix_specific.csv"


def main():
    gen = pd.read_csv(IN_GENERIC, encoding="utf-8")
    spec = pd.read_csv(IN_SPECIFIC, encoding="utf-8")

    # 列校验：两张输入表必须至少包含 job / skill / weight
    for c in ("job", "skill", "weight"):
        if c not in gen.columns:
            raise SystemExit(f"[错误] {IN_GENERIC} 缺少列 {c}")
        if c not in spec.columns:
            raise SystemExit(f"[错误] {IN_SPECIFIC} 缺少列 {c}")

    # 若输入缺 source 列则补默认标签（通用表一般已有 onet/domain，具体表有 seed/onet_tool）
    gen["source"] = gen.get("source", "generic")
    spec["source"] = spec.get("source", "specific")

    # ---------- 第一步：合并成统一技能表 ----------
    combined = pd.concat([gen, spec], ignore_index=True)

    # 先把同一 (job, skill) 的所有来源标签聚合成 "a+b" 形式（如 onet_tool+seed），
    # 保留来源可追溯性；注意这里按字母序拼接，与权重无关。
    source_map = (
        combined.groupby(["job", "skill"])["source"]
        .apply(lambda s: "+".join(sorted(set(s))))
        .reset_index(name="source")
    )
    # 同一岗位×技能取最大权重（避免通用与具体技能重名时双重计权）
    # 先按权重降序排序再 keep="first"，等价于 groupby().max() 但保留对应行。
    best = (
        combined.sort_values("weight", ascending=False)
        .drop_duplicates(["job", "skill"], keep="first")[["job", "skill", "weight"]]
    )
    combined = best.merge(source_map, on=["job", "skill"], how="left")
    combined.to_csv(OUT_UNIFIED, index=False, encoding="utf-8")

    # ---------- 第二步：岗位相似度矩阵（中心化余弦 = 皮尔逊相关） ----------
    # 岗位顺序严格按 config.yaml 的 jobs 顺序，保证矩阵行列与展示一致。
    jobs = list(CFG["jobs"])
    job_vecs = {}
    for j in jobs:
        sub = combined[combined["job"] == j]
        job_vecs[j] = dict(zip(sub["skill"], sub["weight"]))

    # 技能全集 = 所有岗位技能名的并集（行=岗位，列=技能；缺失记 0）
    all_skills = sorted({s for v in job_vecs.values() for s in v})
    M = np.array([[job_vecs.get(j, {}).get(s, 0.0) for s in all_skills] for j in jobs], dtype=float)

    # 按列中心化：减去每个技能在 29 个岗位上的均值。
    # 作用：去掉「几乎所有岗位都有的通用技能」带来的虚高相似度，
    # 让相似度主要由技能组合的差异结构决定（等价于皮尔逊相关）。
    Mc = M - M.mean(axis=0, keepdims=True)

    # 行归一化，得到单位向量；零向量（理论上不会出现）用 1.0 兜底避免除零
    norm = np.linalg.norm(Mc, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    sim_r = (Mc / norm) @ (Mc / norm).T

    # 相关系数在 [-1,1]，线性映射到 [0,1] 方便作为相似度使用/与阈值比较
    sim = (sim_r + 1.0) / 2.0

    # 只保留上三角的岗位对（i<j），生成 406 行三元组
    pairs = []
    for i in range(len(jobs)):
        for k in range(i + 1, len(jobs)):
            pairs.append({
                "Job_A": jobs[i],
                "Job_B": jobs[k],
                "Similarity_Score": round(float(sim[i, k]), 4),
            })
    sdf = pd.DataFrame(pairs)
    sdf.to_csv(OUT_SIM_SPECIFIC, index=False, encoding="utf-8")

    # ---------- 摘要输出 ----------
    print(f"[统一技能表] {len(combined)} 行 | 岗位 {len(jobs)} | 技能维度 {len(all_skills)} -> {OUT_UNIFIED}")
    print(f"[相似度矩阵] {len(pairs)} 对 -> {OUT_SIM_SPECIFIC}")
    # 按 config.yaml 的 graph.sim_quantile 计算阈值，并预估建图时会产生多少跳板边
    vals = sdf["Similarity_Score"].to_numpy()
    q = float(CFG["graph"]["sim_quantile"])
    th = float(np.quantile(vals, q))
    print(f"相似度范围 {vals.min():.3f} ~ {vals.max():.3f} | 中位 {np.median(vals):.3f}")
    print(f"分位 q={q:.2f} -> 阈值 {th:.4f}，跳板边约 {int((vals >= th).sum())} 对")

    # 合理性抽查：这些岗位对「应该」相似度高；若明显偏低，多半是输入数据或匹配出了问题
    print("\n应偏高的岗位对:")
    for a, b in [("Software Developer", "Computer Programmer"),
                 ("Data Scientist", "Statistician"),
                 ("Business Intelligence Analyst", "Data Warehousing Specialist"),
                 ("HR Specialist", "HR Manager"),
                 ("Database Architect", "Data Warehousing Specialist")]:
        row = sdf[((sdf.Job_A == a) & (sdf.Job_B == b)) | ((sdf.Job_A == b) & (sdf.Job_B == a))]
        print(f"  {a:26s} ~ {b:26s} = {float(row['Similarity_Score'].iloc[0]):.3f}")
    # 合理性抽查：这些岗位对「应该」相似度低（跨职能），用于防止相似度整体虚高
    print("\n应偏低的岗位对:")
    for a, b in [("HR Specialist", "Software Developer"),
                 ("Marketing Manager", "Database Architect"),
                 ("Training Specialist", "Operations Research Analyst")]:
        row = sdf[((sdf.Job_A == a) & (sdf.Job_B == b)) | ((sdf.Job_A == b) & (sdf.Job_B == a))]
        print(f"  {a:26s} ~ {b:26s} = {float(row['Similarity_Score'].iloc[0]):.3f}")
    print("\n[下一步] python path_search.py --export path_features_specific.csv")


if __name__ == "__main__":
    main()
