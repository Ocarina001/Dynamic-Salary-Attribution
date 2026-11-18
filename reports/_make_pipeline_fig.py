# -*- coding: utf-8 -*-
"""生成 Module A 管线示意图（三脚本数据流），输出 reports/pipeline_module_a.png。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.2)
ax.axis("off")

C_IN = ("#dbeafe", "#3b82f6")       # 输入数据
C_SCRIPT = ("#dcfce7", "#16a34a")   # 脚本
C_OUT = ("#ffedd5", "#f97316")      # 输出文件
C_CFG = ("#f3e8ff", "#9333ea")      # 配置/模型
C_DOWN = ("#f4f4f5", "#6b7280")     # 下游


def box(x, y, w, h, text, colors, fs=10, dashed=False, bold=False):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.6,
                                ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2, color="#52525b", lw=1.8, dashed=False, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                ls="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}"))


# ---------------- 顶部配置条 ----------------
box(4.3, 7.7, 7.2, 0.62,
    "config.yaml（路径/岗位/权重/图参数，唯一配置源）  +  config/specific_skills.yaml（140 技能词表 + 岗位种子）",
    C_CFG, fs=10.5)

# ---------------- 输入层 ----------------
box(1.5, 6.2, 2.6, 1.0,
    "O*NET 原始数据\nsoftware_skills.csv（31,821 行工具证据）\noccupation_data.csv（职业名→SOC）",
    C_IN, fs=9)
box(1.5, 4.8, 2.6, 0.85,
    "44 份培养方案文本\nonet/bylu_curricula/*.txt\n(+ major_skills_llm.csv 背景技能)",
    C_IN, fs=9)
box(1.5, 3.4, 2.6, 0.6, "BLS 2024 薪资\nus_wages_by_occupation_national_2024.csv", C_IN, fs=9)

# ---------------- 三个脚本 ----------------
box(4.6, 6.2, 2.3, 0.72, "① build_specific_skills.py\n双侧技能匹配", C_SCRIPT, fs=10.5, bold=True)
box(8.1, 6.2, 2.3, 0.72, "② build_unified_specific.py\n合并 + 岗位相似度", C_SCRIPT, fs=10.5, bold=True)
box(11.6, 6.2, 2.1, 0.72, "③ path_search.py\n建图 + 寻路 + 打分", C_SCRIPT, fs=10.5, bold=True)

# ---------------- ① 的输出 ----------------
box(4.6, 4.55, 2.7, 1.15,
    "job_skills_specific.csv（774 行）\nmajor_skills_specific.csv（1,017 行）\nmajor_skill_map_specific.txt\njob_tool_evidence_specific.csv",
    C_OUT, fs=8.8)

# ---------------- ② 的输出 ----------------
box(8.1, 4.55, 2.5, 0.95,
    "job_skills_unified_specific.csv\n（1,863 行 / 145 维）\njob_similarity_matrix_specific.csv\n（406 对，阈值取 90 分位）",
    C_OUT, fs=8.8)

# ---------------- 语义模型 ----------------
box(9.9, 2.95, 1.9, 0.55, "models/bge-small-zh-v1.5\n（本地语义模型）", C_CFG, fs=9, dashed=True)

# ---------------- ③ 的输出 ----------------
box(11.6, 4.55, 2.2, 0.8,
    "path_features_specific.csv\n（44×29 = 1,276 行 × 12 列）",
    C_OUT, fs=9)

# ---------------- 下游 ----------------
box(5.6, 2.95, 2.3, 0.85,
    "recommend_major_jobs.py\n→ major_job_specific_top5.csv\n→ major_job_specific_summary.csv", C_DOWN, fs=8.8)
box(8.7, 1.5, 2.4, 0.95,
    "build_job_salary.py\n输入：path_features + BLS\n→ job_salary.csv（29/29）\n→ job_features_salary.csv", C_DOWN, fs=8.6)
box(11.9, 2.95, 2.0, 0.8,
    "XGBoost + SHAP（计划中）\n用 12 列路径特征训练\n/ 可解释性分析", C_DOWN, fs=8.8, dashed=True)

# ---------------- 连线 ----------------
arrow(2.8, 6.2, 3.45, 6.2, color="#2563eb", lw=2.2)          # O*NET -> ①
arrow(2.8, 4.8, 3.45, 5.9, color="#2563eb", lw=2.2)          # 培养方案 -> ①
arrow(4.6, 5.84, 4.6, 5.15, color="#16a34a", lw=2.0)         # ① -> 其输出
arrow(5.95, 4.55, 7.5, 5.9, color="#f97316", lw=2.0)         # ①输出 -> ②
arrow(8.1, 5.84, 8.1, 5.05, color="#16a34a", lw=2.0)         # ② -> 其输出
arrow(9.25, 6.2, 10.55, 6.2, color="#16a34a", lw=2.0)        # ② -> ③
arrow(9.4, 4.95, 11.05, 5.82, color="#f97316", lw=1.8)       # ②输出(相似度) -> ③
arrow(11.6, 5.84, 11.6, 4.98, color="#16a34a", lw=2.0)       # ③ -> 其输出
arrow(10.55, 3.15, 10.95, 5.8, color="#9333ea", lw=1.6, dashed=True)  # bge -> ③
arrow(11.0, 4.13, 6.75, 3.4, color="#f97316", lw=1.8)        # 特征表 -> 推荐
arrow(1.5, 3.08, 7.55, 1.45, color="#2563eb", lw=1.6, rad=0.18)  # BLS -> 薪资脚本（下绕）
arrow(12.2, 4.13, 12.05, 3.38, color="#f97316", lw=1.8)      # 特征表 -> XGBoost

# 注释文字
ax.text(6.5, 5.05, "合并", fontsize=9, color="#52525b")
ax.text(7.6, 1.05, "路径特征表是下游各部门的唯一输入", fontsize=9,
        color="#52525b", style="italic", ha="center")

ax.set_title("Module A 管线：三脚本数据流（输入 → ① 技能生成 → ② 合并/相似度 → ③ 图搜索 → 下游）",
             fontsize=14, pad=12)

out = Path(__file__).resolve().parent / "pipeline_module_a.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved:", out)
