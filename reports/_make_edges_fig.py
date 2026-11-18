# -*- coding: utf-8 -*-
"""生成 path_search.py 图结构（3 类节点 × 6 类边）示意图，输出 reports/graph_edge_types.png。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(-0.8, 6.2)
ax.set_ylim(-0.6, 4.4)
ax.axis("off")

C_MAJOR = "#dbeafe"
C_SKILL = "#dcfce7"
C_JOB = "#ffedd5"
E_OWN = "#6b7280"
E_REQ = "#2563eb"
E_TOJOB = "#9333ea"
E_JUMP = "#ea580c"
E_COOC = "#a1a1aa"


def major(x, y, t):
    ax.add_patch(FancyBboxPatch((x - 0.55, y - 0.22), 1.1, 0.44,
                                boxstyle="round,pad=0.06", fc=C_MAJOR, ec="#3b82f6", lw=1.5))
    ax.text(x, y, t, ha="center", va="center", fontsize=11)


def skill(x, y, t):
    ax.add_patch(plt.Circle((x, y), 0.17, fc=C_SKILL, ec="#16a34a", lw=1.5))
    ax.text(x, y, t, ha="center", va="center", fontsize=9)


def job(x, y, t):
    ax.add_patch(FancyBboxPatch((x - 0.55, y - 0.22), 1.1, 0.44,
                                boxstyle="round,pad=0.06", fc=C_JOB, ec="#f97316", lw=1.5))
    ax.text(x, y, t, ha="center", va="center", fontsize=10)


major(0.2, 3.4, "专业 A")
major(0.2, 1.0, "专业 B")
skill(2.4, 3.9, "技能1")
skill(2.4, 3.0, "技能2")
skill(2.4, 2.1, "技能3")
skill(2.4, 1.2, "技能4")
job(5.0, 3.0, "岗位 M")
job(5.0, 1.2, "岗位 N")

# major_owns
for (x1, y1), (x2, y2) in [((0.75, 3.4), (2.23, 3.9)), ((0.75, 3.4), (2.23, 3.0)),
                            ((0.75, 1.0), (2.23, 2.1)), ((0.75, 1.0), (2.23, 1.2))]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", color=E_OWN, lw=1.6))

# job_requires（技能->岗位）与 skill_leads_to（岗位->技能）成对偏移画
for mid, (sx, sy), (jx, jy) in [("技能2", (2.4, 3.0), (5.0, 3.0)),
                                 ("技能3", (2.4, 2.1), (5.0, 3.0)),
                                 ("技能4", (2.4, 1.2), (5.0, 1.2))]:
    ax.annotate("", xy=(jx - 0.58, jy), xytext=(sx + 0.2, sy),
                arrowprops=dict(arrowstyle="-|>", color=E_REQ, lw=2.0,
                                connectionstyle="arc3,rad=-0.06"))
    ax.annotate("", xy=(sx + 0.2, sy), xytext=(jx - 0.58, jy),
                arrowprops=dict(arrowstyle="-|>", color=E_REQ, lw=1.2, ls=(0, (4, 2)),
                                connectionstyle="arc3,rad=0.06"))

# major_to_job
ax.annotate("", xy=(4.4, 2.75), xytext=(0.8, 3.3),
            arrowprops=dict(arrowstyle="-|>", color=E_TOJOB, lw=1.8, ls="--",
                            connectionstyle="arc3,rad=-0.15"))

# job_to_job
ax.annotate("", xy=(5.0, 1.45), xytext=(5.0, 2.75),
            arrowprops=dict(arrowstyle="<|-|>", color=E_JUMP, lw=3.0))

# skill_cooccurs
for (x1, y1), (x2, y2) in [((2.4, 3.9), (2.4, 3.0)), ((2.4, 3.0), (2.4, 2.1)), ((2.4, 2.1), (2.4, 1.2))]:
    ax.plot([x1, x2], [y1, y2], color=E_COOC, lw=1.6, ls=(0, (1, 2)))

ax.text(3.35, 3.15, "job_requires（带权重）\nskill_leads_to（镜像）", fontsize=9, color=E_REQ, ha="center")
ax.text(2.25, 2.75, "major_to_job\n（重合≥3 直连）", fontsize=9, color=E_TOJOB, ha="center")
ax.text(5.25, 2.1, "job_to_job\n（相似度≥阈值，跳板）", fontsize=9, color=E_JUMP, ha="left")
ax.text(2.62, 2.55, "skill_cooccurs\n（共现，保证连通）", fontsize=9, color="#71717a", ha="left")

legend = [
    Line2D([0], [0], color=E_OWN, lw=1.6, label="major_owns  专业—技能：专业拥有该技能"),
    Line2D([0], [0], color=E_REQ, lw=2.0, label="job_requires  技能→岗位：岗位要求该技能（带 O*NET 权重）"),
    Line2D([0], [0], color=E_REQ, lw=1.2, ls=(0, (4, 2)), label="skill_leads_to  岗位→技能：同一条边的反向条目"),
    Line2D([0], [0], color=E_TOJOB, lw=1.8, ls="--", label="major_to_job  专业→岗位：技能重合≥3 且排前 K 的直连"),
    Line2D([0], [0], color=E_JUMP, lw=3.0, label="job_to_job  岗位–岗位：相似度≥分位阈值(0.6467)的跳板"),
    Line2D([0], [0], color=E_COOC, lw=1.6, ls=(0, (1, 2)), label="skill_cooccurs  技能–技能：同一岗位内共现，权重=共现岗位数"),
]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=9.5,
          frameon=True, ncol=1, title="六类边（path_search.py build_graph）", title_fontsize=10)

ax.set_title("path_search.py 图结构示意：3 类节点（专业/技能/岗位）× 6 类边", fontsize=14, pad=14)

out = Path(__file__).resolve().parent / "graph_edge_types.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved:", out)
