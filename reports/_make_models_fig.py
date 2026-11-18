# -*- coding: utf-8 -*-
"""生成 models/ 文件夹两分类示意图，输出 reports/models_folder.png。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 7.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.6)
ax.axis("off")

C_REQ = ("#dcfce7", "#16a34a")    # 必需：语义模型
C_AUX = ("#ffedd5", "#f97316")    # 辅助：XGB
C_DIR = ("#f3e8ff", "#9333ea")    # 文件夹
C_FLOW = ("#dbeafe", "#3b82f6")   # 消费方


def box(x, y, w, h, text, colors, fs=10, bold=False, dashed=False, align="center"):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.6,
                                ls="--" if dashed else "-"))
    ax.text(x, y, text, ha=align, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.6)
    return x, y, w, h


def arrow(x1, y1, x2, y2, color="#52525b", lw=1.8, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                ls="--" if dashed else "-"))


# 根节点
box(6.5, 6.9, 2.6, 0.7, "models/", C_DIR, fs=14, bold=True)

# 左：语义模型
box(3.3, 5.35, 4.9, 0.62, "A. bge-small-zh-v1.5/（约 92 MB，第三方开源，未训练）", C_REQ, fs=10.5, bold=True)
box(3.3, 2.85, 4.9, 4.1,
    "• 内容：model.safetensors 91.4 MB + 分词器/config\n"
    "• 结构：4 层 Transformer，隐藏维 512，词表 21,128\n"
    "• 流水线：Transformer → Pooling → Normalize（输出已归一化）\n"
    "• 用途：唯一消费方 path_search.py::_semantic_score\n"
    "   计算「专业名 → 岗位名」余弦 = semantic_score 特征\n"
    "• 离线加载（local_files_only=True），不联网\n"
    "• 加载失败 → semantic_score 统一回退 0.5（中性值）\n"
    "   管线不中断，排序退化为 长度 + 缺口 驱动\n"
    "• 换模型只需改 config.yaml 的 paths.model_dir\n"
    "• 删除它：路径搜索仍可运行，但语义分数全为 0.5",
    C_REQ, fs=9.2, align="left")

# 右：XGB 模型
box(9.9, 5.35, 5.2, 0.62, "B. xgb_*.json（3 个，合计约 2.2 MB，我们训练）", C_AUX, fs=10.5, bold=True)
box(9.9, 2.85, 5.2, 4.1,
    "• xgb_salary.json（配对级薪资回归，1.09 MB）\n"
    "• xgb_high_salary.json（配对级高薪分类，0.53 MB）\n"
    "• xgb_salary_job.json（岗位级回归，0.57 MB）\n"
    "• 格式：XGBoost 原生 JSON，load_model 直接读入\n"
    "• 用途：仅 SHAP 归因与复现分析\n"
    "   ——不参与推荐链路（recommend 只读路径特征 CSV）\n"
    "• 可随时删除并重建：python train_xgb_shap.py\n"
    "   （random_seed=42，结果可复现）\n"
    "• 与代码绑定：按 train_xgb_shap.py 的 FEATURES 顺序训练，\n"
    "   增删/改序特征后必须重训，否则预测错位\n"
    "• 文件大小差异：salary 300 棵树全用满；high_salary 更小\n"
    "   （path_length 从未参与分裂）",
    C_AUX, fs=9.2, align="left")

# 消费方
box(3.3, 0.75, 3.6, 0.62, "path_search.py\n（主流程，必需）", C_FLOW, fs=10)
box(9.9, 0.75, 3.6, 0.62, "reports/ SHAP 图与指标\n（答辩/分析用）", C_FLOW, fs=10)

# 连线
arrow(5.8, 6.55, 3.9, 5.66, color="#9333ea", lw=2.0)
arrow(7.2, 6.55, 9.3, 5.66, color="#9333ea", lw=2.0)
arrow(3.3, 4.85, 3.3, 1.06, color="#16a34a", lw=2.0)
arrow(9.9, 4.85, 9.9, 1.06, color="#f97316", lw=2.0)

ax.text(6.5, 0.15, "A 是管线必需品（影响 semantic_score）；B 是分析工具（可删可重建，不在推荐链路）",
        ha="center", fontsize=10, style="italic", color="#52525b")

ax.set_title("models/ 文件夹两分类：语义模型（必需） vs XGBoost 解释模型（辅助）", fontsize=14, pad=12)

out = Path(__file__).resolve().parent / "models_folder.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved:", out)
