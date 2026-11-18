# -*- coding: utf-8 -*-
"""
app.py —— Streamlit 单页前端
============================

运行：streamlit run app.py

页面结构：
  侧边栏：专业 + 岗位 筛选器
  主区域：
    Tab 1  路径对比  —— 直达 vs 推荐 vs 搜索最优
    Tab 2  薪资归因  —— SHAP 瀑布图 + 特征贡献
    Tab 3  跨切片对比 —— 同岗位跨专业 / 同专业跨岗位
    Tab 4  反事实干预 —— slider 实时改特征看薪资变化
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import yaml

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

FEATURES = [
    "path_length", "skill_gap", "semantic_score", "baseline_gap",
    "baseline_best_gap", "gap_reduction", "gap_reduction_vs_best", "n_paths",
]
FEATURE_LABELS = {
    "path_length": "路径长度",
    "skill_gap": "技能缺口 (当前路径)",
    "semantic_score": "专业-岗位语义匹配",
    "baseline_gap": "直接连缺口 (无跳板)",
    "baseline_best_gap": "搜索空间最优缺口",
    "gap_reduction": "跳板带来的缺口改善",
    "gap_reduction_vs_best": "当前路径 vs 搜索最优",
    "n_paths": "候选路径总数",
}

st.set_page_config(
    page_title="专业→岗位 路径规划 & 薪资归因",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 缓存数据 & 模型（只加载一次）
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    pf = pd.read_csv(ROOT / "path_features_specific.csv", encoding="utf-8")
    sal = pd.read_csv(ROOT / "job_salary.csv", encoding="utf-8")
    shap_all = pd.read_csv(
        ROOT / "reports" / "counterfactual" / "pair_shap_all.csv",
        encoding="utf-8-sig",
    )
    job_cf = pd.read_csv(
        ROOT / "reports" / "counterfactual" / "same_job_counterfactual.csv",
        encoding="utf-8-sig",
    )
    major_cf = pd.read_csv(
        ROOT / "reports" / "counterfactual" / "same_major_counterfactual.csv",
        encoding="utf-8-sig",
    )
    merged = pf.merge(
        sal[["onet_job", "median_wage", "mean_wage"]].drop_duplicates("onet_job"),
        left_on="target_job", right_on="onet_job", how="left",
    )
    merged = merged.dropna(subset=["median_wage"])
    majors = sorted(merged["source_major"].unique())
    jobs = sorted(merged["target_job"].unique())
    return merged, shap_all, job_cf, major_cf, majors, jobs


@st.cache_resource
def load_model():
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(ROOT / "models" / "xgb_salary.json"))
    import shap
    explainer = shap.TreeExplainer(model)
    ev = np.atleast_1d(explainer.expected_value)[0]
    expected = float(ev)
    return model, explainer, expected


# ──────────────────────────────────────────────
# 全局样式 —— 主题感知（亮色/暗色自适应）
# ──────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary-color: #1e3a8a;
        --primary-light: #3b82f6;
        --secondary-color: #0083B8;
        --accent-color: #00B0B9;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #f1f5f9;
        --text-primary: #374151;
        --text-secondary: #4b5563;
        --text-muted: #6b7280;
        --border-color: #e5e7eb;
        --shadow-light: rgba(0, 0, 0, 0.05);
        --shadow-medium: rgba(0, 0, 0, 0.1);
        --shadow-strong: rgba(0, 0, 0, 0.08);
    }
    [data-theme="dark"] {
        --primary-color: #60a5fa;
        --primary-light: #93c5fd;
        --secondary-color: #38bdf8;
        --accent-color: #06b6d4;
        --success-color: #34d399;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --bg-primary: #1f2937;
        --bg-secondary: #111827;
        --bg-tertiary: #374151;
        --text-primary: #f9fafb;
        --text-secondary: #e5e7eb;
        --text-muted: #9ca3af;
        --border-color: #4b5563;
        --shadow-light: rgba(0, 0, 0, 0.3);
        --shadow-medium: rgba(0, 0, 0, 0.4);
        --shadow-strong: rgba(0, 0, 0, 0.5);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #60a5fa;
            --primary-light: #93c5fd;
            --secondary-color: #38bdf8;
            --accent-color: #06b6d4;
            --success-color: #34d399;
            --warning-color: #fbbf24;
            --error-color: #f87171;
            --bg-primary: #1f2937;
            --bg-secondary: #111827;
            --bg-tertiary: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #e5e7eb;
            --text-muted: #9ca3af;
            --border-color: #4b5563;
            --shadow-light: rgba(0, 0, 0, 0.3);
            --shadow-medium: rgba(0, 0, 0, 0.4);
            --shadow-strong: rgba(0, 0, 0, 0.5);
        }
    }

    body { font-family: 'Arial', sans-serif; color: var(--text-primary);
           background-color: var(--bg-secondary);
           transition: background-color 0.3s ease, color 0.3s ease; }
    .stApp { background-color: var(--bg-secondary); transition: background-color 0.3s ease; }

    .main-header {
        font-size: 2.4rem; color: var(--primary-color);
        margin-bottom: 0.3rem; font-weight: 800;
        letter-spacing: -0.5px;
        transition: color 0.3s ease;
    }
    .sub-header {
        font-size: 1.4rem; color: var(--primary-color);
        margin-top: 1.5rem; margin-bottom: 1rem; font-weight: 700;
        border-bottom: 2px solid var(--border-color); padding-bottom: 6px;
        transition: color 0.3s ease, border-color 0.3s ease;
    }

    .insight-box {
        background-color: var(--bg-primary);
        border-radius: 12px; padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px var(--shadow-light), 0 1px 3px var(--shadow-medium);
        border: 1px solid var(--border-color);
        transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-container {
        background-color: var(--bg-primary);
        border-radius: 10px;
        box-shadow: 0 4px 12px var(--shadow-strong);
        padding: 20px; margin-bottom: 15px;
        border-left: 4px solid var(--primary-color);
        transition: transform 0.2s ease-in-out, background-color 0.3s ease,
                    border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px var(--shadow-strong);
    }

    .big-number { font-size: 2.2rem; font-weight: 700; line-height: 1.1; color: var(--primary-color); }
    .big-number-blue   { color: #1565c0; }
    .big-number-green  { color: #2e7d32; }
    .big-number-orange { color: #e65100; }
    .big-number-red    { color: #c62828; }

    .chart-container {
        background-color: var(--bg-primary);
        border-radius: 12px; padding: 20px; margin-bottom: 24px;
        box-shadow: 0 4px 6px var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }

    .path-node {
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        font-size: 0.9rem; font-weight: 500; margin: 2px;
        transition: transform 0.15s ease;
    }
    .path-node:hover { transform: scale(1.05); }
    .node-major   { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
    .node-job     { background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
    .node-skill   { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
    .node-arrow   { color: var(--text-muted); font-size: 1.1rem; padding: 0 4px; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-tertiary) !important;
        border-radius: 6px 6px 0 0;
        padding: 10px 16px;
        border: 1px solid var(--border-color) !important;
        border-bottom: none !important;
        color: var(--text-secondary) !important;
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    .stSelectbox > div > div, .stMultiSelect > div > div,
    .stTextInput > div > div > input {
        background-color: var(--bg-primary) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }

    .positive-shap { color: var(--success-color); font-weight: 600; }
    .negative-shap { color: var(--error-color); font-weight: 600; }

    footer { visibility: hidden; }

    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .sub-header  { font-size: 1.2rem; }
    }
</style>

<script>
(function() {
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme-preference', theme);
    }
    function getSystemTheme() {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    function initTheme() {
        const saved = localStorage.getItem('theme-preference');
        applyTheme(saved || getSystemTheme());
    }
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        if (!localStorage.getItem('theme-preference')) applyTheme(e.matches ? 'dark' : 'light');
    });
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initTheme);
    else initTheme();
    new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.type === 'childList' && m.addedNodes.length > 0) initTheme();
        });
    }).observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────
def fmt_dollar(v):
    if pd.isna(v):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.0f}"


def fmt_gap(v):
    if pd.isna(v):
        return "—"
    return f"{v:.4f}"


def parse_path(nodes_str):
    parts = nodes_str.split(" -> ")
    result = []
    for p in parts:
        if p.startswith("专业:"):
            result.append(("major", p[3:]))
        elif p.startswith("岗位:"):
            result.append(("job", p[3:]))
        elif p.startswith("技能:"):
            result.append(("skill", p[3:]))
        else:
            result.append(("other", p))
    return result


def render_path_nodes(nodes):
    html = '<div style="margin: 8px 0; display: flex; flex-wrap: wrap; align-items: center;">'
    for i, (kind, label) in enumerate(nodes):
        cls = {"major": "node-major", "job": "node-job", "skill": "node-skill"}.get(kind, "node-major")
        icon = {"major": "🎓", "job": "💼", "skill": "🔧"}.get(kind, "•")
        html += f'<span class="path-node {cls}">{icon} {label}</span>'
        if i < len(nodes) - 1:
            html += '<span class="node-arrow">➜</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def cost_breakdown(row):
    base_cost = 1.0 * (row["path_length"] / 4)
    gap_cost = 3.0 * row["skill_gap"]
    sem_cost = 1.0 * (1 - row["semantic_score"])
    return base_cost, gap_cost, sem_cost


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────
def main():
    merged, shap_all, job_cf, major_cf, majors, jobs = load_data()
    model, explainer, expected = load_model()

    st.markdown('<h1 class="main-header">🎯 专业 → 岗位 路径规划与薪资归因</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        <p style="font-size: 1.05rem; margin-bottom: 12px; color: var(--primary-color); font-weight: 600;">
            异构知识图谱路径搜索 · XGBoost 薪资预测 · SHAP 可解释归因
        </p>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div style="text-align: center; padding: 6px 12px; min-width: 160px;">
                <div style="font-size: 1.6rem; font-weight: 700; color: var(--primary-color);">🗺️</div>
                <p style="font-weight: 600; margin: 2px 0; color: var(--text-primary);">路径搜索</p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">238 节点异构知识图谱</p>
            </div>
            <div style="text-align: center; padding: 6px 12px; min-width: 160px;">
                <div style="font-size: 1.6rem; font-weight: 700; color: var(--primary-color);">💻</div>
                <p style="font-weight: 600; margin: 2px 0; color: var(--text-primary);">薪资预测</p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">XGBoost 8 特征模型</p>
            </div>
            <div style="text-align: center; padding: 6px 12px; min-width: 160px;">
                <div style="font-size: 1.6rem; font-weight: 700; color: var(--primary-color);">🔍</div>
                <p style="font-weight: 600; margin: 2px 0; color: var(--text-primary);">SHAP 归因</p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">可解释薪资贡献拆解</p>
            </div>
            <div style="text-align: center; padding: 6px 12px; min-width: 160px;">
                <div style="font-size: 1.6rem; font-weight: 700; color: var(--primary-color);">🎛️</div>
                <p style="font-weight: 600; margin: 2px 0; color: var(--text-primary);">反事实干预</p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">Slider 实时探索薪资变化</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 侧边栏 ──
    with st.sidebar:
        st.header("🔍 筛选器")

        major_default = st.selectbox(
            "选择专业", majors,
            index=majors.index("计算机科学与技术") if "计算机科学与技术" in majors else 0,
        )
        job_default = st.selectbox(
            "选择目标岗位", jobs,
            index=jobs.index("Data Scientist") if "Data Scientist" in jobs else 0,
        )

        st.divider()
        st.subheader("⚡ 快速预设")
        presets = {
            "🤖 AI 方向": ("计算机科学与技术", "Data Scientist"),
            "💼 高薪管理": ("法语", "Financial Manager"),
            "📊 数据分析": ("经济学", "Business Intelligence Analyst"),
            "🌐 语言+科技": ("英语", "Software Developer"),
        }
        cols = st.columns(2)
        for i, (name, (pm, pj)) in enumerate(presets.items()):
            col = cols[i % 2]
            if col.button(name, use_container_width=True, key=f"preset_{i}"):
                st.session_state["major_override"] = pm
                st.session_state["job_override"] = pj

        st.divider()
        st.subheader("📖 系统说明")
        st.markdown("""
        - **路径搜索**：在 238 节点异构图上枚举候选路径
        - **薪资预测**：基于 8 个路径特征的 XGBoost 模型（R² ≈ 0.78）
        - **归因方式**：SHAP 值分解 + 跨切片对比
        - ⚠️ 模型内预测，存在选择偏差，归因是**方向级**的
        """)

    # override via preset
    sel_major = st.session_state.get("major_override", major_default)
    sel_job = st.session_state.get("job_override", job_default)

    pair = merged[(merged["source_major"] == sel_major) &
                  (merged["target_job"] == sel_job)]
    if pair.empty:
        st.error(f"找不到配对：{sel_major} → {sel_job}")
        st.stop()
    pair = pair.iloc[0]
    shap_row = shap_all[(shap_all["source_major"] == sel_major) &
                        (shap_all["target_job"] == sel_job)]
    if shap_row.empty:
        st.error("SHAP 分解数据缺失，请先运行 counterfactual_attribution.py")
        st.stop()
    shap_row = shap_row.iloc[0]

    # ── 顶部核心指标卡片 ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''
        <div class="metric-container" style="border-left-color: var(--primary-color);">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 1.8rem; margin-right: 12px;">🎓</div>
                <div>
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 3px;">专业 → 岗位</p>
                    <p style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin: 0;">{sel_major}</p>
                    <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 2px 0 0 0;">💼 {sel_job}</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="metric-container" style="border-left-color: #1565c0;">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 1.8rem; margin-right: 12px;">💵</div>
                <div>
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 3px;">真实中位工资 (BLS 2024)</p>
                    <p class="big-number big-number-blue" style="margin: 0;">${pair.median_wage:,.0f}</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with c3:
        pred_wage = shap_row["pred_wage"]
        diff = pred_wage - pair.median_wage
        diff_color = "#2e7d32" if abs(diff) < 5000 else ("#e65100" if diff > 0 else "#c62828")
        st.markdown(f'''
        <div class="metric-container" style="border-left-color: {diff_color};">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 1.8rem; margin-right: 12px;">🤖</div>
                <div>
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 3px;">模型预测薪资</p>
                    <p class="big-number" style="color:{diff_color}; margin: 0;">${pred_wage:,.0f}</p>
                    <p style="font-size: 0.8rem; color:{diff_color}; margin: 2px 0 0 0;">偏差 {fmt_dollar(diff)}</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with c4:
        sg = pair.skill_gap
        bg = pair.baseline_gap
        reduction = bg - sg
        border_color = "#2e7d32" if reduction > 0 else "#666"
        st.markdown(f'''
        <div class="metric-container" style="border-left-color: {border_color};">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 1.8rem; margin-right: 12px;">🔧</div>
                <div>
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 3px;">技能缺口</p>
                    <p class="big-number" style="color:{border_color}; margin: 0;">{sg:.4f}</p>
                    <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 2px 0 0 0;">
                        直接连 {bg:.4f} → 改善 {reduction:+.4f}
                    </p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # ── Tabs ──
    tab_path, tab_shap, tab_compare, tab_intervene = st.tabs([
        "🏆 路径对比", "💰 薪资归因", "📊 跨切片对比", "🎛️ 反事实干预"
    ])

    # ═══════════════════════════════════════════════════════
    # Tab 1: 路径对比
    # ═══════════════════════════════════════════════════════
    with tab_path:
        st.subheader("三种路径策略对比")
        st.caption("同一专业→同一岗位，不同路径策略的 cost 拆解与缺口")

        col_a, col_b, col_c = st.columns(3)

        # 路径 1：直达
        with col_a:
            st.markdown("#### ① 直达路径（无跳板）")
            st.markdown('<span class="path-node node-major">🎓 ' + sel_major +
                        '</span><span class="node-arrow">➜</span>' +
                        '<span class="path-node node-job">💼 ' + sel_job + '</span>',
                        unsafe_allow_html=True)

            bl, bgap_val, semv = cost_breakdown(pd.Series({
                "path_length": 2, "skill_gap": pair.baseline_gap,
                "semantic_score": pair.semantic_score,
            }))
            total = bl + bgap_val + semv
            st.metric("cost（代价函数）", f"{total:.4f}")
            st.metric("技能缺口", fmt_gap(pair.baseline_gap))
            st.metric("路径长度", "2 跳")
            st.caption(f"gap 占 cost 的 {bgap_val/total*100:.0f}%")

        # 路径 2：推荐
        with col_b:
            st.markdown("#### ② 推荐路径（最优） ⭐")
            nodes = parse_path(pair.path_nodes)
            render_path_nodes(nodes)

            pl, gapv, semv = cost_breakdown(pair)
            total_rec = pl + gapv + semv
            st.metric("cost（代价函数）", f"{total_rec:.4f}",
                      delta=f"{total - total_rec:.4f} 改善" if total_rec < total else None)
            st.metric("技能缺口", fmt_gap(pair.skill_gap),
                      delta=f"{pair.baseline_gap - pair.skill_gap:.4f}")
            st.metric("路径长度", f"{pair.path_length} 跳")
            st.caption(f"gap 占 cost 的 {gapv/total_rec*100:.0f}%")

        # 路径 3：搜索最优
        with col_c:
            st.markdown("#### ③ 搜索最优缺口（理论下限）")
            st.markdown("候选路径空间中找到的最小 gap，"
                        "可能与推荐路径重合（gap_reduction_vs_best=0）。")
            st.metric("搜索最优缺口", fmt_gap(pair.baseline_best_gap))
            st.metric("候选路径总数", f"{int(pair.n_paths)}")
            st.metric("gap_reduction_vs_best", fmt_gap(pair.gap_reduction_vs_best),
                      help="推荐路径与搜索最优的 gap 差距，0 表示重合")

        # cost 拆解条形图
        st.subheader("Cost 拆解对比")
        import plotly.graph_objects as go

        strategies = ["① 直达路径", "② 推荐路径", "③ 搜索最优"]
        labels_short = ["路径长度代价", "技能缺口代价", "语义匹配代价"]

        def strat_cost(sl, sg, ss):
            return [1.0*(sl/4), 3.0*sg, 1.0*(1-ss)]

        data = np.array([
            strat_cost(2, pair.baseline_gap, pair.semantic_score),
            strat_cost(pair.path_length, pair.skill_gap, pair.semantic_score),
            strat_cost(pair.path_length, pair.baseline_best_gap, pair.semantic_score),
        ])

        fig = go.Figure()
        colors_bars = ["#90caf9", "#ffb74d", "#a5d6a7"]
        for i, (lbl, col) in enumerate(zip(labels_short, colors_bars)):
            fig.add_trace(go.Bar(
                name=lbl, x=strategies, y=data[:, i],
                marker_color=col, text=data[:, i].round(3).astype(str),
                textposition="auto",
            ))
        fig.update_layout(
            barmode="stack", title="Cost 拆解（1.0 × L/4 + 3.0 × gap + 1.0 × (1−semantic)）",
            xaxis_title="路径策略", yaxis_title="Cost 贡献",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        # gap 对比
        st.subheader("技能缺口对比")
        gaps_data = go.Figure()
        gaps_data.add_trace(go.Bar(
            x=["直达", "推荐", "搜索最优"],
            y=[pair.baseline_gap, pair.skill_gap, pair.baseline_best_gap],
            text=[f"{v:.4f}" for v in [pair.baseline_gap, pair.skill_gap, pair.baseline_best_gap]],
            textposition="auto",
            marker_color=["#ef9a9a", "#81c784", "#64b5f6"],
            showlegend=False,
        ))
        gaps_data.update_layout(
            title="三种策略的技能缺口",
            yaxis_title="skill_gap", height=280,
        )
        st.plotly_chart(gaps_data, use_container_width=True)

        # 节点链路展示
        st.subheader("推荐路径完整链路")
        render_path_nodes(parse_path(pair.path_nodes))
        raw_nodes = pair.path_nodes.split(" -> ")
        st.write(f"共 {len(raw_nodes)} 个节点 | path_length = {pair.path_length} | "
                 f"候选路径 {int(pair.n_paths)} 条")

    # ═══════════════════════════════════════════════════════
    # Tab 2: SHAP 归因
    # ═══════════════════════════════════════════════════════
    with tab_shap:
        st.subheader("薪资归因：SHAP 分解")
        st.caption("每个特征对薪资预测的贡献——绿色↑拉高预测，红色↓拉低预测")

        shap_vals = [shap_row[f"shap_{f}"] for f in FEATURES]
        feat_vals = [shap_row[f"feat_{f}"] for f in FEATURES]

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            # 手绘瀑布图
            waterfall = go.Figure()
            cum = expected
            waterfall.add_trace(go.Waterfall(
                x=["基线 (expected)"] + [f"{FEATURE_LABELS[f]}\n{feat_vals[i]:.3f}"
                                         for i, f in enumerate(FEATURES)] + ["预测值"],
                y=[expected] + shap_vals + [pred_wage - expected - sum(shap_vals)],
                measure=["absolute"] + ["relative"] * len(FEATURES) + ["total"],
                decreasing=dict(marker=dict(color="#e74c3c")),
                increasing=dict(marker=dict(color="#2ecc71")),
                totals=dict(marker=dict(color="steelblue")),
                text=[f"${expected:,.0f}"] + [f"${v:+,.0f}" for v in shap_vals] + [f"${pred_wage:,.0f}"],
                textposition="outside",
            ))
            waterfall.update_layout(
                title=f"SHAP 瀑布图 — {sel_major} → {sel_job}",
                waterfallgap=0.4, height=520, showlegend=False,
                yaxis_title="薪资贡献（美元/年）",
            )
            st.plotly_chart(waterfall, use_container_width=True)

        with col_right:
            # 特征贡献排序条形图
            sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
            sorted_feats = [FEATURES[i] for i in sorted_idx]
            sorted_vals = [shap_vals[i] for i in sorted_idx]
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sorted_vals]

            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                x=sorted_vals,
                y=[FEATURE_LABELS[f] for f in sorted_feats],
                orientation="h",
                marker_color=colors,
                text=[f"${v:+,.0f}" for v in sorted_vals],
                textposition="outside",
            ))
            bar_fig.update_layout(
                title="各特征对薪资预测的贡献",
                xaxis_title="SHAP 值（美元/年）",
                yaxis_title="", height=520,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # 详细数据表
        st.subheader("特征值 vs SHAP 贡献（详细）")
        tbl = pd.DataFrame({
            "特征": [FEATURE_LABELS[f] for f in FEATURES],
            "当前值": [f"{feat_vals[i]:.4f}" for i in range(len(FEATURES))],
            "SHAP 贡献": [fmt_dollar(shap_vals[i]) for i in range(len(FEATURES))],
            "占比": [f"{v / sum(abs(s) for s in shap_vals) * 100:.1f}%"
                     for v in [abs(s) for s in shap_vals]],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # 解读
        top_contrib_feat = sorted_feats[0]
        top_contrib_val = sorted_vals[0]
        st.info(
            f"💡 **主要贡献特征：{FEATURE_LABELS[top_contrib_feat]}**\n\n"
            f"它对薪资预测的贡献是 **{fmt_dollar(top_contrib_val)}**，"
            f"占总 |SHAP| 的 {abs(top_contrib_val) / sum(abs(s) for s in shap_vals) * 100:.1f}%。"
        )

    # ═══════════════════════════════════════════════════════
    # Tab 3: 跨切片对比
    # ═══════════════════════════════════════════════════════
    with tab_compare:
        st.subheader("同岗位跨专业 vs 同专业跨岗位")
        st.caption("固定一个维度，看预测薪资在另一维度上的差异——最可信的反事实对比")

        col_left2, col_right2 = st.columns(2)

        with col_left2:
            st.markdown("#### 📈 同岗位跨专业（固定岗位）")
            sub = merged[merged["target_job"] == sel_job].copy()
            shap_sub = shap_all[shap_all["target_job"] == sel_job].copy()
            sub = sub.merge(shap_sub[["source_major", "pred_wage"]], on="source_major", how="left")
            sub = sub.sort_values("pred_wage", ascending=False)

            sel_marker = [("🎯 " if m == sel_major else "　") for m in sub["source_major"]]
            bar_colors = ["#ff6b6b" if m == sel_major else "#90caf9" for m in sub["source_major"]]

            fig_job = go.Figure()
            fig_job.add_trace(go.Bar(
                x=sub["pred_wage"],
                y=[f"{m} {sel_marker[i].strip()}" for i, m in enumerate(sub["source_major"])],
                orientation="h", marker_color=bar_colors,
                text=[f"${w:,.0f}" for w in sub["pred_wage"]],
                textposition="outside",
            ))
            fig_job.update_layout(
                title=f"同岗位（{sel_job}）不同专业的预测薪资",
                xaxis_title="预测薪资（美元/年）", height=max(420, len(sub)*28),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_job, use_container_width=True)

            job_row = job_cf[job_cf["target_job"] == sel_job]
            if not job_row.empty:
                jr = job_row.iloc[0]
                st.info(
                    f"🏆 最优专业：**{jr.best_major}**（预测 ${jr.best_pred:,.0f}）\n"
                    f"📉 最差专业：**{jr.worst_major}**（预测 ${jr.worst_pred:,.0f}）\n"
                    f"📊 差距：**${jr.pred_gap:,.0f}** ({jr.pred_gap_pct:.1f}%)"
                )

        with col_right2:
            st.markdown("#### 📉 同专业跨岗位（固定专业）")
            sub2 = merged[merged["source_major"] == sel_major].copy()
            shap_sub2 = shap_all[shap_all["source_major"] == sel_major].copy()
            sub2 = sub2.merge(shap_sub2[["target_job", "pred_wage"]], on="target_job", how="left")
            sub2 = sub2.sort_values("pred_wage", ascending=False)

            bar_colors2 = ["#ff6b6b" if j == sel_job else "#a5d6a7" for j in sub2["target_job"]]

            fig_major = go.Figure()
            fig_major.add_trace(go.Bar(
                x=[f"${w:,.0f}" for w in sub2["pred_wage"]],
                y=sub2["target_job"],
                orientation="h", marker_color=bar_colors2,
                text=[f"${w:,.0f}" for w in sub2["pred_wage"]],
                textposition="outside",
            ))
            fig_major.update_layout(
                title=f"同专业（{sel_major}）不同岗位的预测薪资",
                xaxis_title="预测薪资（美元/年）", height=max(420, len(sub2)*24),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_major, use_container_width=True)

            maj_row = major_cf[major_cf["source_major"] == sel_major]
            if not maj_row.empty:
                mr = maj_row.iloc[0]
                st.info(
                    f"💎 最高薪岗位：**{mr.high_job}**（预测 ${mr.high_pred:,.0f}）\n"
                    f"🔻 最低薪岗位：**{mr.low_job}**（预测 ${mr.low_pred:,.0f}）\n"
                    f"📊 差距：**${mr.pred_gap:,.0f}** ({mr.pred_gap_pct:.1f}%)"
                )

        # 热力图：所有岗位×专业，标注当前配对
        with st.expander("🌡️ 完整专业-岗位热力图（点击放大）"):
            heat_pivot = shap_all.pivot_table(
                index="source_major", columns="target_job", values="pred_wage"
            )
            fig_hm = go.Figure(data=go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns,
                y=heat_pivot.index,
                colorscale="Viridis",
                hovertemplate="%{y} → %{x}<br>预测薪资: $%{z:,.0f}<extra></extra>",
            ))
            fig_hm.update_layout(height=700, title="全部配对的预测薪资热力图")
            st.plotly_chart(fig_hm, use_container_width=True)

    # ═══════════════════════════════════════════════════════
    # Tab 4: 反事实干预
    # ═══════════════════════════════════════════════════════
    with tab_intervene:
        st.subheader("反事实干预：如果特征变了，薪资会怎样？")
        st.warning(
            "⚠️ 这是**模型内假设**，忽略选择偏差。\n"
            "真实世界中改变某个特征值，往往意味着你换了一个岗位或专业，\n"
            "而不是在同一个岗位上单独改变一个特征。数值仅供方向参考。"
        )

        row_dict = pair[FEATURES].to_dict()

        st.markdown("##### 🔧 拖动 slider 修改特征值")

        slider_cols = st.columns(2)
        new_vals = {}

        with slider_cols[0]:
            for feat in ["skill_gap", "semantic_score", "gap_reduction"]:
                label = FEATURE_LABELS[feat]
                cur_val = float(row_dict[feat])
                rng = 1.0 if "gap" in feat or "semantic" in feat else (4 if feat == "path_length" else 1500)
                step = 0.01 if rng < 1.1 else 1.0
                new_vals[feat] = st.slider(
                    f"{label}（当前值 {cur_val:.4f}）",
                    min_value=float(max(0, cur_val - rng * 0.5)),
                    max_value=float(min(rng, cur_val + rng * 0.5)),
                    value=cur_val, step=step, key=f"sl_{feat}",
                )

        with slider_cols[1]:
            for feat in ["path_length", "baseline_gap", "n_paths"]:
                label = FEATURE_LABELS[feat]
                cur_val = float(row_dict[feat])
                if feat == "path_length":
                    new_vals[feat] = st.slider(
                        f"{label}（当前值 {int(cur_val)}）",
                        min_value=1, max_value=5, value=int(cur_val), step=1, key=f"sl_{feat}",
                    )
                elif feat == "n_paths":
                    new_vals[feat] = st.slider(
                        f"{label}（当前值 {int(cur_val)}）",
                        min_value=100, max_value=1500, value=int(cur_val), step=50, key=f"sl_{feat}",
                    )
                else:
                    rng = 1.0
                    new_vals[feat] = st.slider(
                        f"{label}（当前值 {cur_val:.4f}）",
                        min_value=0.0, max_value=1.0, value=cur_val, step=0.01, key=f"sl_{feat}",
                    )

        # 固定未调的特征
        for f in FEATURES:
            if f not in new_vals:
                new_vals[f] = row_dict[f]

        new_feat_df = pd.DataFrame([new_vals])[FEATURES]
        new_pred = model.predict(new_feat_df)[0]
        delta = new_pred - pred_wage

        st.divider()
        col_pred, col_delta, col_real = st.columns(3)
        with col_pred:
            st.metric("干预后预测薪资", f"${new_pred:,.0f}")
        with col_delta:
            st.metric("vs 当前预测", fmt_dollar(delta),
                      delta_color="normal" if abs(delta) < 1000 else ("inverse" if delta > 0 else "normal"))
        with col_real:
            st.metric("vs 真实工资", fmt_dollar(new_pred - pair.median_wage))

        # SHAP 对比
        st.subheader("干预前后 SHAP 值对比")
        new_shap_vals = explainer.shap_values(new_feat_df[FEATURES])[0]

        compare = pd.DataFrame({
            "特征": [FEATURE_LABELS[f] for f in FEATURES],
            "当前特征值": [feat_vals[i] for i in range(len(FEATURES))],
            "干预后特征值": [new_vals[f] for f in FEATURES],
            "当前 SHAP": [fmt_dollar(shap_vals[i]) for i in range(len(FEATURES))],
            "干预后 SHAP": [fmt_dollar(new_shap_vals[i]) for i in range(len(FEATURES))],
            "SHAP 变化": [fmt_dollar(new_shap_vals[i] - shap_vals[i]) for i in range(len(FEATURES))],
        })
        st.dataframe(compare, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**🧭 预设干预场景**")
        sc1, sc2, sc3 = st.columns(3)

        def run_scenario(scenario_feats):
            sc = {**row_dict, **scenario_feats}
            sc_df = pd.DataFrame([sc])
            return model.predict(sc_df[FEATURES])[0]

        with sc1:
            if st.button("📉 skill_gap 降低 0.2", use_container_width=True):
                new_sg = max(0, row_dict["skill_gap"] - 0.2)
                r = run_scenario({"skill_gap": new_sg})
                st.success(f"gap 从 {row_dict['skill_gap']:.4f} → {new_sg:.4f}\n预测薪资: ${pred_wage:,.0f} → ${r:,.0f} ({fmt_dollar(r - pred_wage)})")

        with sc2:
            if st.button("🔀 加跳板（path_length +1）", use_container_width=True):
                new_pl = min(5, row_dict["path_length"] + 1)
                r = run_scenario({"path_length": new_pl})
                st.success(f"path_length 从 {int(row_dict['path_length'])} → {new_pl}\n预测薪资: ${pred_wage:,.0f} → ${r:,.0f} ({fmt_dollar(r - pred_wage)})")

        with sc3:
            if st.button("✨ 最理想组合（gap-0.2 + sem+0.1）", use_container_width=True):
                r = run_scenario({
                    "skill_gap": max(0, row_dict["skill_gap"] - 0.2),
                    "semantic_score": min(1, row_dict["semantic_score"] + 0.1),
                })
                st.success(f"skill_gap & semantic_score 同时改善\n预测薪资: ${pred_wage:,.0f} → ${r:,.0f} ({fmt_dollar(r - pred_wage)})")


if __name__ == "__main__":
    main()