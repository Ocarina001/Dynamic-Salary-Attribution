# -*- coding: utf-8 -*-
"""
path_search.py —— 专业 → 岗位 路径搜索与特征计算（单配置源版）
=================================================================
在整条数据管线中的位置：

  build_specific_skills.py -> build_unified_specific.py（产出统一技能表 + 相似度矩阵）
        └─> [本脚本] path_search.py
              ├─ 交互式查单对：专业 -> 岗位 的 TopN 路径与基线对比
              └─ --export：全量导出 path_features_specific.csv（后续 XGBoost/SHAP、推荐模块的输入）

数据来源（全部由 config.yaml 指定，禁止在本文件里重复定义岗位/权重/阈值）：
  onet/job_skills_unified_specific.csv                岗位 -> 技能权重
  Module_B_SemanticMatch/job_similarity_matrix_specific.csv  岗位两两相似度
  onet/major_skill_map_specific.txt                   专业 -> 具体技能
  models/bge-small-zh-v1.5                            语义相似度（本地离线加载）

图结构（无向图，节点用前缀区分类型：major:: / skill:: / job::）：
  major --major_owns--> skill            专业拥有的技能（来自 major_skill_map）
  job   --job_requires--> skill          岗位要求技能（带权重）
  skill --skill_leads_to--> job          反向边（同一组边的镜像）
  major --major_to_job--> job            专业技能与岗位技能重合 >= min_overlap 时直连（Top-K）
  job   --job_to_job--> job              岗位相似度 >= 分位阈值时互连（跳板边）
  skill --skill_cooccurs--> skill        技能共现（保证图连通，权重为共同出现岗位数）

三个特征（每个候选路径一套）：
  path_length    跳数（边数）
  semantic_score 专业名与岗位名的向量余弦（同一对专业/岗位对所有路径相同，用于跨岗位比较）
  skill_gap      缺失技能权重占比 ∈ [0,1]（归一口径，见 _compute_skill_gap）

综合代价（三项均归一到 0~1 量级后加权，权重在 config.yaml 的 weights 段）：
  cost = w_len * (L / max_hops) + w_gap * skill_gap + w_sem * (1 - semantic)

用法：
  python path_search.py                          # 演示：数据科学 -> Data Scientist
  python path_search.py --pair 心理学 "Data Scientist"
  python path_search.py --export path_features_specific.csv
"""
from __future__ import annotations

import argparse
import ast
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
# 把 config 中的相对路径统一解析为绝对路径（P["unified_skills"] 等）
P = {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in CFG["paths"].items()}

# 目标岗位清单（显示名 -> O*NET 官方名；官方名可能为 None 的哨兵岗位用于占位实验）
JOB_SELECTION: List[Tuple[str, Optional[str]]] = list(CFG["jobs"].items())

# 路径代价三项权重（config.yaml weights 段）
W_LEN = float(CFG["weights"]["path_length"])
W_GAP = float(CFG["weights"]["skill_gap"])
W_SEM = float(CFG["weights"]["semantic"])

# 图参数（config.yaml graph 段）
G = CFG["graph"]
TOP_K = int(G["major_to_job_top_k"])          # 每专业最多直连多少个岗位
MIN_OVERLAP = int(G["major_to_job_min_overlap"])  # 专业-岗位直连的最低技能重合数
SIM_QUANTILE = float(G["sim_quantile"])       # 岗位跳板边的相似度分位阈值
MAX_HOPS = int(G["max_hops"])                 # 最大跳数，同时用于 path_length 归一化
# 候选路径由 _candidate_paths 结构化枚举（L<=3），不再按条数截断
TRANSIT_CREDIT = float(G["transit_credit"])   # 途经中间岗位时其技能的折算系数
COUNT_MAJOR_SKILLS = bool(G.get("count_major_skills", False))  # 专业自身技能是否计入覆盖


def _major_id(name: str) -> str:
    """专业节点 ID（加前缀避免与同名技能/岗位节点冲突）。"""
    return f"major::{name}"


def _skill_id(name: str) -> str:
    """技能节点 ID。"""
    return f"skill::{name}"


def _job_id(name: str) -> str:
    """岗位节点 ID。"""
    return f"job::{name}"


def clean_major_name(name: str) -> str:
    """去掉培养方案后缀（14/44 个专业名带 '培养方案'，会稀释语义相似度）。

    与 build_specific_skills.clean_major_name 保持同一套规则，保证两侧键一致。
    """
    s = str(name).strip()
    for suf in ("第二学士学位培养方案", "专业培养方案", "培养方案"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            break
    return s


class PathSearchEngine:
    """路径搜索引擎：加载数据 -> 建图 -> 枚举候选路径 -> 计算特征。

    构造后可直接调用：
      get_all_paths_with_features(major, job)  单对查询
      export_features(path)                    全量导出
    """

    def __init__(self, config_path: Optional[str] = None):
        # 允许用 --config 覆盖 paths 段（例如切换到别的数据目录做对比实验）；
        # 只覆盖 paths，不覆盖 weights/graph，避免实验配置悄悄改掉代价口径。
        if config_path:
            cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            P.update({k: (Path(v) if Path(v).is_absolute() else ROOT / v)
                      for k, v in cfg["paths"].items()})

        self.job_nodes = self._load_unified_job_skills()
        self.major_nodes = self._load_major_map_llm()
        self.sim_matrix = self._load_similarity()
        self.sim_threshold = self._resolve_threshold()
        self.semantic_model = None          # 延迟加载（首次算语义时才载入 bge 模型）
        self._sem_cache: Dict[Tuple[str, str], float] = {}  # (专业,岗位) -> 余弦，避免重复编码
        # 可被敏感性分析脚本覆盖（默认取 config.graph.transit_credit）
        self.transit_credit = TRANSIT_CREDIT

        self.G = nx.Graph()                 # 无向图：边的方向语义由 node_type 决定
        self.build_graph()
        self._make_undirected()

    # ------------------------------------------------------------------ 数据
    def _load_unified_job_skills(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """读统一技能表：job -> {skills: {技能: 权重}, total_weight: 权重合计}。

        total_weight 是 skill_gap 的分母（目标岗位技能权重总和）。
        岗位缺失直接终止：说明 build_unified_specific.py 没跑或配置不一致。
        """
        df = pd.read_csv(P["unified_skills"])
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for job, grp in df.groupby("job"):
            out[str(job)] = {
                "skills": {str(r["skill"]): float(r["weight"]) for _, r in grp.iterrows()},
                "total_weight": float(grp["weight"].sum()),
            }
        missing = [d for d, _ in JOB_SELECTION if d not in out]
        if missing:
            raise SystemExit(f"[错误] 统一技能表缺少岗位: {missing}，请先运行 build_unified_specific.py")
        return out

    def _load_major_map_llm(self) -> Dict[str, List[str]]:
        """读专业->技能字典（.txt 里是一个 MAJOR_SKILL_MAP = {...} 的 Python 字面量）。

        用 ast.literal_eval 解析（不是 yaml）；专业名统一清洗；
        清洗后重名说明词典有问题，直接报错。
        """
        txt = P["major_map"].read_text(encoding="utf-8")
        start, end = txt.find("{"), txt.rfind("}")
        if start < 0 or end < 0:
            raise SystemExit(f"[错误] 无法解析专业知识表: {P['major_map']}")
        raw = ast.literal_eval(txt[start:end + 1])
        out: Dict[str, List[str]] = {}
        for k, v in raw.items():
            ck = clean_major_name(k)
            if ck in out:
                raise SystemExit(f"[错误] 专业名清洗后冲突: {ck}")
            out[ck] = list(v)
        return out

    def _load_similarity(self) -> pd.DataFrame:
        """读岗位两两相似度矩阵（列：Job_A, Job_B, Similarity_Score）。"""
        if not P["sim_matrix"].exists():
            raise SystemExit(f"[错误] 相似度矩阵不存在: {P['sim_matrix']}，请先运行 build_unified_specific.py")
        return pd.read_csv(P["sim_matrix"])

    def _resolve_threshold(self) -> float:
        """按分位数确定岗位跳板边的相似度阈值（数据驱动，避免手拍 0.5）。

        取所有岗位对相似度的 SIM_QUANTILE 分位值；超过阈值的岗位对才建跳板边。
        """
        vals = self.sim_matrix["Similarity_Score"].to_numpy()
        th = float(np.quantile(vals, SIM_QUANTILE))
        return th

    # ------------------------------------------------------------------ 建图
    def build_graph(self):
        """构建异构图：专业 / 技能 / 岗位 三类节点 + 六类边（见文件头图结构说明）。

        建完后把规模统计写进 self.stats，供 print_summary 展示。
        """
        # 技能全集 = 岗位技能 ∪ 专业知识（两者的并集才是图中可能出现的技能节点）
        all_skill_names = {s for v in self.job_nodes.values() for s in v["skills"]}
        for sk_list in self.major_nodes.values():
            all_skill_names.update(sk_list)

        # 节点：专业 / 技能 / 岗位（node_type 区分类型，display 存展示名）
        for name in self.major_nodes:
            self.G.add_node(_major_id(name), node_type="major", display=name)
        for name in all_skill_names:
            self.G.add_node(_skill_id(name), node_type="skill", display=name)
        for name, v in self.job_nodes.items():
            self.G.add_node(_job_id(name), node_type="job", display=name, skills=v["skills"])

        # 边 1：major -> skill（专业拥有的技能）
        dropped = 0
        for major, skills in self.major_nodes.items():
            for sk in skills:
                if sk in all_skill_names:
                    self.G.add_edge(_major_id(major), _skill_id(sk), edge_type="major_owns")
                else:
                    dropped += 1
        if dropped:
            print(f"[WARN] 专业知识中有 {dropped} 条技能不在词表内，已忽略")

        # 边 2/3：job <-> skill（岗位要求技能，带官方权重；两个方向都加，
        # 便于后续按无向图统一遍历，方向语义由 edge_type 区分）
        for job, v in self.job_nodes.items():
            for sk, w in v["skills"].items():
                if sk in all_skill_names:
                    self.G.add_edge(_job_id(job), _skill_id(sk),
                                    edge_type="job_requires", weight=float(w))
                    self.G.add_edge(_skill_id(sk), _job_id(job),
                                    edge_type="skill_leads_to", weight=float(w))

        # 边 4：major -> job（专业技能与岗位技能重合 >= MIN_OVERLAP 时直连，
        # 每个专业最多连 TOP_K 个岗位，按重合数降序、岗位名升序稳定排序）
        n_edges = 0
        for major, skills in self.major_nodes.items():
            mset = set(skills)
            cands = []
            for job, v in self.job_nodes.items():
                overlap = len(mset & set(v["skills"]))
                if overlap >= MIN_OVERLAP:
                    cands.append((overlap, job))
            cands.sort(key=lambda x: (-x[0], x[1]))
            for overlap, job in cands[:TOP_K]:
                self.G.add_edge(_major_id(major), _job_id(job),
                                edge_type="major_to_job", weight=float(overlap))
                n_edges += 1

        # 边 5：job <-> job（跳板边：相似度 >= 分位阈值才连，无向图里两个方向各加一次）
        n_jump = 0
        for _, r in self.sim_matrix.iterrows():
            a, b, s = str(r["Job_A"]), str(r["Job_B"]), float(r["Similarity_Score"])
            if a not in self.job_nodes or b not in self.job_nodes:
                continue
            if s < self.sim_threshold:
                continue
            self.G.add_edge(_job_id(a), _job_id(b), edge_type="job_to_job", weight=s)
            self.G.add_edge(_job_id(b), _job_id(a), edge_type="job_to_job", weight=s)
            n_jump += 1

        # 边 6：skill <-> skill（共现边：同一岗位内同时出现的技能两两相连，
        # 权重=共同出现的岗位数；作用是保证图连通、给路径提供技能间"桥"）
        pair_counts: Dict[Tuple[str, str], int] = {}
        for v in self.job_nodes.values():
            for a, b in itertools.combinations(sorted(v["skills"]), 2):
                key = (a, b) if a < b else (b, a)
                pair_counts[key] = pair_counts.get(key, 0) + 1
        for (a, b), c in pair_counts.items():
            self.G.add_edge(_skill_id(a), _skill_id(b), edge_type="skill_cooccurs", weight=float(c))

        # 图规模统计（sentinel_jobs = 配置里映射为 None 的占位岗位数，正常应为 0）
        real = sum(1 for _, t in JOB_SELECTION if t is not None)
        self.stats = {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "jobs": len(self.job_nodes),
            "real_jobs": real,
            "sentinel_jobs": len(self.job_nodes) - real,
            "skills": len(all_skill_names),
            "majors": len(self.major_nodes),
            "major_to_job": n_edges,
            "jump_pairs": n_jump,
            "sim_threshold": round(self.sim_threshold, 4),
        }

    def _make_undirected(self):
        pass  # 直接用 nx.Graph（无向），边的方向语义由 node_type 决定

    # ------------------------------------------------------------------ 语义
    def _load_semantic(self):
        """延迟加载本地 bge 模型（首次算语义时才载入，避免纯导出场景浪费内存）。

        加载失败不中断：semantic_score 统一回退为 0.5（中性值），
        代价函数中该项变成常数，排序退化为长度+缺口驱动。
        """
        if self.semantic_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        path = str(P["model_dir"])
        try:
            self.semantic_model = SentenceTransformer(path, local_files_only=True)
            print(f"[语义模型] 已加载: {path}")
        except Exception as e:
            print(f"[WARN] 语义模型加载失败({e})，semantic_score 统一为 0.5")
            self.semantic_model = False

    def _semantic_score(self, major: str, job: str) -> float:
        """专业名与岗位名的语义相似度：归一化向量的点积（余弦，范围 [-1,1]）。

        结果按 (专业,岗位) 缓存；同一对专业/岗位的所有候选路径共用同一个值。
        """
        key = (major, job)
        if key in self._sem_cache:
            return self._sem_cache[key]
        self._load_semantic()
        if not self.semantic_model:
            val = 0.5
        else:
            embs = self.semantic_model.encode(
                [major, job], convert_to_numpy=True, normalize_embeddings=True)
            val = float(np.dot(embs[0], embs[1]))
        self._sem_cache[key] = val
        return val

    # ------------------------------------------------------------------ 特征
    def _compute_skill_gap(self, path_node_ids: List[str], target_job: str,
                           transit_credit: Optional[float] = None) -> float:
        """缺失技能权重占比（0 = 完全覆盖，1 = 完全不覆盖）。

        公式：gap = max(0, 1 - Σ_sk min(目标权重_sk, 已获得信用_sk) / 目标总权重)

        路径上各节点的"信用"记法：
          - 技能节点：视为已掌握该技能（按目标岗位该技能的权重记满）
          - 中间岗位节点：其技能只按 transit_credit（默认取 config，0.5）折算，
            避免"26/30 个岗位技能名完全相同 → 任何中转都让缺口恒为 0"的退化。
            transit_credit 可传入覆盖，供敏感性分析使用。
          - 终点岗位本身不产生信用（否则 gap 恒为 0）
          - 起点专业自身的技能默认不计入（与旧 v1 静态口径区分），由
            config.graph.count_major_skills 控制；开启则按目标权重记满

        同一技能被多个节点覆盖时取最大信用（max），不做累加。
        """
        tc = self.transit_credit if transit_credit is None else float(transit_credit)
        target = self.job_nodes[target_job]["skills"]
        total = float(self.job_nodes[target_job]["total_weight"])
        if not target or total <= 0:
            return 1.0

        credit: Dict[str, float] = {}
        for idx, nid in enumerate(path_node_ids):
            node = self.G.nodes[nid]
            typ, disp = node.get("node_type"), node.get("display")
            if typ == "skill":
                credit[disp] = max(credit.get(disp, 0.0), target.get(disp, 0.0))
            elif typ == "job" and idx != len(path_node_ids) - 1:
                for sk, w in self.job_nodes.get(disp, {}).get("skills", {}).items():
                    credit[sk] = max(credit.get(sk, 0.0), tc * w)
            elif typ == "major" and COUNT_MAJOR_SKILLS:
                for sk in self.major_nodes.get(disp, []):
                    credit[sk] = max(credit.get(sk, 0.0), target.get(sk, 0.0))

        covered = sum(min(target[sk], credit.get(sk, 0.0)) for sk in target)
        return float(max(0.0, 1.0 - covered / total))

    def _path_display(self, path: List[str]) -> List[str]:
        """把节点 ID 列表转成「专业:xxx / 技能:xxx / 岗位:xxx」的展示串。"""
        icons = {"major": "专业", "skill": "技能", "job": "岗位"}
        return [f"{icons.get(self.G.nodes[n].get('node_type'), '?')}:{self.G.nodes[n].get('display')}"
                for n in path]

    def _cost(self, path_length: int, gap: float, sem: float) -> float:
        """综合代价：长度项归一到 [0,1]（L/max_hops），缺口本身就∈[0,1]，语义取 1-cos。"""
        return (W_LEN * (path_length / max(1, MAX_HOPS))
                + W_GAP * gap
                + W_SEM * (1.0 - sem))

    # ------------------------------------------------------------------ 候选路径
    def _candidate_paths(self, major: str, job: str) -> List[List[str]]:
        """结构化枚举 L<=3 的候选路径（替代被截断的 shortest_simple_paths，
        避免"只取了前 60 条"带来的基线偏差）：
          L1: 专业 -> 岗位
          L2: 专业 -> 技能 -> 岗位 | 专业 -> 岗位A -> 岗位B
          L3: 专业 -> 技能 -> 技能 -> 岗位 | 专业 -> 岗位A -> 技能 -> 岗位
              | 专业 -> 岗位A -> 岗位B -> 岗位

        注意：枚举上限是 L<=3（4 个节点）；config 的 max_hops=4 只用于
        path_length 的归一化分母，不扩大枚举范围。
        """
        mid, jid = _major_id(major), _job_id(job)
        # 专业侧技能（只保留图中存在的技能节点，避免脏数据）
        maj_skills = [s for s in self.major_nodes[major] if _skill_id(s) in self.G.nodes]
        # 目标岗位的技能集（技能->技能展开时只关心这些技能）
        tgt_skills = sorted(self.job_nodes[job]["skills"])
        # 与专业直连的岗位（major_to_job 边），以及目标岗位的相似岗位（jump 边）
        direct_jobs = [n for n in self.G.neighbors(mid) if self.G.nodes[n]["node_type"] == "job"]
        jump_jobs = [n for n in self.G.neighbors(jid) if self.G.nodes[n]["node_type"] == "job"]

        cands: List[List[str]] = []
        # L1：专业直连目标岗位
        if self.G.has_edge(mid, jid):
            cands.append([mid, jid])

        # 以专业技能为跳板
        for s in maj_skills:
            sid = _skill_id(s)
            # L2：专业 -> 技能 -> 岗位（该技能正是目标岗位要求的）
            if self.G.has_edge(sid, jid):
                cands.append([mid, sid, jid])
            # L3：专业 -> 技能 -> 技能 -> 岗位（技能间靠共现边连通）
            if s in self.job_nodes[job]["skills"]:  # 技能->技能只在目标技能集内展开
                for s2 in tgt_skills:
                    if s2 == s:
                        continue
                    s2id = _skill_id(s2)
                    if self.G.has_edge(sid, s2id) and self.G.has_edge(s2id, jid):
                        cands.append([mid, sid, s2id, jid])

        # 以中间岗位为跳板
        for jd in direct_jobs:
            # L2：专业 -> 岗位A -> 目标岗位
            if jd != jid and self.G.has_edge(jd, jid):
                cands.append([mid, jd, jid])
            # L3：专业 -> 岗位A -> 技能 -> 目标岗位
            for s in tgt_skills:
                sid = _skill_id(s)
                if self.G.has_edge(jd, sid) and self.G.has_edge(sid, jid):
                    cands.append([mid, jd, sid, jid])
            # L3：专业 -> 岗位A -> 岗位B -> 目标岗位
            for jj in jump_jobs:
                if jj in (jid, jd):
                    continue
                if self.G.has_edge(jd, jj):
                    cands.append([mid, jd, jj, jid])

        # 去重保序（同一路径可能被多种模板枚举到）
        uniq, seen = [], set()
        for p in cands:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def get_all_paths_with_features(self, source_major: str, target_job: str) -> List[dict]:
        """枚举一对（专业, 岗位）的所有候选路径，并为每条路径算特征 + 代价。

        返回列表，每项：path_ids / path_display / path_length / skill_gap /
        semantic_score（该对专业岗位的常量）/ cost。
        """
        if source_major not in self.major_nodes:
            print(f"[错误] 未找到专业: {source_major}")
            return []
        if target_job not in self.job_nodes:
            print(f"[错误] 未找到岗位: {target_job}")
            return []

        # 语义分数与路径无关：同一对专业/岗位只算一次
        sem = self._semantic_score(source_major, target_job)
        paths = self._candidate_paths(source_major, target_job)
        if not paths:
            print("[寻路] 未找到任何路径")
            return []

        results = []
        for p in paths:
            L = len(p) - 1                       # 跳数 = 节点数 - 1
            gap = self._compute_skill_gap(p, target_job)
            results.append({
                "path_ids": p,
                "path_display": self._path_display(p),
                "path_length": L,
                "skill_gap": round(gap, 4),
                "semantic_score": round(sem, 4),
                "cost": round(self._cost(L, gap, sem), 4),
            })
        return results

    # ------------------------------------------------------------------ 导出
    def export_features(self, out_path: Path) -> pd.DataFrame:
        """全量导出：遍历 44 专业 × 29 岗位，每对只保留 cost 最小的路径作为特征行。

        每行含义：
          source_major / target_job   专业、岗位
          path_length / skill_gap / semantic_score / cost   最优（cost 最小）路径的特征
          baseline_gap                 BFS 最短路径（不看权重）的 gap —— 朴素基线
          baseline_best_gap            同跳数内 gap 最小的值 —— 强基线（上界参考）
          gap_reduction                baseline_gap - 本方法的 gap
          gap_reduction_vs_best        baseline_best_gap - 本方法的 gap
          n_paths                      该对枚举到的候选路径数
          path_nodes                   最优路径的展示串
        """
        rows = []
        majors = list(self.major_nodes)
        jobs = list(self.job_nodes)
        print(f"[导出] {len(majors)} 专业 × {len(jobs)} 岗位 = {len(majors) * len(jobs)} 对")
        for i, major in enumerate(majors, 1):
            for job in jobs:
                res = self.get_all_paths_with_features(major, job)
                if not res:
                    continue
                # 本方法的选路：cost 最小
                best = min(res, key=lambda x: x["cost"])
                # 基线 1：BFS 最短路径（等权最短路，不看 skill_gap）
                minL = min(x["path_length"] for x in res)
                hop_paths = [x for x in res if x["path_length"] == minL]
                bfs_ids = nx.shortest_path(self.G, _major_id(major), _job_id(job))
                bfs_gap = round(self._compute_skill_gap(bfs_ids, job), 4)
                # 基线 2：同跳数内 gap 最优（比 BFS 更强的参照）
                best_hop_gap = min(x["skill_gap"] for x in hop_paths)
                rows.append({
                    "source_major": major,
                    "target_job": job,
                    "path_length": best["path_length"],
                    "skill_gap": best["skill_gap"],
                    "semantic_score": best["semantic_score"],
                    "cost": best["cost"],
                    "baseline_gap": bfs_gap,
                    "baseline_best_gap": best_hop_gap,
                    "gap_reduction": round(bfs_gap - best["skill_gap"], 4),
                    "gap_reduction_vs_best": round(best_hop_gap - best["skill_gap"], 4),
                    "n_paths": len(res),
                    "path_nodes": " -> ".join(best["path_display"]),
                })
            if i % 10 == 0:
                print(f"   ... {i}/{len(majors)}")
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[导出] 完成 {len(df)} 行 -> {out_path}")
        return df

    def print_summary(self):
        """打印图规模、建图参数与当前代价公式（便于核对 config 是否生效）。"""
        s = self.stats
        print("[图规模] 节点 %d | 边 %d | 岗位 %d | 技能 %d | 专业 %d" % (s["nodes"], s["edges"], s["jobs"], s["skills"], s["majors"]))
        print("[建图] major_to_job %d 条 | 跳板对 %d 对(阈值 %.4f, 分位 %.2f) | 中间岗位技能折算 %.2f"
              % (s["major_to_job"], s["jump_pairs"], s["sim_threshold"],
                 SIM_QUANTILE, TRANSIT_CREDIT))
        print("[代价] cost = %.1f*(L/%d) + %.1f*gap + %.1f*(1-sem)"
              % (W_LEN, MAX_HOPS, W_GAP, W_SEM))


def main():
    """命令行入口：
      --pair MAJOR JOB   查一对专业/岗位的路径（默认演示 数据科学与大数据技术 -> Data Scientist）
      --export PATH      全量导出路径特征 CSV
      --config YAML      覆盖 config.yaml 的 paths 段（做数据目录对比实验）
      --top N            交互模式展示前 N 条路径（默认 3）
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("MAJOR", "JOB"))
    ap.add_argument("--export", metavar="PATH")
    ap.add_argument("--config", metavar="YAML", help="使用替代配置（例如 config_specific.yaml）")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    engine = PathSearchEngine(args.config)
    engine.print_summary()

    if args.export:
        engine.export_features(Path(args.export))
        return

    # 交互查询：默认演示一对
    major, job = args.pair if args.pair else ("数据科学与大数据技术", "Data Scientist")
    res = engine.get_all_paths_with_features(major, job)
    if not res:
        return
    print(f"\n[寻路] {major} -> {job} | 路径 {len(res)} 条 | semantic={res[0]['semantic_score']:.4f}")
    # 基线：按（跳数, gap）排序取前 N（同跳数最优 gap 的上界参考）
    base = sorted(res, key=lambda x: (x["path_length"], x["skill_gap"]))[: args.top]
    # 本方法：按 cost 排序取前 N
    ours = sorted(res, key=lambda x: x["cost"])[: args.top]
    # 纯 BFS 最短路径（不看权重的朴素基线）
    bfs_ids = nx.shortest_path(engine.G, _major_id(major), _job_id(job))
    bfs_gap = engine._compute_skill_gap(bfs_ids, job)
    print("\n[Baseline: 最短路径(BFS)]")
    print(f"  gap={bfs_gap:.3f} L={len(bfs_ids) - 1} | {' -> '.join(engine._path_display(bfs_ids))}")
    print("\n[Baseline+(同跳数内最优 gap，上界参考)]")
    for i, r in enumerate(base, 1):
        print(f"  {i}. L={r['path_length']} gap={r['skill_gap']:.3f} "
              f"| {' -> '.join(r['path_display'])}")
    print("\n[Ours: 按 cost 排序]")
    for i, r in enumerate(ours, 1):
        print(f"  {i}. L={r['path_length']} gap={r['skill_gap']:.3f} "
              f"cost={r['cost']:.3f} | {' -> '.join(r['path_display'])}")
    print(f"\n[结论] BFS gap={bfs_gap:.3f} | Ours gap={ours[0]['skill_gap']:.3f} "
          f"(较 BFS 下降 {bfs_gap - ours[0]['skill_gap']:.3f}, "
          f"较同跳数最优下降 {base[0]['skill_gap'] - ours[0]['skill_gap']:.3f})")


if __name__ == "__main__":
    main()
