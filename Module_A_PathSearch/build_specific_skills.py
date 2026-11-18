# -*- coding: utf-8 -*-
"""
build_specific_skills.py —— 技能粒度升级：从“通用能力”升级到“具体工具/领域技能”
=================================================================================
在整条数据管线中的位置：

  O*NET 原始数据 + 培养方案文本
        └─> [本脚本] build_specific_skills.py
              └─> onet/job_skills_specific.csv       (岗位侧具体技能)
                  onet/major_skills_specific.csv     (专业侧具体技能)
                  onet/major_skill_map_specific.txt  (专业 -> 技能字典)
                  onet/job_tool_evidence_specific.csv(工具证据明细，供追溯)
                        └─> build_unified_specific.py（合并通用技能、建岗位相似度矩阵）
                              └─> path_search.py --export path_features_specific.csv

作用（两类匹配，共用同一份技能词表 config/specific_skills.yaml）：
  1) 岗位侧：从 O*NET software_skills.csv 中把工具名归并到规范化的具体技能词表
     (Python / SQL / Tableau / AWS / SAP / HRIS / 会计 / 审计 / 翻译 ...)，
     按 Hot Technology / 冷门工具证据加权，并叠加岗位种子技能，生成
     onet/job_skills_specific.csv；
  2) 专业侧：从 onet/bylu_curricula/*.txt 培养方案文本中匹配专业侧具体技能，
     叠加 LLM 提取的通用背景技能（低权重），生成
     onet/major_skills_specific.csv 和 onet/major_skill_map_specific.txt。

输入（路径全部由 config.yaml 的 paths 段指定，本文件不写死路径）：
  paths.software_raw    onet/software_skills.csv       O*NET 工具证据原始表
  paths.occupation      onet/occupation_data.csv       职业名 -> O*NET-SOC 代码
  paths.curricula_dir   onet/bylu_curricula/*.txt      44 个专业的培养方案文本
  paths.onet_dir/major_skills_llm.csv                  专业通用背景技能（LLM 提取）
  config/specific_skills.yaml                          skills（词表+别名）/ job_seed（岗位种子）

权重口径（细节见各函数注释）：
  岗位侧：种子固定 4.0；工具证据 w = 2.0 + 2.0*raw/(raw+4.0)，raw = 1.0*hot + 0.3*cold
          两者取 max；每岗位最多 30 个技能（种子永不丢弃）。
  专业侧：课程命中 w = min(1.0, 0.55 + 0.15*ln(1+c))；具体技能最多 25 个/专业；
          通用背景 w*0.5 且封顶 0.5，最多 12 个/专业。

用法:
  python build_specific_skills.py
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
# 把 config 中的相对路径统一解析为绝对路径：P["software_raw"] 等
P = {k: (ROOT / v if not Path(v).is_absolute() else Path(v)) for k, v in CFG["paths"].items()}
JOBS = CFG["jobs"]  # 显示名 -> O*NET 官方职业名（29 个目标岗位）

# 技能词表：skills = 规范技能名 -> {category, aliases}；job_seed = 岗位 -> 保底核心技能
SPEC = yaml.safe_load((ROOT / "config" / "specific_skills.yaml").read_text(encoding="utf-8"))
SKILL_ALIASES = SPEC["skills"]
JOB_SEEDS = SPEC["job_seed"]

SOFTWARE_PATH = P["software_raw"]
OCC_PATH = P["occupation"]
CURRICULA_DIR = P["curricula_dir"]
MAJOR_LLM_PATH = P["onet_dir"] / "major_skills_llm.csv"

OUT_JOB_SPECIFIC = P["onet_dir"] / "job_skills_specific.csv"
OUT_MAJOR_SPECIFIC = P["onet_dir"] / "major_skills_specific.csv"
OUT_MAJOR_MAP = P["onet_dir"] / "major_skill_map_specific.txt"
OUT_TOOL_EVIDENCE = P["onet_dir"] / "job_tool_evidence_specific.csv"

# 截断上限：控制特征维度，避免长尾噪声；种子/课程命中的技能优先保留
MAX_SPEC_PER_JOB = 30     # 每岗位最多保留的具体技能数（种子不计入淘汰）
MAX_SPEC_PER_MAJOR = 25   # 每专业最多保留的课程命中学技能数
MAX_GENERIC_PER_MAJOR = 12  # 每专业最多保留的 LLM 通用背景技能数


def clean_major_name(name: str) -> str:
    """去掉培养方案文件名里的“培养方案”等后缀，得到干净的专业名做键。

    例：“人工智能专业培养方案” -> “人工智能”；“金融学第二学士学位培养方案” -> “金融学”。
    后缀按长的在前匹配，防止 “专业培养方案” 被短后缀截断。
    """
    s = str(name).strip()
    for suf in ("第二学士学位培养方案", "专业培养方案", "培养方案"):
        if s.endswith(suf):
            return s[: -len(suf)].strip()
    return s


def compile_alias_matchers(aliases: list[str]) -> list[tuple[str, re.Pattern[str] | None]]:
    """把技能别名编译成匹配器，返回 (别名, 正则或 None) 列表。

    匹配规则（与别名写法强相关）：
      - 含空格/符号（空格 . / + - # & ( )）或中日韩字符的别名 -> 子串匹配（返回 None，
        由 _match_text 用 str.count 统计），如 "pl/sql"、"c++"、"数据库"
      - 纯英文单词 -> 词边界正则 \\b...\\b，如 java 不会误命中 javascript
    """
    out = []
    for a in aliases:
        key = str(a).strip().lower()
        if not key:
            continue
        if any(ch in key for ch in " ./+-#&()") or any("\u4e00" <= ch <= "\u9fff" for ch in key):
            out.append((key, None))
        else:
            out.append((key, re.compile(rf"\b{re.escape(key)}\b")))
    return out


def _match_text(text: str, matchers: list[tuple[str, re.Pattern[str] | None]]) -> int:
    """统计 text 中命中该技能别名的总次数（同义词叠加计数）。

    text 需由调用方先转小写；子串别名用 count、词边界别名用 regex.findall。
    """
    cnt = 0
    for key, pat in matchers:
        if pat is not None:
            cnt += len(pat.findall(text))
        else:
            cnt += text.count(key)
    return cnt


def load_skill_matchers() -> dict[str, list[tuple[str, re.Pattern[str] | None]]]:
    """为词表中每个规范技能预编译别名匹配器（一次编译，多次复用）。"""
    return {sk: compile_alias_matchers(info.get("aliases", [])) for sk, info in SKILL_ALIASES.items()}


def build_job_specific(skill_matchers: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """岗位侧：O*NET 工具证据 + 岗位种子 -> 每岗位的具体技能及其权重。

    流程：
      1) occupation_data.csv 建「O*NET 职业名 -> SOC 代码」查找表；
      2) 对 29 个目标岗位，取 software_skills.csv 中该 SOC 的所有工具行；
      3) 每行只取 Workplace Example（具体工具名）转小写，用别名匹配每个技能，
         同时按 Hot Technology 标记分别累计 hot / cold 计数；
         注意：绝不把 Element Name（工具类别名，如 "Data base ... software"/"Portal
         server software"）拼进去匹配——类别名会大面积误报（例如 "server software"
         含子串 "r software" 会误命中技能 R，"CAD software" 会误命中 AutoCAD），
         实测拼上 Element Name 会额外制造 154 个纯误报的（岗位, 技能）对；
      4) 权重计算（取 max 合并种子与工具证据）：
           种子技能：4.0（保底，避免工具证据稀疏的岗位缺核心技能）
           工具证据：raw = 1.0*hot + 0.3*cold
                     w   = 2.0 + 2.0*raw/(raw+4.0)   -> [2, 4) 的饱和曲线
      5) 先放入全部种子，再按权重从高到低补非种子技能，直到每岗位 30 个上限
         （种子即使权重低也不会被截掉）。

    返回：
      job_df  岗位×技能表（job/skill/weight/source/n_evidence），写 job_skills_specific.csv
      ev_df   证据明细表（含 hot/cold 计数与命中的具体工具名），写 job_tool_evidence_specific.csv
    """
    occ = pd.read_csv(OCC_PATH, encoding="utf-8")
    occ["O*NET-SOC Code"] = occ["O*NET-SOC Code"].astype(str).str.strip()
    # 官方职业名 -> O*NET-SOC 代码（config.yaml 的 jobs 用的是官方职业名）
    title2code = dict(zip(occ["Title"], occ["O*NET-SOC Code"]))

    soft = pd.read_csv(SOFTWARE_PATH, encoding="utf-8", low_memory=False)
    for c in ("O*NET-SOC Code", "Workplace Example", "Hot Technology", "In Demand"):
        if c not in soft.columns:
            raise SystemExit(f"[错误] software_skills.csv 缺少列 {c}")
    soft["O*NET-SOC Code"] = soft["O*NET-SOC Code"].astype(str).str.strip()

    rows = []            # job_skills_specific.csv 的行
    evidence_rows = []   # job_tool_evidence_specific.csv 的行
    missing_jobs = []    # 在 occupation_data.csv 查不到 SOC 的岗位

    for disp, title in JOBS.items():
        code = title2code.get(title)
        if code is None:
            missing_jobs.append(disp)
            continue

        srows = soft[soft["O*NET-SOC Code"] == code]
        # skill -> [hot 命中数, cold 命中数, 总命中数]
        counts = defaultdict(lambda: [0, 0, 0])
        # skill -> 命中的具体工具名（去重，用于证据追溯）
        examples: dict[str, list[str]] = defaultdict(list)

        for _, r in srows.iterrows():
            tool = str(r.get("Workplace Example", "")).strip()
            # 只匹配具体工具名（Workplace Example）。Element Name 是跨岗位复用的
            # 类别名（"Portal server software"、"CAD software" 等），参与匹配会
            # 大面积误报（"server software" 撞 R 的 "r software"、"cad software"
            # 撞 AutoCAD），实测会多出上百个纯误报的岗位×技能对。
            text = tool.lower()
            hot = str(r.get("Hot Technology", "N")).strip().upper() == "Y"

            for sk, matchers in skill_matchers.items():
                if _match_text(text, matchers):
                    counts[sk][2] += 1
                    if hot:
                        counts[sk][0] += 1
                    else:
                        counts[sk][1] += 1
                    if tool and tool not in examples[sk]:
                        examples[sk].append(tool)

        # 第一步：种子技能先按 4.0 保底，来源标记 seed
        weights: dict[str, float] = {}
        sources: dict[str, str] = {}
        for sk in JOB_SEEDS.get(disp, []):
            weights[sk] = 4.0
            sources[sk] = "seed"

        # 第二步：工具证据权重与种子取 max，来源标记合并情况
        #   权重：seed 保底 4.0；O*NET 工具证据 2.0 ~ 4.0（饱和曲线，永不想到达 4）
        for sk, (hot_n, cold_n, total_n) in counts.items():
            raw = hot_n * 1.0 + cold_n * 0.3
            w = round(2.0 + 2.0 * raw / (raw + 4.0), 2)
            if sk in weights:
                weights[sk] = max(weights[sk], w)
                sources[sk] = "seed+onet_tool"
            else:
                weights[sk] = w
                sources[sk] = "onet_tool"

        # 第三步：种子全部保留；其余技能按权重降序补入，直到 30 个上限
        kept = {sk: w for sk, w in weights.items() if sk in JOB_SEEDS.get(disp, [])}
        for sk, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            if sk not in kept and len(kept) < MAX_SPEC_PER_JOB:
                kept[sk] = w

        # 第四步：落行（技能表 + 证据表各一行）
        for sk, w in kept.items():
            hot_n, cold_n, total_n = counts.get(sk, [0, 0, 0])
            evidence_rows.append({
                "job": disp,
                "onet_soc": code,
                "skill": sk,
                "weight": w,
                "source": sources.get(sk, "seed"),
                "hot_count": hot_n,
                "cold_count": cold_n,
                "evidence_count": total_n,
                "example_tools": "; ".join(examples.get(sk, [])),
            })
            rows.append({
                "job": disp,
                "skill": sk,
                "weight": w,
                "source": sources.get(sk, "seed"),
                "n_evidence": total_n,
            })

    # 岗位映射缺失是配置错误，直接终止并列出问题岗位
    if missing_jobs:
        raise SystemExit(f"[错误] 以下岗位在 occupation_data.csv 未找到 O*NET 代码: {missing_jobs}")

    df = pd.DataFrame(rows)
    edf = pd.DataFrame(evidence_rows)
    return df, edf


def build_major_specific(skill_matchers: dict) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """专业侧：培养方案文本匹配 + LLM 通用背景 -> 每个专业技能及其权重。

    流程：
      1) 读 onet/bylu_curricula/*.txt，专业名清洗后作键，整篇转小写；
      2) 读 major_skills_llm.csv 作为通用背景技能（若不存在则跳过）；
      3) 每专业统计每个技能别名的命中次数 c：
           w = min(1.0, 0.55 + 0.15*ln(1+c))   对数饱和，命中越多越高但不超过 1
         只保留命中的作具体技能，按权重截断到 25 个；
      4) 通用背景技能 w*0.5 且封顶 0.5（避免掩盖具体技能），按权重截断到 12 个；
      5) 最终技能列表 = 具体技能（权重降序）+ 通用背景（权重降序），
         供 path_search.py 建「专业 -> 技能」边。

    返回：
      df        专业×技能表（major/skill/weight/source/n_evidence），写 major_skills_specific.csv
      major_map 专业 -> 有序技能名列表，另由 write_major_map 写成 Python 字典文件
    """
    # 课程文本：专业名（已清洗）-> 全文小写
    curriculum: dict[str, str] = {}
    for p in sorted(CURRICULA_DIR.glob("*.txt")):
        major = clean_major_name(p.stem)
        curriculum[major] = p.read_text(encoding="utf-8", errors="ignore").lower()

    # 已有 LLM 通用技能（作为背景，权重压低，避免掩盖具体技能）
    generic: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if MAJOR_LLM_PATH.exists():
        llm = pd.read_csv(MAJOR_LLM_PATH, encoding="utf-8")
        for _, r in llm.iterrows():
            generic[clean_major_name(r["major"])].append((str(r["skill"]), float(r["weight"])))

    rows = []
    major_map: dict[str, list[str]] = {}
    majors = sorted(curriculum.keys())

    for major in majors:
        text = curriculum[major]
        spec: dict[str, float] = {}   # 课程命中的具体技能 -> 权重
        evid: dict[str, int] = {}     # 课程命中的具体技能 -> 命中次数
        for sk, matchers in skill_matchers.items():
            c = _match_text(text, matchers)
            if c:
                # 对数饱和：命中 1 次约 0.65，命中 10 次约 0.91，上限 1.0
                w = min(1.0, round(0.55 + 0.15 * math.log1p(c), 3))
                spec[sk] = w
                evid[sk] = c

        # 通用技能取权重最高的若干，压缩到 0.5 以下
        gen_skills = sorted(generic.get(major, []), key=lambda x: -x[1])[:MAX_GENERIC_PER_MAJOR]
        # 合并：具体技能优先；同名技能（如 Statistics）取两者较高权重
        combined: dict[str, tuple[float, str, int]] = {}
        for sk, w in spec.items():
            combined[sk] = (w, "curriculum", evid[sk])
        for sk, w in gen_skills:
            w2 = min(0.5, w * 0.5)
            if sk not in combined or w2 > combined[sk][0]:
                combined[sk] = (w2, "generic", 0)

        # 具体技能：按权重降序，截断到 25 个
        specific = [
            (sk, w, ev) for sk, (w, src, ev) in combined.items()
            if src == "curriculum"
        ]
        specific.sort(key=lambda x: (-x[1], x[0]))
        specific = specific[:MAX_SPEC_PER_MAJOR]

        # 通用背景：按权重降序，截断到 12 个
        generic_keep = [
            (sk, w) for sk, (w, src, _) in combined.items() if src == "generic"
        ]
        generic_keep.sort(key=lambda x: (-x[1], x[0]))
        generic_keep = generic_keep[:MAX_GENERIC_PER_MAJOR]

        # 顺序有意义：具体技能在前、通用背景在后（下游按此顺序使用）
        final_skills = [sk for sk, _, _ in specific] + [sk for sk, _ in generic_keep]

        major_map[major] = final_skills
        for sk in final_skills:
            w, src, ev = combined[sk]
            rows.append({
                "major": major,
                "skill": sk,
                "weight": round(w, 3),
                "source": src,
                "n_evidence": ev,
            })

    df = pd.DataFrame(rows)
    return df, major_map


def write_major_map(major_map: dict[str, list[str]]):
    """把“专业 -> 技能列表”写成 Python 字面量文件（MAJOR_SKILL_MAP = {...}）。

    下游 path_search.py 用 ast.literal_eval 解析（不是 yaml），因此保持纯 Python 字面量格式。
    """
    lines = ["MAJOR_SKILL_MAP = {"]
    for major in sorted(major_map):
        skills = ", ".join(f"'{s}'" for s in major_map[major])
        lines.append(f"    '{major}': [{skills}],")
    lines.append("}")
    OUT_MAJOR_MAP.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """主流程：加载词表 -> 生成岗位侧/专业侧技能表 -> 落盘 -> 打印统计摘要。"""
    skill_matchers = load_skill_matchers()

    job_df, ev_df = build_job_specific(skill_matchers)
    job_df.to_csv(OUT_JOB_SPECIFIC, index=False, encoding="utf-8")
    ev_df.to_csv(OUT_TOOL_EVIDENCE, index=False, encoding="utf-8")

    major_df, major_map = build_major_specific(skill_matchers)
    major_df.to_csv(OUT_MAJOR_SPECIFIC, index=False, encoding="utf-8")
    write_major_map(major_map)

    print(f"[岗位具体技能] {len(job_df)} 条 -> {OUT_JOB_SPECIFIC}")
    print(f"[岗位工具证据] {len(ev_df)} 条 -> {OUT_TOOL_EVIDENCE}")
    print(f"[专业具体技能] {len(major_df)} 条 -> {OUT_MAJOR_SPECIFIC}")
    print(f"[专业技能字典] -> {OUT_MAJOR_MAP}")

    # 摘要 1：具体技能最多的前 10 个岗位
    print("\n各岗位具体技能数 (前 10):")
    print(job_df.groupby("job")["skill"].nunique().sort_values(ascending=False).head(10).to_string())
    # 摘要 2：课程命中具体技能最多的前 10 个专业
    print("\n各专业具体技能数 (前 10):")
    print(major_df[major_df["source"] == "curriculum"].groupby("major")["skill"].nunique().sort_values(ascending=False).head(10).to_string())
    # 摘要 3：一个具体课程技能都没命中的专业（多为艺术/语言类，只能靠通用背景）
    spec_majors = set(major_df.loc[major_df["source"] == "curriculum", "major"])
    zero_spec = [m for m in major_map if m not in spec_majors]
    print("\n无具体课程命中的专业数:", len(zero_spec))
    if zero_spec:
        print("  " + ", ".join(zero_spec[:10]))


if __name__ == "__main__":
    main()
