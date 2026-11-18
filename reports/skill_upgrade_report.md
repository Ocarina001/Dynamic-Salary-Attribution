# 技能粒度升级报告：从“通用能力”到“具体工具/领域技能”

## 设计目标
输出“专业 → 岗位技能匹配 + 路径 + 薪资参考”。  
**薪资只作为参考列，不参与匹配分数或排名；排名只使用 `fit_score`、`core_coverage` 和路径长度。**

## 问题
旧版只有 O*NET 35 项通用能力 + 5 项领域技能，`skill_gap` 集中在 0.39~0.55，标准差仅 0.03，推荐容易被微小差异带偏。旧版 `recommend_major_jobs.py` 还把薪资加权进 `composite_score`，导致排名不是纯技能匹配。

## 做了什么
- `config/specific_skills.yaml`：140 项具体工具/领域技能及中英文别名，29 个岗位核心技能种子。
- `build_specific_skills.py`：从 O*NET `software_skills.csv` 归并各岗位具体工具技能；从 `onet/bylu_curricula/*.txt` 匹配 44 个专业的具体技能。
- `build_unified_specific.py`：合并原 40 项通用技能 + 具体技能，输出 `onet/job_skills_unified_specific.csv`，并重建中心化余弦岗位相似度 `job_similarity_matrix_specific.csv`。
- `path_search.py --export path_features_specific.csv`：在“专业–技能–岗位”图上搜索路径，输出 `path_length`、`skill_gap`、`semantic_score`、`cost` 等特征。
- `build_job_salary.py`：用 BLS OEWS 2024 全国薪资生成 `job_salary.csv`，作为岗位参考薪资。
- `recommend_major_jobs.py`：
  - `fit_score = 1 - skill_gap`
  - `core_coverage`：只计算培养方案命中的专业技能 vs 岗位 Top10 具体技能（seed / onet_tool），排除低权重通用背景技能。
  - `high_fit = fit_score >= 0.50 且 core_coverage >= 0.50`
  - `match_rank`：先高匹配，再按 `fit_score`、`core_coverage`、`path_length` 排序；**不使用薪资排序**。
  - `median_wage / p25_wage / p75_wage` 仅作为参考列附在输出表中。

## 效果

| 指标 | 通用能力版 | 具体技能版 |
|---|---:|---:|
| `skill_gap` 标准差 | 0.030 | 0.061 |
| `skill_gap` 范围 | 0.163 | 0.281 |
| 有高匹配岗位的专业 | 44/44 | 13/44 |

具体技能版区分度约翻倍；31/44 个专业在当前 29 个岗位池中没有达到高匹配门槛，输出会诚实标注“无高匹配岗位”，不强行推荐。

## 示例：技能匹配最高的首选岗位（薪资仅参考）

| 专业 | 首选高匹配岗位 | fit | 核心覆盖 | 年薪中位数 | 高匹配岗位数 |
|---|---|---:|---:|---:|---:|
| 人力资源管理专业（大数据应用） | HR Specialist | 0.555 | 0.60 | $72,910 | 2 |
| 计算机科学与技术 | Software Developer | 0.531 | 0.50 | $133,080 | 3 |
| 会计学 | Financial Manager | 0.535 | 0.70 | $161,700 | 3 |
| 人工智能 | Software Developer | 0.531 | 0.50 | $133,080 | 3 |
| 计算机科学与技术 | Computer Research Scientist（第 2 高匹配） | 0.529 | 0.70 | $140,910 | — |
| 音乐学（音乐国际教育与传播） | 无高匹配岗位 | — | — | — | 0 |

注意：人工智能专业按纯技能匹配，`Software Developer` fit 略高于 `Computer Research Scientist`；后者薪资更高但不是排名第一，说明薪资没有干预匹配排名。

## 新文件
`config/specific_skills.yaml`、`onet/job_skills_specific.csv`、`onet/major_skills_specific.csv`、`onet/major_skill_map_specific.txt`、`onet/job_skills_unified_specific.csv`、`Module_B_SemanticMatch/job_similarity_matrix_specific.csv`、`path_features_specific.csv`、`job_soc.csv`、`job_salary.csv`、`major_job_specific_ranking.csv / top5 / summary.csv`、`reports/major_job_specific_*.png`。

## 重跑
```powershell
python Module_A_PathSearch/build_specific_skills.py
python Module_A_PathSearch/build_unified_specific.py
python Module_A_PathSearch/path_search.py --export path_features_specific.csv
python build_job_salary.py
python recommend_major_jobs.py --major 人工智能
```

## 保持不变
BLS OEWS 全国薪资：`data/bls/us_wages_by_occupation_national_2024.csv`，29 岗位薪资覆盖 `coverage = 1.0`，来源 `bls_oews`。