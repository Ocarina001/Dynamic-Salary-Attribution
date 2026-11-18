# onet/ 数据目录说明

本目录存放 O*NET 相关的外部输入数据，以及由项目脚本生成的中间表 / 最终表。
所有 CSV 均为 `utf-8` 编码，第一行为表头。**CSV 内不要加注释行**（会让 `pandas.read_csv`
把它当数据，破坏解析）；所有说明写在本文档。

---

## 数据流总览

```
外部输入（O*NET 官方数据 + 早期项目积累，不随代码变）
  occupation_data.csv          O*NET 职业主表：SOC 代码 <-> 职业名
  software_skills.csv          O*NET 工具证据：每个职业用什么软件
  job_skills_unified.csv       旧版岗位技能表：29 岗位 × 40 项通用技能
  major_skills_llm.csv         专业通用背景技能：44 专业 × LLM 提取的能力

build_specific_skills.py（读 config/specific_skills.yaml + 上面 4 个输入 + bylu_curricula/*.txt）
  ├─ 生成 job_skills_specific.csv         29 岗位 × 具体技能（工具证据 + 种子）
  ├─ 生成 major_skills_specific.csv       44 专业 × 具体技能（课程命中 + 通用背景）
  ├─ 生成 major_skill_map_specific.txt    专业 -> 技能字典（给 path_search 建图）
  └─ 生成 job_tool_evidence_specific.csv  工具证据明细（可追溯用，不参与下游计算）

build_unified_specific.py（读 job_skills_unified + job_skills_specific）
  ├─ 生成 job_skills_unified_specific.csv 通用 40 项 + 具体技能合并表（144 技能）
  └─ 生成 ../Module_B_SemanticMatch/job_similarity_matrix_specific.csv 岗位相似度矩阵

path_search.py（读 unified_specific + major_map + 相似度矩阵 + models/bge-small-zh-v1.5）
  -> 建图寻路 -> ../path_features_specific.csv
```

---

## 一、外部输入（4 个）

### occupation_data.csv —— O*NET 职业主表

- 规模：1,016 行 × 3 列
- 列：`O*NET-SOC Code`（如 `15-2051.00`）、`Title`（官方职业名）、`Description`
- 作用：项目里唯一做「职业名 -> SOC 代码」翻译的字典。config.yaml 的 29 个岗位用
  O*NET 官方名，靠它查到 SOC 代码，才能去 software_skills.csv 取工具证据、
  去 data/bls 薪资表取工资
- 消费方：build_specific_skills.py、build_job_salary.py

### software_skills.csv —— O*NET 工具/软件证据

- 规模：31,821 行 × 7 列；8,753 个不同工具名；其中 Hot Technology=Y 的 11,571 行
- 列：`O*NET-SOC Code`、`Title`、`Workplace Example`（具体工具名，如 Adobe Acrobat）、
  `Element ID`、`Element Name`（工具类别，如 "Data base user interface and query software"）、
  `Hot Technology`（Y/N）、`In Demand`（Y/N）
- 作用：build_specific_skills.py 只取 `Workplace Example`（具体工具名）转小写，用
  specific_skills.yaml 的别名匹配。**绝不把 `Element Name`（类别名）拼进匹配文本**——
  类别名跨岗位复用，会大面积误报（如 "Portal server software" 含子串 "r software"
  会误命中技能 R，"CAD software" 误命中 AutoCAD），实测拼上后会多出 154 个纯误报的
  （岗位, 技能）对
- 权重来源：Hot Technology=Y 记 1.0、冷门工具记 0.3，即 raw = 1.0*hot + 0.3*cold

### job_skills_unified.csv —— 旧版通用岗位技能表

- 规模：1,122 行 × 4 列；29 岗位 × 40 项技能（1,015 行 source=onet + 107 行 source=domain）
- 列：`job`、`skill`、`weight`、`source`
- 40 项 = 35 项 O*NET 通用能力（IM 评分技能：Reading Comprehension、Mathematics、
  Critical Thinking ...）+ 5 项领域技能（Databases / Data Analysis / Statistics /
  Data Visualization / Machine Learning，source=domain）
- 权重：直接来自 O*NET IM 官方分数（0~5 量纲，如 4.38、3.5）
- 作用：升级前的「旧版背景表」，build_unified_specific.py 把它作为通用能力一侧合并进新表

### major_skills_llm.csv —— 专业通用背景技能

- 规模：432 行 × 4 列；44 专业 × 31 个通用能力
- 列：`major`、`skill`、`weight`、`evidence`（提取依据，如「统计学、概率论与数理统计课程」）
- 作用：LLM 从培养方案中提取的通用背景技能。专业侧只作低权重背景：
  合并时权重 ×0.5 且封顶 0.5，避免掩盖真正从课程词命中的具体技能

---

## 二、生成文件（可删除、可由脚本重建）

### job_skills_specific.csv —— 29 岗位 × 具体技能

- 规模：800 行 × 5 列；29 岗位；109 个不同技能；平均每岗位 27.6 个
- 列：`job`、`skill`、`weight`、`source`、`n_evidence`
- source 三类：
  - `onet_tool`（553 行）：纯靠 O*NET 工具证据算出权重
  - `seed+onet_tool`（172 行）：种子技能同时有工具证据
  - `seed`（75 行）：只有种子保底（工具表里没证据）
- 权重规则：工具证据 w = 2.0 + 2.0*raw/(raw+4.0)（区间 [2,4)）；种子固定 4.0；
  两者取 max；每岗位截断到 30 个（MAX_SPEC_PER_JOB），种子永不丢弃
- n_evidence：该技能命中的工具证据条数

### major_skills_specific.csv —— 44 专业 × 具体技能

- 规模：1,017 行 × 5 列；44 专业；source=curriculum 646 行 + source=generic 371 行
- 列：`major`、`skill`、`weight`、`source`、`n_evidence`
- 权重规则：课程命中 w = min(1.0, 0.55 + 0.15*ln(1+c))（c 为命中次数）；
  具体技能最多保留 25 个/专业；通用背景最多 12 个且权重 <= 0.5
- n_evidence：该技能在培养方案文本中被命中的次数

### major_skill_map_specific.txt —— 专业 -> 技能字典

- 本质是 Python 字面量（`MAJOR_SKILL_MAP = {...}`），path_search.py 用
  ast.literal_eval 解析（不是 yaml.safe_load）
- 每个专业一个有序技能列表：具体技能在前、通用背景在后，顺序即权重降序
- 建图时用于生成「专业 -> 技能」的 major_owns 边

### job_skills_unified_specific.csv —— 最终合并表

- 规模：1,878 行 × 4 列；29 岗位；144 个不同技能（40 通用 + 具体技能去重后）
- 列：`job`、`skill`、`weight`、`source`
- 合并规则：新旧两表拼接后按「岗位×技能」去重、保留最大权重（避免 Statistics 这类
  通用/具体重名技能被双重计权）；source 用 + 连接成组合标签
  （如 domain+seed+onet_tool）
- 作用：path_search.py 的岗位侧唯一输入（config.yaml 的 paths.unified_skills）

### job_tool_evidence_specific.csv（当前目录中可能不存在）

- build_specific_skills.py 理论上会生成它：每个岗位×技能一行，记录命中的
  hot/cold 次数与具体工具名列表（example_tools），用于追溯权重来源
- 只用于人工排查，不参与下游任何计算；缺失不影响管线运行

---

## 三、权重口径一览

| 侧 | 来源 | 公式 | 量纲/区间 |
|---|---|---|---|
| 岗位（通用） | O*NET IM 官方分数 | 直接取值 | 0~5 |
| 岗位（工具证据） | software_skills.csv 命中 | 2.0 + 2.0*raw/(raw+4.0)，raw = 1.0*hot + 0.3*cold | [2, 4) |
| 岗位（种子） | config/specific_skills.yaml 的 job_seed | 固定 4.0 | 4.0 |
| 专业（课程命中） | 培养方案文本命中次数 c | min(1.0, 0.55 + 0.15*ln(1+c)) | (0.55, 1.0] |
| 专业（通用背景） | major_skills_llm.csv | w*0.5，封顶 0.5 | (0, 0.5] |

每岗位技能数上限：具体 30（job_skills_specific.csv）；每专业：具体 25 + 通用背景 12。

---

## 四、易误读点

1. 「40 项」不是纯 O*NET：其中 5 项（Databases / Data Analysis / Statistics /
   Data Visualization / Machine Learning）是早期项目自建的领域技能，以 source=domain
   保留在 job_skills_unified.csv 这张静态表里（即使 config.yaml 中的 domain_evidence
   定义已删除，数据文件不受影响）。
2. 通用权重可能大于具体权重：O*NET IM 是 0~5 量纲（Mathematics 4.88），具体技能
   封顶约 4.0——所以合并后个别岗位的高重要性通用技能排在前列，这是设计使然
   （IM 是官方分数）。
3. 本目录生成文件全部可重建：删除后按下面的顺序重跑即可完整复现，不丢信息。

---

## 五、重建顺序

```powershell
python Module_A_PathSearch/build_specific_skills.py    # 生成 job/major specific + major map（及工具证据明细）
python Module_A_PathSearch/build_unified_specific.py   # 生成 unified specific + 岗位相似度矩阵
python Module_A_PathSearch/path_search.py --export path_features_specific.csv
python build_job_salary.py                             # 薪资表（读 data/bls/）
python recommend_major_jobs.py --major "人工智能"        # 推荐结果
```

---

## 六、维护约定

1. 新增/改名技能必须改 config/specific_skills.yaml，然后按「重建顺序」从头跑；
   直接手改本目录的生成 CSV 会在下次重跑时被覆盖。
2. CSV 内禁止写注释行；说明写在本文档或 SOURCE.txt（data/bls/）里。
3. 修改 paths / 筛选阈值请改 config.yaml，不要写死在脚本中。
