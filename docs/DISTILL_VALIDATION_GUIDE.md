# 知识蒸馏验证体系：业界标准 vs 当前实现 — 差距分析与改进建议

> 目标：逐项对照业界最佳实践，找出当前项目验证体系的不足，给出可执行的改进方案。

---

## 总览：差距矩阵

| # | 验证维度 | 业界标准 | 当前实现 | 差距等级 | 优先级 |
|---|---------|---------|---------|---------|--------|
| 1 | 训练中监控 | WandB/TensorBoard 实时看板 | YOLOv5 控制台 + CSV/JSON | ⚠️ 中等 | P1 |
| 2 | 自动指标 | PPL + BLEU + ROUGE + BERTScore + BLEURT | PPL + ROUGE-L (自实现) | 🔴 严重 | P0 |
| 3 | LLM-as-Judge | 多裁判 + 位置校准 + 评分一致性 | 单裁判 4 维度 | ⚠️ 中等 | P1 |
| 4 | Benchmark 评测 | MMLU / C-Eval / GSM8K / HumanEval | **无** | 🔴 严重 | P0 |
| 5 | 灾难性遗忘检测 | 蒸馏前后 benchmark diff | **无** | 🔴 严重 | P0 |
| 6 | 生成质量统计 | distinct-n / 重复率 / 长度分布 | **无** | ⚠️ 中等 | P2 |
| 7 | 数据质量验证 | 语义去重 / 多样性 / 难度分层 | 仅 config 占位，未接入代码 | 🔴 严重 | P1 |
| 8 | 师生差距分析 | KL 散度 / 分层对比 / 难度分层 | 无显式分析 | ⚠️ 中等 | P2 |
| 9 | 消融实验框架 | 系统化超参扫描 + 报告 | **无** | 🟡 轻微 | P3 |

---

## 1. 训练中监控

### 业界怎么做

```
WandB / TensorBoard / MLflow 实时看板：
├── Loss 曲线 (train + eval, 叠加显示)
├── 学习率衰减曲线
├── 梯度范数 (grad_norm) 随时间变化
├── GPU 利用率 / 显存 / 功耗 时间序列
├── Token 准确率 (mean_token_accuracy) 趋势
├── 各层参数更新幅度 (weight update ratio)
└── 自动异常告警 (loss 突增 / grad 消失 / OOM)
```

典型做法：
- **HuggingFace Transformers**: 内置 `report_to="wandb"` 一行代码接入
- **LLaMA-Factory**: 训练日志 + TensorBoard + WandB 三通道
- **Axolotl**: WandB 默认开启，自动记录所有 TrainingArguments

### 当前项目实现

```
控制台 YOLOv5 风格输出
  └── train.log (文本文件)
  └── results.csv (epoch 级别)
  └── train_history.json (step 级别, 最近新增)
  └── eval_history.json (epoch 级别, 最近新增)
```

**有的**：
- ✅ step 级别的 loss / grad_norm / lr / accuracy 记录（`trainer.state.log_history` → `train_history.json`）
- ✅ epoch 级别的 train_loss / eval_loss / lr（`results.csv`）
- ✅ NaN 梯度自动检测与停止

**缺的**：
- ❌ 无可视化 dashboard（需手动 Python/Excel 绘图）
- ❌ 无梯度健康趋势监控（grad_norm 是否逐步增大/缩小？）
- ❌ 无 token 准确率趋势（mean_token_accuracy 未记录到 CSV）
- ❌ `report_to="none"` 硬编码，无法接入 WandB

### 差距影响

训练出问题（如 loss 不收敛）时，只能翻文本日志排查。无法快速通过曲线图判断问题阶段。多实验对比时需要手动收集 CSV。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P0** | `results.csv` 增加 `grad_norm` 和 `mean_token_accuracy` 列 | 5 行代码 | 这两个是判断训练健康的核心信号，目前 CSV 缺失 |
| **P1** | `report_to` 改为可配置（`config.yaml` 增加 `report_to: "wandb"` 字段） | 10 行代码 | WandB 免费额度足够个人使用，一行配置即可开启 |
| **P2** | 训练结束后自动用 matplotlib 生成 `results.png`（loss/lr/grad 三合一图） | ~20 行 | 不依赖外部服务，每个实验目录自带可视化 |

---

## 2. 自动指标评估

### 业界怎么做

| 指标 | 衡量什么 | 使用场景 |
|------|---------|---------|
| **PPL** | 语言模型拟合度 | 基础指标，所有蒸馏论文必报 |
| **ROUGE-1/2/L** | n-gram / LCS 重叠 | 摘要/问答（ROUGE-1 查关键词，ROUGE-2 查短语，ROUGE-L 查顺序） |
| **BLEU** | n-gram 精确匹配 | 机器翻译，AI Infra 代码生成场景也适用 |
| **BERTScore** | 语义相似度（BERT embedding 余弦） | **LLM 评估的核心指标**，不受字面差异影响 |
| **BLEURT** | 基于 BERT 的 learned metric | 更准确的语义质量评估 |
| **METEOR** | 同义词匹配 + 词序 | 比 BLEU 对改写更友好 |
| **chrF** | 字符级 n-gram F-score | 对中文等无空格语言更友好 |

**标准评估组合**（来自 MiniLLM / Orca / Qwen 蒸馏论文）：

```
自动指标 = PPL + ROUGE-L + BERTScore + BLEU
         └─ PPL: 分布拟合
         └─ ROUGE-L: 字面重叠（保守下限）
         └─ BERTScore: 语义相似（核心）
         └─ BLEU: n-gram 精确度（代码/公式场景关键）
```

### 当前项目实现

```python
# eval.py — evaluate_perplexity() + evaluate_rouge()
# rouge_l_score() — 自实现 LCS，中文逐字比较

当前: PPL + ROUGE-L
缺失: BERTScore, BLEU, ROUGE-1, ROUGE-2
```

### 差距影响

这是**当前最严重的差距**。ROUGE-L 对中文 LLM 评估有致命缺陷：

**例子**：两个回答表述完全不同但都正确：
- 参考答案："FFmpeg 使用 `-c:v libx264 -preset slow` 进行 H.264 编码"
- LoRA 生成："可以通过 `ffmpeg -c:v libx264 -preset veryslow` 启用 x264 编码器"

> ROUGE-L ≈ 0.15（字面重叠低） → ❌ 误判为差
> BERTScore ≈ 0.92（语义高度相似） → ✅ 正确判定

ROUGE-L **惩罚改写**，而 LLM 的天然优势就是灵活改写。仅靠 ROUGE-L 会系统性地低估模型质量。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P0** | 增加 **BERTScore**（`pip install bert-score`，`bert_score.score()` 一行调用） | ~15 行 | 解决 ROUGE-L 对改写误判的核心缺陷，业界标准 |
| **P1** | ROUGE-L 拆分为 Recall/Precision/F1 三项分别报告 | ~5 行 | Precision 低 = 生成冗余，Recall 低 = 内容缺失，诊断价值高 |
| **P2** | AI Infra 代码类问题增加 BLEU | ~10 行 | CUDA/FFmpeg 命令等有标准答案的问题，BLEU 比 ROUGE 更准确 |

---

## 3. LLM-as-Judge

### 业界怎么做

```
AlpacaEval 2.0 / MT-Bench / Chatbot Arena 标准做法：

1. 多裁判模型评估（GPT-4 + Claude + Gemini → 取均值/投票）
2. 位置偏置消除（swap position：A/B 交换位置评估两次，取平均）
3. 评分一致性检验（Cohen's Kappa / Kendall's W）
4. 长度偏置校正（长回答天然高分 → length-controlled win rate）
5. 参考-free 评估（不依赖参考答案，纯开放式判断）
6. 评分标准细粒度化（rubric-based: 1-10 分，每分有明确标准）
7. 裁判模型自评一致性（同一问题问两次，检查评分是否稳定）
```

### 当前项目实现

```python
# eval.py — evaluate_with_llm_judge()
# 4 维度打分 (accuracy/relevance/completeness/overall, 1-5 分)
# 每个问题只评一次（LoRA + Base 分开调用）
# 单裁判模型

已有的：
✅ 4 维度结构化评分
✅ Base vs LoRA 对比（△ 值）
✅ 逐条详细点评 (comment)

缺失的：
❌ 无位置偏置校正（Base/LoRA 调用顺序是否影响评分？未检验）
❌ 无评分一致性检验（裁判模型自身是否稳定？）
❌ 无长度偏置校正（长回答天然高分？）
❌ 无多裁判交叉验证
```

### 差距影响

单裁判模型的评分误差无法量化。如果裁判模型 API 不稳定（如本地量化模型），可能导致评分波动，无法准确判断蒸馏效果。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P1** | report 中增加裁判模型分数的标准差/方差 | ~10 行 | 至少知道评分稳定性 |
| **P2** | 增加位置交换校准（swap LoRA/Base 评估顺序，取两次平均） | ~20 行 | 消除裁判模型偏见，AlpacaEval 标准做法 |
| **P3** | 支持多裁判模型配置（list 形式，取均值） | ~30 行 | 显著提升评分可靠性 |

---

## 4. Benchmark 评测

### 业界怎么做

蒸馏论文（Orca / Phi / MiniLLM / Qwen）标准 benchmark 套件：

| Benchmark | 领域 | 题型 | 蒸馏论文必报 |
|-----------|------|------|-------------|
| **MMLU** | 57 学科知识 | 选择题 (4-option) | ✅ |
| **C-Eval** | 中文 52 学科 | 选择题 (4-option) | ✅ (中文模型) |
| **GSM8K** | 小学数学推理 | 自由回答 | ✅ |
| **HumanEval** | Python 代码生成 | 补全 + 测试 | ✅ |
| **IFEval** | 指令遵循 | 约束检查 | ⚠️ (部分) |
| **HellaSwag** | 常识推理 | 选择题 | ✅ |
| **BBH** | 复杂推理 | 自由回答 | ⚠️ (大模型) |
| **AGIEval** | 考试题 | 选择题 | ⚠️ (中文) |

**评估方式**：lm-evaluation-harness（EleutherAI 标准框架），一行命令跑完所有 benchmark。

### 当前项目实现

```
❌ 完全缺失。无任何 benchmark 评估。
```

### 差距影响

这是**和业界最大的差距**。没有 benchmark 意味着：

1. **无法检测灾难性遗忘**：蒸馏后通用能力是否退化？完全不知道。
2. **无法和其他蒸馏方案对比**：所有论文/开源项目都报 benchmark 分数，没有分数的模型无法被社区认可。
3. **无法验证"蒸馏 vs 从头训练"的收益**：benchmark 是衡量预训练知识保留的唯一客观标准。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P0** | 集成 `lm-evaluation-harness`，最少跑 **C-Eval**（中文核心） | ~30 行脚本 | 了解中文通用能力基线 |
| **P1** | 增加 **GSM8K**（推理）和 **HumanEval**（代码） | ~10 行配置 | AI Infra 领域需要推理和代码能力 |
| **P2** | 蒸馏前后自动跑 benchmark 并对比 △ | ~20 行 | 自动化灾难性遗忘检测 |

**最小可行实现**（P0）：

```bash
pip install lm_eval
python scripts/benchmark.py --model runs/train/exp5 --tasks ceval --num_fewshot 5
```

输出：
```
| Task    | Base (0-shot) | LoRA (0-shot) | Δ    | 状态 |
|---------|---------------|---------------|------|------|
| C-Eval  | 62.3          | 60.1          | -2.2 | ⚠️ 轻微退化 |
```

---

## 5. 灾难性遗忘检测

### 业界怎么做

```
蒸馏前：Base 模型跑 benchmark → baseline scores
蒸馏后：LoRA 模型跑 benchmark → distilled scores
对比：△ = distilled - baseline

判断：
  △ > +2%  → 正向迁移（罕见但可能，领域知识辅助通用推理）
  △ ∈ [-3%, +2%] → 正常范围，蒸馏未伤害通用能力
  △ ∈ [-10%, -3%] → 轻微遗忘，可接受但要优化
  △ < -10% → 严重灾难性遗忘，必须修复
```

**论文标准做法**：

| 论文 | 检测方法 |
|------|---------|
| **Orca** (2023) | BigBench-Hard + AGIEval + MMLU，报告 Base vs Orca-13B 全量对比 |
| **MiniLLM** (2024) | MMLU + GSM8K + HumanEval + DROP，蒸馏前后分项对比 |
| **Qwen 技术报告** | C-Eval + MMLU + HumanEval + GSM8K + BBH，展示各规格模型的完整能力矩阵 |

### 当前项目实现

```
❌ 完全缺失。蒸馏后通用能力未知。
```

### 差距影响

对生产部署是致命问题。一个 AI Infra 回答得很好但无法做基础数学推理的模型，在实际使用中会严重翻车。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P0** | 至少用 20 条通用中文问题做蒸馏前后 quick check | ~30 行 | 零成本检测严重遗忘 |
| **P1** | 跑 C-Eval benchmark 完整对比（依赖 #4 的 lm-evaluation-harness） | 依赖 #4 | 量化遗忘程度 |

**最小可行实现**（P0，不依赖 benchmark 框架）：

```python
# 蒸馏前后各跑 20 条通用问题（常识/数学/逻辑），对比 △
GENERAL_QUESTIONS = [
    "请解释什么是牛顿第二定律。",
    "计算 123 * 456 的结果。",
    "如果所有的猫都怕水，而 Tom 是一只猫，那么 Tom 怕水吗？为什么？",
    ...
]
```

---

## 6. 生成质量统计

### 业界怎么做

LLM 蒸馏论文通常报告以下生成统计：

| 指标 | 公式 | 含义 |
|------|------|------|
| **distinct-1 / distinct-2** | 不重复 unigram/bigram / 总 unigram/bigram | 生成多样性（越高越好） |
| **rep-n** | 重复 n-gram 的比例 | 模式崩溃检测（越低越好） |
| **平均生成长度** | mean(token_count) | 过长可能冗余，过短可能不完整 |
| **截断率** | 被 max_new_tokens 截断的比例 | 生成是否完整 |
| **Self-BLEU** | 生成样本之间的 BLEU | 多样性（越低越好） |

**典型发现**：蒸馏模型容易产生"模式崩溃"——针对不同问题都生成类似结构的回答。distinct-n 可以检测这个问题。

### 当前项目实现

```
❌ 无任何生成统计。仅保存逐条生成样本到报告。
```

### 差距影响

中等。人工抽查 5 条样本可以发现严重的模式崩溃，但无法量化。如果数据量大（50+ 条），人工很难发现细微的模板化问题。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P2** | 在 `collect_generation_samples()` 中增加 distinct-1/2 和平均长度统计 | ~15 行 | 低成本检测模式崩溃 |
| **P2** | 报告中增加"生成多样性"章节 | ~5 行 | 用户可快速判断 |

---

## 7. 数据质量验证

### 业界怎么做

蒸馏数据的质量直接决定模型上限：

```
数据质量验证 pipeline (参考 LIMA / Orca / Alpaca):

1. 格式验证
   ├── JSON/Parquet schema 完整性检查
   ├── 必填字段 (messages/response) 非空检查
   └── thinking 字段存在率统计

2. 内容质量
   ├── 语义去重 (embedding-based, 阈值 0.85)
   ├── 长度过滤 (min/max tokens)
   ├── 语言检测 (确保中文/英文一致)
   └── 模板化检测 (检测 "As an AI..." 等机械开头)

3. 多样性
   ├── 话题分布 (embedding 聚类 → 话题数)
   ├── 难度分层 (教师 perplexity 分桶)
   └── 指令类型分布 (问答/解释/代码/对比/分析)

4. 教师质量
   ├── 教师回答 PPL (过高 = 质量可疑)
   ├── 教师回答长度分布
   └── 多教师一致性 (同一问题的 GPT/Claude 回答差异)
```

### 当前项目实现

```yaml
# config.yaml — quality_filter 字段存在
quality_filter:
  enabled: true
  min_response_length: 50
  max_response_length: 4096
  skip_empty: true
  skip_duplicates: true
```

```python
# dataset.py — 仅加载 Parquet，无任何过滤逻辑
# 实际执行: 直接 load → format → return
# quality_filter 配置未接入数据加载流程！
```

**现实**：`config.yaml` 中虽然有 `quality_filter` 配置段，但 `dataset.py` **完全没有使用这些配置**。数据加载是裸 `load_dataset("parquet")`，无任何过滤。

### 差距影响

严重。如果训练数据中有低质量样本（教师回答过短、重复、格式错误），模型会学到噪声。由于 quality_filter 未接入代码，这些问题直接进入训练。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P0** | `dataset.py` 接入 quality_filter 配置：长度过滤 + 空回答过滤 + 精确去重 | ~20 行 | 配置已有，代码缺失，低代价高收益 |
| **P1** | 数据加载时输出质量统计：过滤前/后数量、长度分布、thinking 覆盖率 | ~15 行 | 知道数据长什么样才能调参 |
| **P2** | 语义去重（sentence-transformers embedding + cosine > 0.85） | ~30 行 | 高质量数据的关键 |

---

## 8. 师生差距分析

### 业界怎么做

```
蒸馏效果的多维度分析:

1. 难度分层 (Difficulty Stratification)
   ├── 按教师 PPL 分桶（低/中/高困惑度 = 简单/中等/困难问题）
   ├── 按问题长度分桶
   └── 每个桶分别报告 ROUGE / BERTScore
   → 发现：模型在简单问题上提升大，困难问题上可能反而退步

2. 知识类型分层
   ├── 事实型问题 (What is...?)
   ├── 解释型问题 (Explain why...)
   ├── 代码型问题 (Write code to...)
   ├── 对比型问题 (Compare A and B)
   └── 每个类型分别评估
   → 发现：蒸馏对不同知识类型的迁移效率不同

3. 输出分布对比
   ├── 生成长度分布 (teacher vs student)
   ├── 词汇重叠度分布
   └── 回答结构相似度
```

### 当前项目实现

```
❌ 无分层分析。所有样本统一计算 PPL/ROUGE 均值。
eval.py 中 Base 和 LoRA 的评估结果仅在报告末尾做整体对比。
```

### 差距影响

均值掩盖了重要信息。举例：
- 整体 ROUGE-L = 0.25
- 但简单问题 ROUGE-L = 0.40，困难问题 ROUGE-L = 0.05

如果不知道这个分层，可能误判模型质量。实际使用中用户更关心困难问题的表现。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P2** | 按生成长度分桶（短/中/长），每桶报告 PPL + ROUGE | ~20 行 | 快速发现短板 |
| **P3** | 按问题类型（事实/解释/代码）手工标注 20 条，分层评估 | 人工 1h | 定性理解模型能力分布 |

---

## 9. 消融实验框架

### 业界怎么做

蒸馏论文的标准实验章节：

```
消融实验矩阵 (Ablation Study):

1. 蒸馏温度: T=1 / 2 / 4 / 8
2. LoRA rank: r=4 / 8 / 16 / 32 / 64
3. 数据量: 100 / 500 / 1000 / 2000 / 5000 / 10000 条
4. 教师数量: 1 / 2 / 3 教师
5. Thinking 占比: 0% / 50% / 100%
6. Target modules: q+v / q+v+k+o / full
7. Epochs: 10 / 20 / 30 / 50

每组实验 → 自动记录到独立 exp dir → 汇总对比表
```

### 当前项目实现

```
有 YOLOv5 风格 exp dir (exp / exp2 / exp3 ...)，但无自动化消融框架。
每次实验需手动修改 config.yaml → 手动记录参数 → 手动对比结果。
train_results.json 可以辅助对比，但需要手动收集。
```

### 差距影响

轻微。当前阶段（原型验证）消融不是最紧迫的。但进入优化阶段后，手动对比多个实验会严重拖慢迭代速度。

### 改进建议

| 优先级 | 建议 | 工作量 | 原因 |
|--------|------|--------|------|
| **P3** | 脚本 `scripts/compare_experiments.py`：扫描所有 `runs/train/exp*/train_results.json`，生成对比表 | ~50 行 | 实验多了以后必备 |
| **P3** | 支持 `--sweep` 模式：自动遍历参数组合 | ~100 行 | 进入规模化优化阶段后再做 |

---

## 改进路线图

按优先级和依赖关系排列：

```
Phase 1 (本周, P0): 基础补齐
├── #2  增加 BERTScore 指标           (15 行, eval.py)
├── #4  集成 lm-evaluation-harness   (30 行, 新 scripts/benchmark.py)
├── #5  蒸馏前后 quick check (20 条)  (30 行, eval.py)
├── #7  接入 quality_filter 到数据加载 (20 行, dataset.py)
└── #1  results.csv 增加 grad_norm/accuracy 列 (5 行, trainer.py)

Phase 2 (下周, P1): 质量提升
├── #1  report_to 可配置化 (WandB)    (10 行, config.yaml + trainer.py)
├── #2  ROUGE 拆分 R/P/F1 分别报告    (5 行, eval.py)
├── #3  LLM-Judge 增加分数标准差     (10 行, eval.py)
├── #7  数据加载时输出质量统计        (15 行, dataset.py)
└── #5  跑 C-Eval 完整对比           (依赖 Phase 1 #4)

Phase 3 (下月, P2-P3): 完善
├── #1  自动生成 results.png          (20 行, trainer.py)
├── #6  生成多样性统计 distinct-n     (15 行, eval.py)
├── #8  按生成长度分层评估            (20 行, eval.py)
├── #3  位置交换校准                  (20 行, eval.py)
└── #9  实验对比脚本                  (50 行, 新 scripts/)
```

---

## 总结：当前最紧急的 5 件事

| # | 问题 | 影响 | 代价 |
|---|------|------|------|
| 1 | **仅有 ROUGE-L 无 BERTScore** | 系统性地低估生成质量，改写被误判 | 15 行代码 |
| 2 | **无 benchmark 评估** | 不知道蒸馏是否伤害通用能力（灾难性遗忘） | 30 行 + pip install |
| 3 | **quality_filter 未接入代码** | 低质量数据直接进入训练，浪费算力 | 20 行代码 |
| 4 | **无蒸馏前后对比** | 无法量化蒸馏的实际增益 | 30 行代码 |
| 5 | **results.csv 缺 grad_norm/accuracy** | 训练异常时缺少诊断数据 | 5 行代码 |

> **核心原则**：在黑盒蒸馏中，"模型学到了什么" 比 "模型记住了什么" 更重要。当前评估体系偏向检测"记住"（PPL/ROUGE），缺少检测"学到"（BERTScore/Benchmark/泛化测试）的手段。

---

*文档生成时间: 2026-06-04 | 基于项目代码 commit `110f632` 的实际实现状态*
