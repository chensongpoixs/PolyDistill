# P0 级问题开发实现文档

> 基于 `DISTILL_VALIDATION_GUIDE.md` 差距分析，5 个 P0 问题已全部实现。
> **状态**: ✅ 全部完成 (2026-06-04) | **增强**: ✅ 完成 (2026-06-05)

---

## 实现进度总览

| # | 问题 | 改动文件 | 状态 | 完成时间 |
|---|------|---------|------|---------|
| P0-5 | results.csv 增加 grad_norm + accuracy | `poly_distill/trainer.py` | ✅ 已完成 | 2026-06-04 |
| P0-3 | quality_filter 接入数据加载 | `poly_distill/dataset.py` + `config.py` | ✅ 已完成 | 2026-06-04 |
| P0-1 | BERTScore 语义评估指标 | `poly_distill/eval.py` | ✅ 已完成 | 2026-06-04 |
| P0-4 | 蒸馏前后对比评估 | `poly_distill/eval.py` | ✅ 已完成 | 2026-06-04 |
| P0-2 | Benchmark 评测集成 | `scripts/benchmark.py` (新建) | ✅ 已完成 | 2026-06-04 |

### 增强功能（2026-06-05）

| # | 功能 | 改动文件 | 状态 |
|---|------|---------|------|
| E1 | 评估维度 6 项独立开关 | `poly_distill/eval.py` + `config.py` + `config.yaml` | ✅ 已完成 |
| E2 | LLM-as-Judge 5 维度三方对比 Prompt | `poly_distill/eval.py` | ✅ 已完成 |
| E3 | 模型导出自动探测 + 智能命名 | `scripts/export.py` | ✅ 已完成 |
| E4 | --eval-only adapter 自动探测 | `poly_distill/eval.py` | ✅ 已完成 |
| E5 | eval.py 直接运行路径修复 | `poly_distill/eval.py` | ✅ 已完成 |

---

## P0-5: results.csv 增加 grad_norm 和 accuracy 列

### 原理

```
当前 results.csv:
  epoch,train_loss,eval_loss,lr,elapsed_seconds

训练异常诊断时:
  - loss=0 → 可能是 DataCollator 问题
  - 但 loss=0 + mean_token_accuracy=0 → 确认是 DataCollator（labels全mask）
  - 但 loss=0 + mean_token_accuracy>0 → 可能是模型的权重NaN

  - grad_norm=nan → 梯度爆炸（数值问题）
  - grad_norm 逐步增大 → 可能需要降低 lr
  - grad_norm 逐步减小 → 训练可能即将收敛
```

### 实现详情

**文件**: `poly_distill/trainer.py`

- `YOLOStyleProgressCallback.__init__`: 新增 `self._grad_norm = None`, `self._mean_token_accuracy = None`
- `YOLOStyleProgressCallback.on_log`: 从 `logs` 中捕获 `grad_norm` 和 `mean_token_accuracy`
- `YOLOStyleProgressCallback.on_evaluate`: `_eval_history.append` 中增加这两个字段
- `_save_eval_report` CSV 写入列: `epoch,train_loss,eval_loss,grad_norm,mean_token_accuracy,lr,elapsed_seconds`

### 验证

```bash
# 训练后查看 results.csv
head runs/train/exp{N}/results.csv
# 预期: 包含 grad_norm 和 mean_token_accuracy 列
```

---

## P0-3: quality_filter 接入数据加载

### 原理

```
蒸馏数据质量 = 蒸馏效果的上限 (Garbage In → Garbage Out)

黑盒蒸馏场景下，数据来自教师 API 调用，存在:
  1. API 超时/错误 → 空回答
  2. 教师模型"偷懒" → 过短回答 ("是的。")
  3. 数据预处理 bug → 重复样本
  4. 教师回答过长被截断 → 不完整回答
```

### 实现详情

**文件**: `poly_distill/dataset.py` (第 22-121 行) + `poly_distill/config.py`

`_apply_quality_filter()` 函数 — 4 步过滤：

| 步骤 | 规则 | 动作 |
|------|------|------|
| Step 1 | `skip_empty`: response 为空字符串 | 丢弃 |
| Step 2 | `min_response_length`: len(response) < 阈值 | 丢弃 |
| Step 3 | `max_response_length`: len(response) > 阈值 | 截断至阈值 |
| Step 4 | `skip_duplicates`: response 内容完全相同 | 仅保留第一条（hash 去重） |

Config 新增字段（`config.py`）:
- `QUALITY_FILTER_ENABLED` ← `distillation.quality_filter.enabled`
- `QUALITY_FILTER_MIN_RESPONSE_LENGTH` ← `distillation.quality_filter.min_response_length`
- `QUALITY_FILTER_MAX_RESPONSE_LENGTH` ← `distillation.quality_filter.max_response_length`
- `QUALITY_FILTER_SKIP_EMPTY` ← `distillation.quality_filter.skip_empty`
- `QUALITY_FILTER_SKIP_DUPLICATES` ← `distillation.quality_filter.skip_duplicates`

调用位置: `load_and_prepare_data()` 中，在 `concatenate_datasets` 之后、chat_template map 之前。

### 验证

```python
# 手动构造脏数据测试
test_data = [
    {"response": ""},                    # 应被过滤 (空)
    {"response": "OK"},                  # 应被过滤 (过短)
    {"response": "A" * 5000},            # 应被截断
    {"response": "正常回答1"},
    {"response": "正常回答1"},            # 应被去重
    {"response": "正常回答2"},
]
# 预期: 过滤后保留 3 条 (正常1 + 截断后的长回答 + 正常2)
```

---

## P0-1: BERTScore 语义评估指标

### 原理

**为什么 ROUGE-L 不够**

```
ROUGE-L = LCS(生成, 参考) / len(参考)   ← 最长公共子序列，逐字严格匹配

问题示例:
  参考答案: "FFmpeg 使用 -c:v libx264 -preset slow 进行 H.264 编码"
  LoRA生成: "可以通过 ffmpeg -c:v libx264 -preset veryslow 启用 x264 编码器"

  ROUGE-L F1 ≈ 0.15  → ❌ 字面重叠低，误判为差
  BERTScore   ≈ 0.92  → ✅ 语义高度相似，正确判定
```

**BERTScore 原理**

```
1. 将参考答案和生成文本分别通过 BERT 编码为 token embeddings
   ref_embeds  = BERT(reference)  → [N, 768]
   cand_embeds = BERT(candidate)  → [M, 768]

2. 对候选文本的每个 token，找参考答案中最相似的 token（余弦相似度）
   Recall = mean(max_{j}(cos(cand_i, ref_j)) for each ref_i)

3. 对参考答案的每个 token，找候选文本中最相似的 token
   Precision = mean(max_{i}(cos(ref_j, cand_i)) for each cand_j)

4. F1 = 2 * R * P / (R + P)
```

### 实现详情

**文件**: `poly_distill/eval.py` (第 32-38 行导入, 第 260-361 行函数, 报告体)

`evaluate_bertscore()` 函数:
- 对每个样本: model.generate() → 收集 refs + cands
- 调用 `bert_score.score(cands, refs, model_type="bert-base-chinese", lang="zh", batch_size=16)`
- 返回 `{"label", "bertscore_f1", "bertscore_precision", "bertscore_recall", "num_samples"}`
- 未安装 `bert-score` 时静默回退，返回 `{"label", "enabled": False}`

报告集成 (`generate_report()`):
- 新增 **2.5 BERTScore 语义相似度对比** 章节（介于 ROUGE 和生成样本之间）
- 对比表含 F1 / Precision / Recall / 样本数
- `json_results["bertscore"]` 字段
- 终端摘要输出 BERTScore

### 验证

```python
# 单元测试: 两个语义相同但字面不同的句子
ref = "FFmpeg 使用 -c:v libx264 进行 H.264 编码"
cand = "可以通过 ffmpeg -c:v libx264 启用 x264 编码器"

# 预期: ROUGE-L F1 < 0.2, BERTScore F1 > 0.85
```

---

## P0-4: 蒸馏前后对比评估

### 原理

知识蒸馏的本质是"学生模型学习教师模型的知识"。但学习过程中可能发生灾难性遗忘:

```
预训练知识 (通用)
  MMLU: 65%  C-Eval: 62%  GSM8K: 45%
     ↓ LoRA SFT (领域知识注入)
领域知识 ↑ (ROUGE/BERTScore 提升)
通用知识 ? (未知！没有 benchmark 检测不到)
```

**对比维度**:
1. **领域能力** (PPL/ROUGE/BERTScore) — 预期显著提升
2. **通用能力** (20 道跨领域问题) — 预期基本持平
3. **综合判断** — 自动 PASS/WARNING/FAIL

### 实现详情

**文件**: `poly_distill/eval.py`

**20 道通用问题集** (第 368-400 行):
| 类别 | 题数 | 示例 |
|------|------|------|
| 数学推理 | 4 | "计算 123 × 456 的结果" |
| 科学常识 | 4 | "请解释什么是牛顿第二定律" |
| 逻辑推理 | 4 | "如果所有的猫都怕水，Tom 是一只猫..." |
| 中文能力 | 4 | "请用'春风'为题写一首五言绝句" |
| 代码能力 | 4 | "用 Python 实现斐波那契数列的前 20 项" |

**`evaluate_general_ability()`** 函数 (第 403-490 行):
- 对每道题 model.generate() 生成回答
- 统计: avg_length, min/max_length, truncation_rate
- 按类别 (math/science/logic/chinese/code) 分组统计
- 无参考答案时不自动评分

**报告集成** (`generate_report()`):
- 新增 **4.5 通用能力评估（灾难性遗忘检测）** 章节 — 总体对比表 + 按类别对比表
- 重写 **5. 蒸馏效果综合判断** 章节 — 逐项动态检查 + 自动 PASS/WARNING/FAIL 判定:
  - PPL 变化（显著下降/小幅下降/未下降）
  - ROUGE-L 阈值（>0.3 / >0.15 / <0.15）
  - BERTScore 变化（提升/持平/下降）
  - 通用能力长度变化（<20% / 20-30% / >30%）
  - 截断率上升检测
- `json_results["general_ability"]` 字段
- 终端摘要输出 General 结果

### 综合判定逻辑

```
有 ❌ → FAIL — 存在严重问题，建议检查训练数据和参数后重训
有 ⚠️ → PASS WITH WARNINGS — 蒸馏基本成功，但有些指标需要关注
无 ❌/⚠️ → PASS — 领域能力显著提升，通用能力无明显退化
```

### 验证

```bash
# 跑完整评估流程，检查报告中 "综合判断" 章节是否存在
python poly_distill/eval.py
# 预期: eval_report.md 包含 4.5 通用能力评估 + 5 蒸馏效果综合判断 + 自动 PASS/WARNING/FAIL 结论
```

---

## P0-2: Benchmark 评测集成

### 原理

**lm-evaluation-harness** (EleutherAI 标准化评估框架):

```
1. 加载模型 (HF format)
2. 对每个 benchmark task:
   a. 加载标准测试集 (如 C-Eval 的 1,346 道选择题)
   b. 按 task 定义的 prompt 格式构造输入
   c. 模型推理 → 取 logits → 选最高概率选项
   d. 计算 accuracy
3. 输出结构化结果
```

### 实现详情

**文件**: `scripts/benchmark.py` (新建，~280 行)

核心函数:
- `run_benchmark()` — 调用 `lm_eval.simple_evaluate(model="hf", model_args=..., tasks=...)`
  - Base: `model_args = "pretrained={model_id},dtype=float16,trust_remote_code=True"`
  - LoRA: 追加 `,peft={lora_path}` — lm_eval 原生支持 PEFT
- `_build_comparison_table()` — 构造 Base vs LoRA 对比表
- `generate_benchmark_report()` — 生成 Markdown 报告 + 综合判定

CLI 参数:
| 参数 | 说明 |
|------|------|
| `--tasks ceval` | 评估任务（逗号分隔） |
| `--limit 20` | 每个 task 限制 N 题（快速验证） |
| `--base-only` | 仅跑 Base 模型 |
| `--lora-only` | 仅跑 LoRA 模型 |
| `--output-dir` | 输出目录（默认自动检测最新 exp{N}） |

支持的任务:
| 别名 | lm_eval task | 题目数 | 领域 |
|------|-------------|--------|------|
| `ceval` | `ceval-valid` | 1,346 | 中文多学科 |
| `mmlu` | `mmlu` | ~14K | 英文多学科 |
| `gsm8k` | `gsm8k` | 1,319 | 数学推理 |
| `hellaswag` | `hellaswag` | 10K | 常识推理 |

输出文件:
- `{output_dir}/benchmark_results.json` — 结构化原始数据
- `{output_dir}/benchmark_report.md` — 可读对比报告

### 验证

```bash
# 快速验证（20 题）
python scripts/benchmark.py --tasks ceval --limit 20

# 完整 C-Eval
python scripts/benchmark.py

# 多 benchmark
python scripts/benchmark.py --tasks ceval,mmlu --limit 100

# 预期: 正常输出 Base/LoRA 对比分数，生成 benchmark_report.md
```

---

## E1: 评估维度 6 项独立开关

### 原理

原先评估维度硬编码全部执行，无法按需关闭。现支持 `config.yaml` 中逐项开关：

```yaml
eval:
  ppl:
    enabled: true            # PPL 困惑度
  rouge:
    enabled: false           # ROUGE-L（默认关闭，字面匹配对改写不友好）
  bertscore:
    enabled: true            # BERTScore 语义相似度
  gen_samples:
    enabled: true            # 生成样本对比
  general_ability:
    enabled: true            # 通用能力评估（灾难性遗忘检测）
  llm_judge:
    enabled: true            # LLM-as-Judge 大模型打分
```

### 实现详情

**文件**: `poly_distill/config.py` + `config.yaml` + `poly_distill/eval.py`

Config 类新增 6 个布尔字段（默认全开，ROUGE 默认关）：
```python
EVAL_PPL_ENABLED: bool = True
EVAL_ROUGE_ENABLED: bool = False
EVAL_BERTSCORE_ENABLED: bool = True
EVAL_GEN_SAMPLES_ENABLED: bool = True
EVAL_GENERAL_ABILITY_ENABLED: bool = True
EVAL_LLM_JUDGE_ENABLED: bool = True
```

`_FIELD_MAP` 映射路径:
- `eval.ppl.enabled` → `EVAL_PPL_ENABLED`
- `eval.rouge.enabled` → `EVAL_ROUGE_ENABLED`
- `eval.bertscore.enabled` → `EVAL_BERTSCORE_ENABLED`
- `eval.gen_samples.enabled` → `EVAL_GEN_SAMPLES_ENABLED`
- `eval.general_ability.enabled` → `EVAL_GENERAL_ABILITY_ENABLED`
- `eval.llm_judge.enabled` → `EVAL_LLM_JUDGE_ENABLED`

`run_evaluation()` 中每个维度用 `if config.EVAL_*_ENABLED:` 守卫，禁用时终端输出 `>>> XXX 评估: 已禁用 (eval.xxx.enabled=false)`。

报告集成：禁用维度在报告中显示 "已禁用" 说明。

---

## E2: LLM-as-Judge 5 维度三方对比 Prompt

### 原理

原 prompt 过于简单，仅单维度打分。新 `DISTILLATION_EVAL_PROMPT` 支持：

- **5 个评分维度**: 准确性 / 相关性 / 完整性 / 清晰度 / 整体质量（1-5 分制，含评分锚点）
- **三方对比**: Base（原始基座模型）vs LoRA（指令微调后）vs Teacher（教师参考答案）
- **输出字段**: `improvement_over_baseline`（相对基座提升值）+ `gap_to_teacher`（与教师差距）

### 实现详情

**文件**: `poly_distill/eval.py`

`DISTILLATION_EVAL_PROMPT` 常量（~80 行中文 prompt）包含：
- 角色设定：AI Infra 领域蒸馏质量评估专家
- 5 维度评分标准 + 1-5 分行为锚点
- 三方回答并排展示格式
- JSON 输出格式规范（含 improvement_over_baseline / gap_to_teacher）

`_build_judge_prompt(question, reference, generated, baseline)`:
- 接收 baseline（Base 模型回答）参数
- LoRA 评估时传入 baseline → 三方对比
- Base 评估时不传 baseline → 两方对比（Base vs Teacher）

`evaluate_with_llm_judge()`:
- 收集 `improvement_over_baseline` 和 `gap_to_teacher` 列表
- 计算 `improvement_summary`（平均提升）和 `gap_summary`（平均差距）

报告新增章节:
- "蒸馏增益分析" — improvement_over_baseline 统计
- "与教师差距" — gap_to_teacher 统计

---

## E3: 模型导出自动探测 + 智能命名

### 原理

原先 `scripts/export.py` 依赖硬编码路径，需手动指定 adapter 位置和输出名称。

现在：
- **自动探测 adapter**: 搜索 `runs/train/exp{N}/` 下最新包含 `adapter_config.json` 的目录
- **智能命名**: `Qwen/Qwen3-4B` → `TinySage-4B`（正则提取规格后缀）
- **多 CLI 支持**: `--list` / `--exp` / `--adapter` / `--output`

### 实现详情

**文件**: `scripts/export.py`（重度改写，~330 行）

核心函数：

| 函数 | 职责 |
|------|------|
| `_derive_model_name(model_id)` | 正则 `r"(\d+(?:\.\d+)?B)"` 提取规格 → `TinySage-{规格}` |
| `_derive_export_dir(config)` | 优先级: CLI > config.yaml > 自动推导 |
| `_detect_adapter_dir(config)` | 搜索 `runs/train/exp{N}/` 按 mtime 倒序找 adapter_config.json |
| `_list_experiments(config)` | 列出所有实验（mtime + adapter 状态 + 模型信息） |
| `_write_model_card(output_path, config)` | 动态生成模型卡片（适配模型规格） |

CLI 参数:

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径 |
| `--exp exp3` | 导出指定实验 |
| `--adapter ./path/to/adapter` | 手动指定 adapter 路径（与 --exp 互斥） |
| `--output ./models/my-tinysage` | 自定义输出路径 |
| `--list` | 列出所有可导出实验后退出 |

用法示例:
```bash
python scripts/export.py                     # 自动导出最新实验
python scripts/export.py --list              # 列出所有可导出实验
python scripts/export.py --exp exp3          # 导出指定实验
python scripts/export.py --output ./my-model # 自定义输出路径
```

Config 新增字段：
- `EXPORT_DIR` ← `export.output_dir`（留空 = 自动推导）

---

## E4: --eval-only adapter 自动探测

### 原理

训练时 `OUTPUT_DIR` 会被重定向到 `runs/train/exp{N}/`，但 `config.yaml` 中仍是旧路径。`--eval-only` 模式读取 config 中的 `output_dir` 找不到 adapter 导致崩溃。

### 实现详情

**文件**: `poly_distill/eval.py` (第 1254-1296 行)

`_detect_adapter_dir(config)` 函数，查找优先级：
1. `config.OUTPUT_DIR`（如果包含 `adapter_config.json`）
2. `runs/train/exp{N}/` 下最新包含 `adapter_config.json` 的目录（按 mtime 倒序）
3. 抛出 `FileNotFoundError` 并附带诊断信息

调用位置：`run_evaluation()` 中加载 LoRA 模型前（第 1398 行），替代原先直接使用 `config.OUTPUT_DIR`。

---

## E5: eval.py 直接运行路径修复

### 原理

`poly_distill/eval.py` 位于包内部，直接 `python poly_distill/eval.py` 运行时 Python 不会自动将项目根目录加入 `sys.path`，导致 `ModuleNotFoundError: No module named 'poly_distill'`。

### 实现详情

**文件**: `poly_distill/eval.py` (第 22, 28 行)

在 `from poly_distill.config import Config` 之前添加：
```python
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

与 `scripts/train.py` 和 `scripts/export.py` 的路径处理保持一致。

---

## 实现顺序与依赖

```
P0-5 (results.csv 列) ──→ P0-3 (quality_filter) ──→ P0-1 (BERTScore)
                              ↓
                         P0-4 (蒸馏前后对比)
                              ↓
                         P0-2 (Benchmark)

依赖关系:
  - P0-5: 无依赖，先做（最简单，改 1 个文件）✅
  - P0-3: 无依赖，改 dataset.py + config.py ✅
  - P0-1: 无依赖，改 eval.py（需要 pip install bert-score）✅
  - P0-4: 依赖 P0-1（报告中用到 BERTScore）✅
  - P0-2: 依赖 P0-4 的结论逻辑（最后的验证步骤）✅
```

---

## 评估流水线全景（当前）

```
python scripts/train.py
  │
  ├─ 1. 训练 (trainer.py)
  │     ├─ quality_filter (P0-3) — 数据清洗
  │     ├─ NaNDetectionCallback — 异常检测
  │     └─ YOLOStyleProgressCallback — results.csv (P0-5: grad_norm + accuracy)
  │
  └─ 2. 评测 (eval.py) — 每维度独立开关守卫 (E1)
        ├─ PPL 评估          ← eval.ppl.enabled
        ├─ ROUGE-L 评估       ← eval.rouge.enabled (默认关闭)
        ├─ BERTScore 评估     ← eval.bertscore.enabled
        ├─ 生成样本对比        ← eval.gen_samples.enabled
        ├─ 通用能力评估        ← eval.general_ability.enabled
        ├─ LLM-as-Judge       ← eval.llm_judge.enabled
        │     └─ DISTILLATION_EVAL_PROMPT (E2) — 5维度三方对比
        └─ 综合判断 — PASS/WARNING/FAIL

python scripts/export.py (E3)
  ├─ _detect_adapter_dir() — 自动探测最新 LoRA adapter
  ├─ _derive_model_name()  — Qwen3-4B → TinySage-4B
  ├─ merge_and_unload()    — LoRA 权重合并
  └─ _write_model_card()   — 动态模型卡片

python scripts/benchmark.py (P0-2)
  ├─ C-Eval / MMLU / GSM8K / HellaSwag
  └─ Base vs LoRA 通用能力定量对比
```

---

## 新增依赖

| 依赖 | 用途 | 安装 |
|------|------|------|
| `bert-score` | P0-1 BERTScore 语义评估 | `pip install bert-score` |
| `lm-eval` | P0-2 Benchmark 评测 | `pip install lm-eval` |

两者均为可选依赖，未安装时相应功能静默跳过。

---

*文档版本: v3.0 | 2026-06-05 | 5/5 P0 已实现 + 5 项增强*
