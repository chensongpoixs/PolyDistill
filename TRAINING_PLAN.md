# AI Infra 知识蒸馏 — 模型训练优化规划

> 目标：从"能跑通"到"工业级可用"，系统性地提升模型质量。
> 当前基线：Qwen2.5-0.5B + LoRA r=16，432 条数据，300 epochs。

---

## Phase 1：数据质量 → 决定模型上限

> 模型质量天花板由数据决定，训练技巧只能在数据质量基础上微调。

### 1.1 数据质量审计

| 检查项 | 方法 | 标准 |
|--------|------|------|
| 思考链准确性 | 人工抽查 20 条 `thinking` 字段 | 逻辑链完整、无事实错误 |
| 答案一致性 | `thinking` 结论是否与 `output` 一致 | 100% 一致 |
| 数据去重 | `instruction` 字段相似度去重 | 余弦相似度 > 0.95 视为重复 |
| 领域覆盖度 | 按 `type` 字段统计分布 | 每类 ≥ 10 条 |

```bash
# 快速统计数据类型分布
python -c "
import json, os, glob
from collections import Counter
types = Counter()
for f in glob.glob('data/**/*.json', recursive=True):
    for item in json.load(open(f)):
        types[item.get('type','unknown')] += 1
for t, n in types.most_common():
    print(f'{t}: {n}')
"
```

### 1.2 数据扩展策略（按优先级）

| 优先级 | 手段 | 预期增量 | 周期 |
|--------|------|---------|------|
| P0 | 人工编写高质量 QA（邀请领域专家） | +50~100 条 | 1 周 |
| P0 | 用 GPT-4/Claude 生成 QA，人工审核修正 | +200~500 条 | 3 天 |
| P1 | 从技术文档/论文提取 QA（RAG 辅助） | +300 条 | 1 周 |
| P1 | 对现有数据改写：同义替换、角度变换 | 2x 当前量 | 1 天 |
| P2 | 错误案例回注：收集模型答错的题，添加正确答案 | +50 条 | 迭代 |

### 1.3 数据质量标注

每条样本增加质量标记：
```json
{
  "instruction": "...",
  "thinking": "...",
  "output": "...",
  "quality": "gold"  // gold | silver | bronze
}
```

训练时优先使用 gold 数据，或用 quality 加权 loss。

---

## Phase 2：训练策略 → 把数据价值榨干

### 2.1 基座模型升级

| 模型 | 参数量 | 显存需求 | 预期效果 |
|------|--------|---------|---------|
| Qwen2.5-0.5B（当前） | 0.5B | ~3GB | 基线 |
| **Qwen2.5-1.5B** | 1.5B | ~8GB | ⬆ 显著提升 |
| Qwen2.5-3B | 3B | ~16GB | ⬆⬆ 需 24G 显存卡 |
| Qwen3-1.7B（thinking） | 1.7B | ~10GB | 原生思考链支持 |

> 推荐：优先升级到 Qwen2.5-1.5B，单张 4090 可跑，效果提升明显。

### 2.2 超参数搜索（Grid Search）

在 432 条数据上，关键超参数搜索空间：

```
┌─────────────────────┬──────────────────────┬────────────────┐
│ 参数                 │ 搜索范围              │ 推荐起步值      │
├─────────────────────┼──────────────────────┼────────────────┤
│ learning_rate        │ [1e-4, 2e-4, 5e-4]   │ 2e-4           │
│ LoRA rank (r)        │ [8, 16, 32, 64]      │ 16             │
│ LoRA target_modules  │ [q+v, q+k+v+o, all]  │ q_proj+v_proj  │
│ num_epochs           │ [50, 100, 200, 300]   │ 100            │
│ warmup_ratio         │ [0.03, 0.05, 0.1]     │ 0.05           │
│ dropout              │ [0.05, 0.1, 0.15]     │ 0.1            │
│ weight_decay         │ [0.0, 0.01, 0.05]     │ 0.01           │
└─────────────────────┴──────────────────────┴────────────────┘
```

执行策略：不要全量 Grid Search（组合爆炸），用 **渐进式搜索**：

```
Step 1: 固定其他参数，找最优 lr（3 组实验）
Step 2: 用最优 lr，找最优 rank + dropout（4 组实验）
Step 3: 用最优 lr + rank，找最优 epochs（3 组实验）
总计：~10 组实验即可收敛
```

### 2.3 训练技巧

| 技巧 | 原理 | 实现难度 |
|------|------|---------|
| **Early Stopping** | eval loss 不再下降时停止 | 低（TrainingArguments 内置） |
| **Gradient Clipping** | 防止梯度过大导致训练不稳定 | 低（`max_grad_norm=1.0`） |
| **NEFTune** | 训练时注入噪声 → 提升泛化 | 低（TRL 的 `neftune_noise_alpha=5`） |
| **Multi-stage Training** | 先训练思考链，再训练完整答案 | 中（需两阶段 config） |
| **Data Curriculum** | 从简单题 → 难题渐进训练 | 中（需按难度标注数据） |
| **LoRA+ (LoRA Plus)** | 为 A/B 矩阵设不同 lr | 低（PEFT 0.13+ 支持） |

### 2.4 防止过拟合

432 条数据训练 0.5B 模型，过拟合是最大敌人：

| 措施 | 配置 |
|------|------|
| 增大 dropout | `LORA_DROPOUT: 0.1` → `0.15` |
| 减少 epochs | `NUM_TRAIN_EPOCHS: 300` → `100`（配合 early stopping） |
| 增加 weight_decay | `weight_decay: 0.01` |
| NEFTune 噪声 | `neftune_noise_alpha: 5` |
| 减少 LoRA rank | `LORA_R: 16` → `8` |

---

## Phase 3：评估体系 → 量化"好"

### 3.1 评估指标矩阵

| 指标 | 衡量什么 | 好模型标准 |
|------|---------|-----------|
| **PPL gap** (train - test) | 过拟合程度 | < 5 |
| **ROUGE-L F1** | 与参考答案的重叠 | > 0.35 |
| **BERTScore F1** | 语义相似度 | > 0.75 |
| **LLM-as-Judge 综合分** | 准确性/完整性/结构 | > 7/10 |
| **人工评估 win rate** | 真实可用性 | LoRA > Base 显著 |

### 3.2 LLM-as-Judge 评分维度

用 GPT-4/Claude/DeepSeek 作为裁判，对每个回答打分（1-10）：

```yaml
评分维度:
  准确性: 技术事实是否正确，无幻觉
  完整性: 是否覆盖了问题的关键要点
  结构化: 回答是否有清晰的逻辑结构（面试官目的→思路→过程→答案）
  专业性: 术语使用是否准确，深度是否足够
  简洁性: 是否答即所问，无冗余
```

### 3.3 评测流程

```
训练完成
  → PPL 检测（train vs test gap < 5？）
  → ROUGE-L + BERTScore 自动评分
  → LLM-as-Judge 多维度打分（随机 30 题）
  → 人工抽查 10 题（重点关注低分题）
  → 输出评估报告
```

---

## Phase 4：迭代飞轮

```
          ┌──────────────┐
          │  训练模型     │
          └──────┬───────┘
                 │
    ┌────────────▼───────────┐
    │  评估 → 找出弱项       │
    │  (哪些 topic 答得差？)  │
    └────────────┬───────────┘
                 │
    ┌────────────▼───────────┐
    │  针对性改进：           │
    │  - 弱 topic 追加数据    │
    │  - 错误案例加正确答案   │
    │  - 调整超参数          │
    └────────────┬───────────┘
                 │
          ┌──────▼───────┐
          │  重新训练     │◄── 循环
          └──────────────┘
```

---

## 执行路线图

```
Week 1: 数据审计 + 清洗 + 扩展
  □ 统计 topic 分布，识别弱覆盖区
  □ 人工审核 20 条 thinking 质量
  □ 用 Claude/GPT-4 生成 200+ QA，人工审核

Week 2: 超参搜索 + 防过拟合
  □ 实现 early stopping + eval split
  □ 跑 10 组 Grid Search 实验
  □ 选出最优 baseline config

Week 3: 评估体系搭建
  □ 接入 LLM-as-Judge 评分
  □ 人工评估 50 题 win rate
  □ 输出首次完整评估报告

Week 4: 迭代优化
  □ 根据评估结果追加弱项数据
  □ 尝试 1.5B 模型
  □ 最终模型冻结 + 发布
```

---

## 立刻可做的最小改进（30 分钟内）

```yaml
# config.yaml 改动
training:
  num_train_epochs: 100         # 300 → 100
  weight_decay: 0.01            # 新增
  max_grad_norm: 1.0            # 新增梯度裁剪

lora:
  r: 8                          # 16 → 8（减半）
  dropout: 0.1                  # 0.05 → 0.1（翻倍）
```
