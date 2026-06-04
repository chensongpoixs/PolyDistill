# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PolyDistill** — 多教师知识蒸馏训练框架。统一调度 GPT、Claude、Gemini 等商业 API 教师模型，将它们的集体知识蒸馏到本地学生模型。

产出模型 **TinySage**：支持 `Qwen3-0.6B` / `Qwen3-1.7B` / `Qwen3-4B` / `Qwen3-8B` 四规格学生模型，经多教师黑盒蒸馏 + LoRA SFT 微调，专注 AI Infra（音视频/流媒体/GPU/CUDA）领域。修改 `config.yaml` 中 `model_id` 即可切换。

PolyDistill 核心能力：
- 跨架构商业大模型的黑盒蒸馏
- 多教师回答生成与融合
- 大规模教师语料的高效流式处理
- 灵活的训练策略（logit 层蒸馏、特征层蒸馏、指令蒸馏）

### Qwen2.5 → Qwen3 升级亮点

- **QK-Norm**: 移除 QKV 偏置，引入 QK 归一化，训练更稳定
- **思考/非思考双模式**: 单模型同时支持快速响应与深度推理，自动切换 + 思考预算控制
- **36T tokens 预训练** (Qwen2.5 为 18T)，多语言 29→119 种
- **四阶段后训练**: 长 CoT 冷启动→推理 RL→思维融合→通用 RL
- **On-policy + Off-policy 蒸馏**: 大模型高效带动小模型

## Commands

```bash
# 安装依赖
pip install -r requirements.txt

# JSON → Parquet 格式转换（首次）
python poly_distill/json_to_parquet.py --input ./data --output ./data

# 完整流水线：训练 → 推理对比 → 全量评测 → 输出报告
python scripts/train.py

# 指定配置文件
python scripts/train.py --config prod.yaml

# 跳过全量评测（仅做单样本快速推理对比）
python scripts/train.py --skip-eval

# 仅评测已有模型（不重新训练）
python scripts/train.py --eval-only

# 单独运行评测
python poly_distill/eval.py

# Benchmark 通用能力评测（灾难性遗忘检测）
python scripts/benchmark.py                          # 完整 C-Eval（1,346 题）
python scripts/benchmark.py --limit 20               # 快速验证 20 题
python scripts/benchmark.py --tasks ceval,mmlu       # 多 benchmark

# LoRA 合并导出为独立模型（TinySage）
python scripts/export.py                             # 自动导出最新实验 → ./models/TinySage-{规格}
python scripts/export.py --list                      # 列出所有可导出实验
python scripts/export.py --exp exp3                  # 导出指定实验
python scripts/export.py --adapter ./runs/train/exp  # 手动指定 adapter 路径
python scripts/export.py --output ./my-tinysage      # 自定义输出路径
```

## Architecture

项目采用 6 层工业结构，依赖关系单向：`config ← dataset ← trainer ← eval ← export ← scripts/train`

```
PolyDistill/
├── poly_distill/             # 训练框架核心代码
│   ├── __init__.py            # 框架入口，导出所有公共 API
│   ├── config.py              # Config 默认值 + load_config() + 环境初始化
│   ├── dataset.py             # 数据加载 + quality_filter + chat_template 格式化
│   ├── trainer.py             # 模型加载(BF16/SDPA/TF32) + LoRA + SFT + 实验目录
│   ├── eval.py                # 6维度评估: PPL/ROUGE-L/BERTScore/生成样本/通用能力/LLM-Judge(5维三方对比)
│   ├── llm_client.py          # OpenAI-compatible 公共客户端 (LLMClient)
│   ├── json_to_parquet.py     # JSON → Parquet 格式转换脚本
│   ├── teachers/              # GPT/Claude/Gemini API 适配器
│   │   └── __init__.py
│   └── aggregation/           # 多教师知识聚合
│       └── __init__.py
├── scripts/
│   ├── train.py               # 入口 main：训练 → 推理对比 → 全量评测
│   ├── export.py              # LoRA 合并导出 → TinySage 独立模型
│   └── benchmark.py           # C-Eval/MMLU 等标准化 benchmark 对比
├── config.yaml                # YAML 配置文件（覆盖 Config 默认值）
├── requirements.txt           # Python 依赖清单（Python ≥ 3.10）
├── data/                      # 训练数据目录（仅加载 Parquet；JSON 需先转换）
├── models/                    # 模型缓存目录 + TinySage 导出目录
├── runs/train/exp{N}/         # YOLOv5 风格实验目录（LoRA adapter + 日志 + 报告）
├── dataset_cache/             # HuggingFace datasets 缓存（自动生成）
├── img/                       # 流程图
└── docs/                      # 详细文档（验证指南 + P0 实现）
```

### 各层职责

| 层 | 文件 | 核心职责 |
|----|------|---------|
| 配置层 | `config.py` + `config.yaml` | `Config` class 提供默认值；`config.yaml` 覆盖；`load_config()` 合并两者 |
| 数据层 | `dataset.py` | Parquet 加载 → `_apply_quality_filter()` 质量过滤(空/短/长/重) → chat_template 格式化 → 仅保留 "text" |
| 训练层 | `trainer.py` | `train(config)` → 模型(BF16/SDPA/TF32) → 数据集 → DataCollatorForCompletionOnlyLM → LoRA → SFTTrainer → YOLOv5 实验目录 |
| 评测层 | `eval.py` | 6维度独立开关守卫 → PPL/ROUGE-L/BERTScore/生成样本/通用能力(灾难遗忘检测)/LLM-Judge(5维三方对比+improvement/gap) → 综合判定 PASS/WARNING/FAIL |
| 公共层 | `llm_client.py` | `LLMClient` 封装 OpenAI SDK — `chat()`/`chat_json()`，自动提取 base_url，支持 max_retries/timeout/top_p/seed |
| 导出层 | `scripts/export.py` | LoRA merge_and_unload → TinySage 独立模型 + 模型卡片；支持 --list/--exp/--adapter |
| Benchmark | `scripts/benchmark.py` | C-Eval/MMLU/GSM8K 标准化评测 → Base vs LoRA 通用能力定量对比 |
| 入口层 | `scripts/train.py` | `__main__` 串联全流程；支持 `--eval-only` / `--skip-eval` 模式 |

### 数据流

```
Parquet 文件 (reasoning-distill schema: messages/thinking/response/system)
  → dataset.py 加载 Parquet → _apply_quality_filter() 过滤空/短/长/重
  → _format_conversation() 拼装: system + user + assistant(reasoning_content + content)
  → tokenizer.apply_chat_template() → "text" 字段
  → DataCollatorForCompletionOnlyLM 仅对 assistant 部分计算 loss
  → SFTTrainer + LoRA → 保存到 runs/train/exp{N}/
  → scripts/export.py merge_and_unload → ./models/TinySage-{规格}/
```

### 关键设计决策

| 决策 | 理由 |
|------|------|
| 6 层分离 | 配置/数据/训练/评测/导出/Benchmark 独立，修改一处不影响其他层 |
| tokenizer 通过参数传递（非 global） | 避免闭包依赖全局变量，函数签名更清晰 |
| `train()` 返回 `(trainer, tokenizer)` | inference 层可复用 tokenizer，避免重复加载 |
| `setup_environment()` 在入口调用 | 环境变量必须在 import torch 前设置；入口处调用一次即可 |
| 配置优先级: YAML > Config 默认值 | 改参数只需编辑 `config.yaml`，无需改 Python 代码；PyYAML 未安装时静默回退 |
| `--config` CLI 参数 | 支持多环境配置（dev.yaml / prod.yaml） |
| 单卡，禁用 DDP | `_reset_ddp_env()` 清理残留分布式环境变量，防止误触发 |
| HF 镜像 `hf-mirror.com` | 国内加速下载模型和数据集 |
| 训练强制 Parquet 格式 | reasoning-distill schema（messages/thinking/response 列） |
| SDPA 替代 FA3（Blackwell） | FA3 用 FP8 中间精度，4B+ 模型 attention score 超范围 → NaN 梯度 |
| TF32 使能 | Ampere+ tensor cores 10-bit 尾数，8× BF16 精度，自动加速 matmul |
| quality_filter 4 步过滤 | 空回答/过短/过长截断/精确去重，保障训练数据质量 |
| BERTScore 语义评估 | BERT 向量余弦相似度，弥补 ROUGE-L 字面匹配对改写惩罚的缺陷 |
| 通用能力评估 (20题) | 数学/科学/逻辑/中文/代码 5 维度，检测灾难性遗忘 |
| 综合判定 PASS/WARNING/FAIL | 多指标联动自动判定蒸馏成败，无需人工逐项比对 |
| benchmark.py C-Eval 集成 | 标准化 1,346 题中文评测，定量验证通用能力保持 |
| YOLOv5 风格实验目录 | runs/train/exp{N}/ 自增编号，所有产物归入同一目录 |
| 评测固定随机种子 42 | 确保每次评测抽取相同样本，结果可复现 |
| `--eval-only` 模式 | 可不重新训练反复评测，快速迭代 prompt/参数 |
| Early Stopping + Best Checkpoint | patience 内 eval_loss 未改善则自动停止，加载最优权重 |
| NEFTune 噪声注入 | embedding 层注入均匀噪声（alpha=5），提升小数据集指令微调泛化性 |
| 日志系统 | Python logging 标准库，`%(filename)s:%(lineno)d` 格式定位源码位置；同时输出控制台和文件 |
| LLMClient 公共类 | 封装 OpenAI SDK，`chat()`/`chat_json()` 双接口；`LLMClient(endpoint, model, api_key, timeout, max_retries)`；后续 teachers/ 也可复用 |
| 6 维度独立开关 | `config.yaml` 中 `eval.{ppl,rouge,bertscore,gen_samples,general_ability,llm_judge}.enabled`；ROUGE 默认关闭（字面匹配对改写不友好） |
| LLM-Judge 5 维三方对比 | `DISTILLATION_EVAL_PROMPT`: 准确性/相关性/完整性/清晰度/整体(1-5分锚点)；Base vs LoRA vs Teacher；输出 improvement_over_baseline + gap_to_teacher |
| LLM-Judge HTTP 全配置化 | timeout(600s)/temperature(0.0)/max_tokens(4096)/top_p(1.0)/seed(42)/max_retries(2) 均在 config.yaml 配置 |
| LLM-Judge 依赖 gen_samples | sample 收集条件: `gen_samples.enabled or llm_judge.enabled`，任一开启即触发；n_show 自动同步 max_samples |
| 导出自动探测 | `_detect_adapter_dir()` 搜索 `runs/train/exp{N}/` 按 mtime 倒序；`_derive_model_name()` 正则提取规格后缀 → TinySage-{规格} |
| `--eval-only` adapter 探测 | eval.py 复用 `_detect_adapter_dir()`，不再依赖 config.yaml 中可能过时的 output_dir |

### 训练参数速查（当前: Qwen3-4B）

| 参数 | 值 |
|------|-----|
| 基座模型 | `Qwen/Qwen3-4B`（可选 0.6B/1.7B/8B） |
| LoRA | r=32, alpha=32, dropout=0.05, target=(q_proj, v_proj, k_proj, o_proj) |
| Attention | SDPA（Blackwell 用 FA3 会导致 NaN 梯度） |
| Effective batch | 1 × 8 = 8 |
| LR | 2e-4, cosine + 3% warmup |
| Epochs | 2（配合 early stopping patience=0，用 eval_split_ratio=0.1 验证） |
| 正则化 | weight_decay=0.05, max_grad_norm=1.0, NEFTune=5 |
| Train/Val split | 90/10（seed=42） |
| 精度 | BF16 (训练) / FP16 (推理/评测) |
| GPU | RTX 5080 16GB (Blackwell SM120) |
| DataLoader | num_workers=2, pin_memory=true, prefetch_factor=4 |
