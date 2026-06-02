# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PolyDistill** — 多教师知识蒸馏训练框架。统一调度 GPT、Claude、Gemini 等商业 API 教师模型，将它们的集体知识蒸馏到本地学生模型。

产出模型 **TinySage**：支持 `Qwen3-0.6B` / `Qwen3-1.7B-Base` 双规格学生模型，经多教师黑盒蒸馏 + LoRA SFT 微调，专注 AI Infra（音视频/流媒体/GPU/CUDA）领域。修改 `config.yaml` 中 `model_id` 即可切换。

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

# LoRA 合并导出为独立模型（TinySage-0.6B）
python scripts/export.py
python scripts/export.py --output ./my-tinysage
```

## Architecture

项目采用 5 层工业结构，依赖关系单向：`config ← dataset ← trainer ← eval ← scripts/train`

```
PolyDistill/
├── poly_distill/             # 训练框架核心代码
│   ├── __init__.py            # 框架入口，导出所有公共 API
│   ├── config.py              # Config 默认值 + load_config() + 环境初始化
│   ├── dataset.py             # 数据加载与 chat_template 格式化（依赖 config）
│   ├── trainer.py             # 模型加载、LoRA 配置、SFT 训练（依赖 config + dataset）
│   ├── eval.py                # PPL / ROUGE-L / 生成样本 评测（依赖 config）
│   ├── json_to_parquet.py     # JSON → Parquet 格式转换脚本
│   ├── teachers/              # GPT/Claude/Gemini API 适配器
│   │   └── __init__.py
│   └── aggregation/           # 多教师知识聚合
│       └── __init__.py
├── scripts/
│   └── train.py               # 入口 main：训练 → 推理对比 → 全量评测
├── config.yaml                # YAML 配置文件（覆盖 Config 默认值）
├── requirements.txt           # Python 依赖清单（Python ≥ 3.10）
├── data/                      # 训练数据目录（仅加载 Parquet；JSON 需先转换）
├── models/                    # 模型缓存目录
├── dataset_cache/             # HuggingFace datasets 缓存（自动生成）
├── img/                       # 流程图
├── eval_report.md             # 评测报告（自动生成）
└── eval_results.json          # 结构化评测结果（自动生成）
```

### 各层职责

| 层 | 文件 | 核心职责 |
|----|------|---------|
| 配置层 | `config.py` + `config.yaml` | `Config` class 提供默认值；`config.yaml` 覆盖；`load_config()` 合并两者 |
| 数据层 | `dataset.py` | 仅加载 Parquet 文件 → chat_template 格式化 → thinking+response 拼为 assistant content → 仅保留 "text" |
| 训练层 | `trainer.py` | `train(config)` → 加载模型(BF16) → 数据集 → DataCollatorForCompletionOnlyLM → LoRA(r=8) → SFTTrainer → 保存 adapter |
| 评测层 | `eval.py` | `run_evaluation(config, tokenizer)` → PPL / ROUGE-L / 生成样本对比 → 输出 `eval_report.md` + `eval_results.json` |
| 入口层 | `scripts/train.py` | `__main__` 串联全流程；支持 `--eval-only` / `--skip-eval` 模式 |

### 数据流

```
Parquet 文件 (reasoning-distill schema: messages/thinking/response/system)
  → dataset.py 加载 Parquet → _format_conversation() 拼装对话:
      system + user + assistant(thinking + "\n\n" + response)
  → tokenizer.apply_chat_template() → "text" 字段
  → DataCollatorForCompletionOnlyLM 仅对 assistant 部分计算 loss
  → SFTTrainer + LoRA (r=8, target: q_proj/v_proj)
  → 保存 LoRA adapter 到 ./lora_sft_ai_infra_audio_video_output/
```

### 关键设计决策

| 决策 | 理由 |
|------|------|
| 5 层分离 | 配置/数据/训练/评测/推理 独立，修改一处不影响其他层 |
| tokenizer 通过参数传递（非 global） | 避免闭包依赖全局变量，函数签名更清晰 |
| `train()` 返回 `(trainer, tokenizer)` | inference 层可复用 tokenizer，避免重复加载 |
| `setup_environment()` 在入口调用 | 环境变量必须在 import torch 前设置；入口处调用一次即可 |
| 配置优先级: YAML > Config 默认值 | 改参数只需编辑 `config.yaml`，无需改 Python 代码；PyYAML 未安装时静默回退 |
| `--config` CLI 参数 | 支持多环境配置（dev.yaml / prod.yaml） |
| 单卡，禁用 DDP | `_reset_ddp_env()` 清理残留分布式环境变量，防止误触发 |
| 仅训练 `q_proj/v_proj` | ~2M 可训练参数，降低过拟合风险 |
| HF 镜像 `hf-mirror.com` | 国内加速下载模型和数据集 |
| 训练强制 Parquet 格式 | reasoning-distill schema（messages/thinking/response 列）；JSON 需先 `python poly_distill/json_to_parquet.py` 转换 |
| 评测 3 维度: PPL + ROUGE-L + 生成样本 | PPL 衡量拟合度，ROUGE-L 衡量内容重叠，生成样本供人工判断 |
| ROUGE-L 自实现（逐字 LCS） | 避免引入 rouge-score 等额外依赖，中文逐字比较无需分词 |
| 评测固定随机种子 42 | 确保每次评测抽取相同样本，结果可复现 |
| `--eval-only` 模式 | 可在不重新训练的情况下反复评测，快速迭代 prompt/参数 |
| Early Stopping + Best Checkpoint | patience=10 epoch 内 eval_loss 未改善则自动停止，加载最优权重 |
| NEFTune 噪声注入 | embedding 层注入均匀噪声（alpha=5），显著提升小数据集指令微调泛化性 |
| Train/Val Split 90/10 | 从全量数据划分验证集，支持早停和过拟合检测 |
| 日志系统 | Python logging 标准库，`setup_logging()` 在 `setup_environment()` 中自动调用；同时输出到控制台和 `train.log` 文件；所有模块通过 `logging.getLogger(__name__)` 获取 logger |

### 训练参数速查

| 参数 | 值 |
|------|-----|
| 基座模型 | `Qwen/Qwen3-0.6B`（可选 `Qwen/Qwen3-1.7B-Base`） |
| LoRA | r=8, alpha=16, dropout=0.1, target=(q_proj, v_proj) |
| Effective batch | 4 × 8 = 32 |
| LR | 2e-4, cosine + 3% warmup |
| Epochs | 100（配合 early stopping patience=10） |
| 正则化 | weight_decay=0.01, max_grad_norm=1.0, NEFTune=5 |
| Train/Val split | 90/10（seed=42） |
| 精度 | BF16 (训练) / FP16 (推理对比) |
| GPU | 单卡 Ampere+（BF16 需要） |
