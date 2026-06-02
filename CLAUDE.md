# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Infra 领域知识蒸馏项目 — 使用 LoRA SFT 微调 `Qwen2.5-0.5B-Instruct`，将音视频/流媒体/GPU/CUDA 等 AI 基础设施领域的面试问答知识注入小模型。

## Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 完整流水线：训练 → 推理对比 → 全量评测 → 输出报告
python inference.py

# 指定配置文件
python inference.py --config prod.yaml

# 跳过全量评测（仅做单样本快速推理对比）
python inference.py --skip-eval

# 仅评测已有模型（不重新训练）
python inference.py --eval-only

# 单独运行评测
python eval.py
```

## Architecture

项目采用 5 层工业结构，依赖关系单向：`config ← dataset ← train ← eval ← inference`

```
ai_infra/
├── config.py         # Config 默认值 + load_config() + 环境初始化
├── config.yaml       # YAML 配置文件（覆盖 Config 默认值）
├── dataset.py        # 数据加载与 chat_template 格式化（依赖 config）
├── train.py          # 模型加载、LoRA 配置、SFT 训练（依赖 config + dataset）
├── eval.py           # PPL / ROUGE-L / 生成样本 评测（依赖 config）
├── inference.py      # 入口 main：训练 → 推理对比 → 全量评测（依赖 config + train + eval）
├── requirements.txt  # Python 依赖清单（Python ≥ 3.10）
├── data/             # 训练数据目录（自动加载所有 .json 文件并合并）
├── img/              # 流程图
├── eval_report.md    # 评测报告（自动生成）
└── eval_results.json # 结构化评测结果（自动生成）
```

### 各层职责

| 层 | 文件 | 核心职责 |
|----|------|---------|
| 配置层 | `config.py` + `config.yaml` | `Config` class 提供默认值；`config.yaml` 覆盖；`load_config()` 合并两者 |
| 数据层 | `dataset.py` | `load_and_prepare_data(config, tokenizer)` 加载 JSON → chat_template 格式化 → 仅保留 "text" |
| 训练层 | `train.py` | `train(config)` → 加载模型(BF16) → 数据集 → DataCollatorForCompletionOnlyLM → LoRA(r=16) → SFTTrainer → 保存 adapter |
| 评测层 | `eval.py` | `run_evaluation(config, tokenizer)` → PPL / ROUGE-L / 生成样本对比 → 输出 `eval_report.md` + `eval_results.json` |
| 入口层 | `inference.py` | `__main__` 串联全流程；支持 `--eval-only` / `--skip-eval` 模式 |

### 数据流

```
ai_infra_audio_video.json
  → dataset.py:_format_conversation() 使用 tokenizer.apply_chat_template
  → DataCollatorForCompletionOnlyLM 仅对 assistant 部分计算 loss
  → SFTTrainer + LoRA (r=16, target: q_proj/v_proj)
  → 保存 LoRA adapter 到 ./lora_sft_ai_infra_audio_video_output/
```

### 关键设计决策

| 决策 | 理由 |
|------|------|
| 4 层分离 | 配置/数据/训练/推理 独立，修改一处不影响其他层 |
| tokenizer 通过参数传递（非 global） | 避免闭包依赖全局变量，函数签名更清晰 |
| `train()` 返回 `(trainer, tokenizer)` | inference 层可复用 tokenizer，避免重复加载 |
| `setup_environment()` 在 `inference.py` 入口调用 | 环境变量必须在 import torch 前设置；入口处调用一次即可 |
| 配置优先级: YAML > Config 默认值 | 改参数只需编辑 `config.yaml`，无需改 Python 代码；PyYAML 未安装时静默回退 |
| `--config` CLI 参数 | 支持多环境配置（dev.yaml / prod.yaml） |
| 单卡，禁用 DDP | `_reset_ddp_env()` 清理残留分布式环境变量，防止误触发 |
| 仅训练 `q_proj/v_proj` | ~2M 可训练参数，降低过拟合风险 |
| HF 镜像 `hf-mirror.com` | 国内加速下载模型和数据集 |
| 评测 3 维度: PPL + ROUGE-L + 生成样本 | PPL 衡量拟合度，ROUGE-L 衡量内容重叠，生成样本供人工判断 |
| ROUGE-L 自实现（逐字 LCS） | 避免引入 rouge-score 等额外依赖，中文逐字比较无需分词 |
| 评测固定随机种子 42 | 确保每次评测抽取相同样本，结果可复现 |
| `--eval-only` 模式 | 可在不重新训练的情况下反复评测，快速迭代 prompt/参数 |

### 训练参数速查

| 参数 | 值 |
|------|-----|
| 基座模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| LoRA | r=16, alpha=32, dropout=0.05, target=(q_proj, v_proj) |
| Effective batch | 4 × 8 = 32 |
| LR | 2e-4, cosine + 3% warmup |
| Epochs | 300 |
| 精度 | BF16 (训练) / FP16 (推理对比) |
| GPU | 单卡 Ampere+（BF16 需要） |
