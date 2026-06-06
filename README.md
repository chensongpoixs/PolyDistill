# PolyDistill / TinySage

# TinySage: 多教师蒸馏小语言模型

TinySage 是基于 Qwen3 家族的多规格小模型，使用自研蒸馏训练框架 **PolyDistill** 统一调度 GPT、Claude、Gemini 等商业 API 教师模型与学生模型的训练。支持 0.6B / 1.7B / 4B / 8B 四规格，修改 config.yaml 中 model_id 即可切换。

### 学生模型规格对比

| 规格 | 参数量 | 层数 | hidden_size | BF16 显存（训练估算） | 推荐 GPU | 适用场景 |
|------|--------|------|-------------|----------------------|---------|---------|
| [**TinySage-0.6B**](https://www.modelscope.cn/models/chensongpoixs/TinySage-0.8B) | 0.6B | 28 | 1024 | ~1.2 GB | 8GB+ | 端侧部署 / 快速原型 / 实时推理 |
| [**TinySage-1.7B**]() | 1.7B | 28 | 2048 | ~3.4 GB | 16GB+ | 服务器部署 / 较高回答质量 |
| [**TinySage-4B**](https://www.modelscope.cn/models/chensongpoixs/TinySage-4B) | 4B | 36 | 2560 | ~7.6 GB | 16GB+ | 平衡质量与速度，性价比之选 |
| [**TinySage-8B**]() | 8B | 36 | 4096 | ~16 GB | 24GB+ | 接近生产级效果，强推理能力 |

> 注：表中 BF16 显存为单卡训练中对模型参数占用的经验估算（不含激活值、优化器状态）。训练时总显存需求会显著高于此估算（需加上激活值、梯度、优化器状态和额外内存开销）。
>
> 训练峰值估算（示例条件：单卡、BF16、seq_len=2048、batch_size=4、无 activation_checkpoint）：
> - TinySage-0.6B — 约 4 GB
> - TinySage-1.7B — 约 8 GB
> - TinySage-4B   — 约 14 GB
> - TinySage-8B   — 约 22 GB
>
> 实际显存峰值会受 batch_size、seq_len、activation_checkpoint、optimizer（如 AdamW）和并行配置影响，请按实际配置与测试为准。

---

PolyDistill 是一个面向「多教师黑盒知识蒸馏」的训练框架，能够统一调度 GPT、Claude、Gemini 等商业大模型作为教师，把它们的集体知识蒸馏到本地可部署的小模型（TinySage 系列：0.6B / 1.7B / 4B / 8B）。此 README 以简体中文为主，包含项目亮点、快速上手、目录结构、配置与导出流程等必要信息。

---

## 一句话说明

将多家大模型的回答作为教师，以聚合和清洗策略生成高质量训练目标，通过 LoRA/SFT/蒸馏策略把能力迁移到低成本可部署的 TinySage 小模型。

---

## 主要特性

- 多教师黑盒蒸馏（GPT / Claude / Gemini 等）
- 多教师回答聚合与去噪（投票、加权融合、过滤）
- 支持 logits 蒸馏、特征层蒸馏、指令蒸馏、LoRA/SFT 微调等多种策略
- 流式/增量数据处理以支持大规模蒸馏语料
- 支持导出为独立模型（合并 LoRA）以便部署

---

## 适合谁用

- 想把闭源/商业大模型能力迁移到本地的小模型上
- 在资源受限环境需要可部署、低成本模型
- 希望为特定领域（如 AI Infra、音视频、GPU/CUDA）训练定制化小模型

---

## 快速开始（推荐 Python ≥ 3.10）

克隆并进入仓库：

```bash
git clone https://github.com/chensongpoixs/PolyDistill.git
cd PolyDistill
```

安装依赖：

```bash
pip install -r requirements.txt
```

数据准备（若数据为 JSON，先转为 Parquet）：

```bash
python poly_distill/json_to_parquet.py --input ./data --output ./data
```

运行完整流水线（训练 → 推理对比 → 全量评测 → 报告）：

```bash
python scripts/train.py
```

常用命令：
- 指定配置文件：`python scripts/train.py --config prod.yaml`
- 跳过全量评测：`python scripts/train.py --skip-eval`
- 仅评测已有模型：`python scripts/train.py --eval-only`
- 单独运行评测：`python poly_distill/eval.py`

---

## 导出（合并 LoRA 为独立模型）

导出已训练的 LoRA adapter 为可独立分发的 TinySage 模型：

```bash
python scripts/export.py                # 自动探测最新实验并导出
python scripts/export.py --list         # 列出可导出实验
python scripts/export.py --exp exp3     # 导出指定实验
python scripts/export.py --adapter ./runs/train/exp3 --output ./my-tinysage
```

导出后，模型位于 `./models/TinySage-{规格}`，可像普通 transformers 模型加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./models/TinySage-0.6B", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("./models/TinySage-0.6B")
```

---

## 项目结构（概览）

```
PolyDistill/
├── poly_distill/             # 框架核心代码
│   ├── config.py             # 默认配置、load_config
│   ├── dataset.py            # 数据加载、quality_filter、模板化
│   ├── trainer.py            # 训练主流程、LoRA/SFT、检查点管理
│   ├── eval.py               # 多维度评估（PPL/BERTScore/LLM-Judge 等）
│   ├── json_to_parquet.py    # JSON -> Parquet
│   ├── llm_client.py         # OpenAI-compatible LLM 客户端封装
│   ├── teachers/             # 各商业教师的适配器（GPT/Claude/Gemini）
│   └── aggregation/          # 多教师输出聚合策略
├── scripts/                  # CLI：train.py / export.py / benchmark.py
├── config.yaml               # 用户覆盖默认配置的 YAML
├── requirements.txt
├── data/                     # Parquet 格式训练数据
├── models/                   # 导出模型目录
└── runs/train/exp{N}/        # 实验目录（adapter + 日志 + 报告）
```

---

## 重要概念

- 教师（Teacher）：外部闭源/商业大模型，通过 API 生成参考回答。
- 学生（Student）：本地小模型（TinySage），通过蒸馏/微调学习教师知识。
- 聚合（Aggregation）：对多教师输出进行清洗、融合，形成更可靠训练目标。
- 蒸馏策略（Distill Strategy）：logits、特征层、指令级等，可在配置中选择。

---

## 配置说明

默认配置位于 `poly_distill/config.py`，用户可通过根目录 `config.yaml` 覆盖常用字段。常见项：

- model_id: 基座模型 ID（例如 "Qwen/Qwen3-0.6B"）
- training: num_train_epochs / learning_rate / per_device_batch_size
- teacher_apis: 各教师（GPT/Claude/Gemini）的 API 与 Key 配置
- distill_strategy: logits / feature / instruction
- eval: 各评测维度开关与 LLM-as-Judge 配置

示例片段（config.yaml）：

```yaml
model_id: "Qwen/Qwen3-0.6B"
training:
  num_train_epochs: 100
  learning_rate: 2e-4
  per_device_batch_size: 4
```

---

## 评估与 Benchmark

内置评估维度（PPL、ROUGE-L、BERTScore、生成样本对比、通用能力、LLM-Judge）。常用命令：

```bash
python scripts/benchmark.py               # 完整 Benchmark（如 C-Eval）
python scripts/benchmark.py --limit 20    # 快速验证少量题目
python poly_distill/eval.py               # 运行评测模块
```

---

## 环境与可选依赖

