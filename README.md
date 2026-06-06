# PolyDistill / TinySage

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

## 适用人群

- 想把闭源/商业大模型能力迁移到本地的小模型上
- 在资源受限环境需要可部署、低成本模型
- 希望为特定领域（如 AI Infra、音视频、GPU/CUDA）训练定制化小模型

---

## 快速开始（推荐 Python ≥ 3.10）

1. 克隆仓库并进入目录

```bash
git clone https://github.com/chensongpoixs/PolyDistill.git
cd PolyDistill
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3.（若源数据为 JSON）先转换为 Parquet：

```bash
python poly_distill/json_to_parquet.py --input ./data --output ./data
```

4. 运行完整训练流水线（训练 → 推理对比 → 全量评测）

```bash
python scripts/train.py
```

常用命令：
- 指定配置文件：`python scripts/train.py --config prod.yaml`
- 跳过全量评测：`python scripts/train.py --skip-eval`
- 仅评测已有模型：`python scripts/train.py --eval-only`
- 单独运行评测：`python poly_distill/eval.py`

---

## 导出与部署

合并 LoRA 并导出为独立 TinySage 模型：

```bash
python scripts/export.py                # 自动探测并导出最新实验
python scripts/export.py --list         # 列出可导出实验
python scripts/export.py --exp exp3     # 导出指定实验
python scripts/export.py --adapter ./runs/train/exp3 --output ./my-tinysage
```

导出后模型位于 `./models/TinySage-{规格}`，可像普通 transformers 模型加载并部署（无需 PEFT）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./models/TinySage-0.6B", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("./models/TinySage-0.6B")
```

---

## 项目结构（概要）

```
PolyDistill/
├── poly_distill/             # 框架核心代码
│   ├── config.py             # 配置加载与默认值
│   ├── dataset.py            # 数据加载、清洗与模板化
│   ├── trainer.py            # 训练流程、LoRA/SFT、检查点
│   ├── eval.py               # 多维度评估（PPL/BERTScore/LLM-Judge 等）
│   ├── json_to_parquet.py    # JSON → Parquet 工具
│   ├── llm_client.py         # LLM 客户端适配层（OpenAI 兼容）
│   ├── teachers/             # 教师适配器（GPT/Claude/Gemini）
│   └── aggregation/          # 多教师回答聚合策略
├── scripts/                  # CLI 脚本：train.py / export.py / benchmark.py
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

默认配置位于 `poly_distill/config.py`，并可通过根目录的 `config.yaml` 覆盖。常见字段：

- model_id：基座模型 ID（例如 Qwen/Qwen3-0.6B）
- training：num_train_epochs / learning_rate / per_device_batch_size
- teacher_apis：教师模型 API 配置与密钥
- distill_strategy：蒸馏类型（logits/feature/instruction）
- eval：评估开关及 LLM-as-Judge 配置

示例 config.yaml 片段：

```yaml
model_id: "Qwen/Qwen3-0.6B"
training:
  num_train_epochs: 100
  learning_rate: 2e-4
  per_device_batch_size: 4
```

---

## 评估与 Benchmark

内置多维度评估（PPL、ROUGE-L、BERTScore、生成样本对比、通用能力、LLM-Judge）。常用命令：

```bash
python scripts/benchmark.py            # 完整 Benchmark（如 C-Eval）
python scripts/benchmark.py --limit 20 # 快速验证若干题目
python poly_distill/eval.py            # 直接运行评测模块
```

LLM-as-Judge 支持兼容 OpenAI API 的服务作为裁判来对比 Base / LoRA / Teacher，并输出结构化报告（eval_results.json）与可读报告（eval_report.md）。

---

## 环境与可选依赖

建议 Python >= 3.10，核心依赖在 requirements.txt。可选组件：
- flash-attn（显存/速度优化）
- bert-score（BERTScore 评估）
- lm-eval（Benchmark）
- openai（LLM-as-Judge）

安装基础依赖：

```bash
pip install -r requirements.txt
```

Flash Attention 等需按硬件/CUDA 版本选装，仓库内有说明。

---

## 数据格式

SFT/蒸馏用 JSON 示例：

```json
[
  {
    "instruction": "问题...",
    "input": "",
    "thinking": "可选的思考链/中间过程",
    "output": "教师最终回答"
  }
]
```

训练流程以 Parquet 为首选，使用 `json_to_parquet.py` 转换。

---

## 导出产物（runs/train/exp{N}/）

- adapter_model.safetensors — LoRA 权重
- adapter_config.json
- checkpoints/
- config_snapshot.yaml
- eval_report.md / eval_results.json

---

## 常见问题

Q: 数据需要什么格式？
A: 推荐 Parquet（reasoning-distill schema），若是 JSON 请先转换。

Q: 如何新增教师适配器？
A: 在 poly_distill/teachers/ 下添加模块，实现统一的 LLM 客户端接口（参考已有适配器）。

---

## 许可证

默认 MIT 许可证（请在仓库中维护 LICENSE 文件以确认）。

---

如果你需要：
- 英文版 README
- 更短的快速入门（1页）
- 针对首次搭建的详细配置示例（包含 API Key 环境变量与 config.yaml 完整样例）

告诉我你的偏好，我会继续按目标写出对应版本或提交 PR。