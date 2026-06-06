# PolyDistill / TinySage

# TinySage: 多教师蒸馏小语言模型

TinySage 是基于 Qwen3 家族的多规格小模型，使用自研蒸馏训练框架 **PolyDistill** 统一调度 GPT、Claude、Gemini 等商业 API 教师模型与学生模型的训练。支持 0.6B / 1.7B / 4B / 8B 四规格，修改 config.yaml 中 model_id 即可切换。

### 学生模型规格对比

| 规格 | 参数量 | 层数 | hidden_size | BF16 参数占用（估算） | 训练峰值显存（估算，单卡 BF16） | 推荐 GPU | 适用场景 |
|------|--------|------|-------------|----------------------:|--------------------------------:|---------|---------|
| [**TinySage-0.6B**](https://www.modelscope.cn/models/chensongpoixs/TinySage-0.8B) | 0.6B | 28 | 1024 | ~1.2 GB | ~4 GB  | 8GB+  | 端侧部署 / 快速原型 / 实时推理 |
| [**TinySage-1.7B**]() | 1.7B | 28 | 2048 | ~3.4 GB | ~8 GB  | 16GB+ | 服务器部署 / 较高回答质量 |
| [**TinySage-4B**](https://www.modelscope.cn/models/chensongpoixs/TinySage-4B) | 4B | 36 | 2560 | ~7.6 GB | ~14 GB | 16GB+ | 平衡质量与速���，性价比之选 |
| [**TinySage-8B**]() | 8B | 36 | 4096 | ~16 GB  | ~22 GB | 24GB+ | 接近生产级效果，强推理能力 |

> 说明：
> - BF16 参数占用为仅模型参数在 BF16 下的估算（不含激活值、梯度、优化器状态）。
> - 训练峰值显存为经验估算，示例条件：单卡、BF16、seq_len=2048、batch_size=4、无 activation_checkpoint。实际显存需求会随 batch_size、seq_len、activation_checkpoint、optimizer（如 AdamW）和并行配置变化，请据实际测试为准。

---

PolyDistill 是一个面向「多教师黑盒知识蒸馏」的训练框架，能够统一调度 GPT、Claude、Gemini 等商业大模型作为教师，把它们的集体知识蒸馏到本地可部署的小模型（TinySage 系列：0.6B / 1.7B / 4B / 8B）。此 README 以简体中文为主，包含项目亮点、快速上手、目录结构、配置与导出流程等必要信息。

---

## 一句话说明

将多家大模型的回答作为教师，以聚合和清洗策略生成高质量训练目标，通过 LoRA/SFT/蒸馏策略把能力迁移到低成本可部署的 TinySage 小模型。

---

## 主要特性

- 多教师黑盒蒸馏：支持多个闭源商业 LLM 作为教师，统一抽取回答作为训练目标。
- 多教师聚合：投票/加权/去噪策略融合多教师回答，提升训练目标的稳定性与可靠性。
- 灵活蒸馏策略：支持 logits 蒸馏、特征层蒸馏、指令蒸馏、On-/Off-policy 流程、LoRA/SFT 微调。
- 大规模流式处理：针对海量教师语料设计的增量/流式数据管线，节约 IO 与内存开销。
- 一键导出：LoRA 合并后导出为可独立加载的 TinySage 模型，方便下游部署。

---

## 适合谁用

- 希望把商业大模型能力迁移到本地小模型的工程/研究团队。
- 在成本/资源受限环境（推理延迟、部署空间）需要可生产化小模型的场景。
- 需要为特定领域（如音视频、流媒体、GPU/CUDA）定制专用能力的团队。

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
