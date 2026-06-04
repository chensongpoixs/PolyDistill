# TinySage: 多教师蒸馏小语言模型

TinySage 是基于 Qwen3 家族的多规格小模型，使用自研蒸馏训练框架 **PolyDistill** 
统一调度 GPT、Claude、Gemini 等商业 API 教师模型与学生模型的训练。
支持 0.6B / 1.7B / 4B / 8B 四规格，修改 config.yaml 中 model_id 即可切换。

### 学生模型规格对比

| 规格 | 参数量 | 层数 | hidden_size | BF16 显存 | 推荐 GPU | 适用场景 |
|------|--------|------|-------------|----------|---------|---------|
| **TinySage-0.6B** | 0.6B | 28 | 1024 | ~1.2 GB | 8GB+ | 端侧部署/快速原型/实时推理 |
| **TinySage-1.7B** | 1.7B | 28 | 2048 | ~3.4 GB | 16GB+ | 服务器部署/较高回答质量 |
| **TinySage-4B** | 4B | 36 | 2560 | ~7.6 GB | 16GB+ | 平衡质量与速度，性价比之选 |
| **TinySage-8B** | 8B | 36 | 4096 | ~16 GB | 24GB+ | 接近生产级效果，强推理能力 |

> 训练显存需求显著高于 BF16 模型大小（需激活值+梯度+优化器状态）。
> 训练峰值估算：0.6B~4GB / 1.7B~8GB / 4B~14GB / 8B~22GB。

PolyDistill 核心能力：
- 支持跨架构商业大模型的黑盒蒸馏
- 多教师回答生成与融合机制
- 大规模教师语料的高效流式处理
- 灵活的训练策略（逻辑层蒸馏、指令蒸馏等）

最终产出的 TinySage 集三家之长，体量 0.6B/1.7B/4B/8B 可选，可灵活部署至端侧或服务器。

**训练框架**：PolyDistill（本仓库提供）  
**模型**：TinySage（0.6B / 1.7B / 4B / 8B 四规格）

---

## Qwen2.5 → Qwen3 升级说明

PolyDistill 从 Qwen2.5-0.5B 切换至 Qwen3 家族，以下是核心升级点：

### 架构改进

| 维度 | Qwen2.5 | Qwen3 |
|------|---------|-------|
| 注意力机制 | 标准 QKV 偏置 | **QK-Norm** 归一化，移除 QKV 偏置，训练更稳定 |
| 预训练数据 | 18T tokens | **36T tokens**，分三阶段（通用→推理→长文本 32K） |
| 多语言支持 | 29 种 | **119 种语言和方言** |
| 上下文长度 | 32K | 32K（训练阶段即支持，0.6B 实测可用） |

### 核心创新：思考/非思考双模式

Qwen3 最重要的突破——**单一模型同时支持两种推理模式**：

- **思考模式 (Thinking)**：模型内部生成思维链，逐步推理复杂问题（数学证明、多步逻辑、代码调试），效果对标专有大推理模型
- **非思考模式 (Non-Thinking)**：快速上下文驱动响应，适用于常规问答、翻译、摘要等场景
- **自动切换**：根据用户查询复杂度自动选择模式，无需手动切换模型
- **思考预算 (Thinking Budget)**：可控制推理计算量，简单问题少算、复杂问题多算，在延迟与性能间灵活平衡

### 训练策略升级

| 阶段 | Qwen2.5 | Qwen3 |
|------|---------|-------|
| 后训练 | SFT + 多阶段 RL | **四阶段**：长 CoT 冷启动 → 推理 RL → 思维模式融合 → 通用 RL |
| 蒸馏策略 | 无内置 | **On-policy + Off-policy 双轨蒸馏**，大模型带动小模型 |
| RL 样本效率 | — | 推理 RL 阶段仅需 **<4000 个问题** 即显著提升 |

### 性能对比

- 代码生成、数学推理、Agent 任务等多项基准达到 **SOTA**
- **0.6B 小模型即可匹敌更大 MoE 模型和部分闭源模型**，性价比极高
- 在各 benchmark 上对 Qwen2.5 全面显著提升

### 对本项目的影响

- 学生模型基础能力更强，同等蒸馏数据下微调效果上限更高
- 思考模式与 PolyDistill 的 `thinking` 数据字段天然契合，可充分挖掘推理链蒸馏潜力
- 多语言扩展使模型可覆盖更广泛的用户场景

---

## 项目结构

```
PolyDistill/
├── poly_distill/          # 训练框架核心代码
│   ├── config.py          # Config 默认值 + load_config() + 环境初始化
│   ├── dataset.py         # 数据加载 + quality_filter 质量过滤 + chat_template 格式化
│   ├── trainer.py         # LoRA SFT 训练 + GPU SM/功耗监控 + YOLOv5 实验目录
│   ├── eval.py            # 6 维度评估: PPL/ROUGE-L/BERTScore/生成样本/通用能力/LLM-Judge
│   ├── json_to_parquet.py # JSON → Parquet 格式转换
│   ├── teachers/          # GPT/Claude/Gemini 适配器
│   └── aggregation/       # 多教师知识聚合
├── scripts/
│   ├── train.py           # 入口 main：训练 → 推理对比 → 全量评测
│   ├── export.py          # LoRA 合并导出 → TinySage 独立模型 + 模型卡片
│   └── benchmark.py       # C-Eval/MMLU 等标准化 benchmark Base vs LoRA 对比
├── config.yaml            # YAML 配置文件（覆盖默认值，推荐编辑此文件）
├── requirements.txt       # Python 依赖清单（Python ≥ 3.10）
├── data/                  # 蒸馏数据集（Parquet 格式）
├── models/                # 模型缓存目录 + TinySage 导出目录
├── runs/train/exp{N}/     # YOLOv5 风格实验目录（LoRA adapter + 日志 + 报告）
├── img/                   # 流程图 / SVG 原理图
└── docs/                  # 详细文档
```

6 层工业结构，依赖关系：`config ← dataset ← trainer ← eval ← scripts/train`

| 层 | 文件 | 核心职责 |
|----|------|---------|
| 配置层 | `config.py` + `config.yaml` | Config class 默认值 + YAML 覆盖 + load_config() |
| 数据层 | `dataset.py` | Parquet 加载 → quality_filter 4步过滤 → chat_template → "text" 字段 |
| 训练层 | `trainer.py` | BF16/SDPA/TF32 → LoRA → SFTTrainer → YOLOv5 实验目录 + GPU 监控 |
| 评测层 | `eval.py` | PPL/ROUGE-L/BERTScore/通用能力/LLM-Judge 5维三方对比 → PASS/WARNING/FAIL |
| 导出层 | `scripts/export.py` | LoRA merge_and_unload → TinySage 独立模型 + 模型卡片 |
| Benchmark | `scripts/benchmark.py` | C-Eval/MMLU/GSM8K 标准化评测 Base vs LoRA 对比 |
| 入口层 | `scripts/train.py` | `__main__` 串联全流程；支持 `--eval-only` / `--skip-eval` 模式 |

## 环境依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| Python | ≥ 3.10 | - |
| torch | ≥ 2.3.0 | 深度学习框架 |
| transformers | ≥ 4.44.0 | 模型加载、Tokenizer、TrainingArguments |
| datasets | ≥ 2.20.0 | 数据集加载与处理 |
| trl | ≥ 0.9.0 | SFTTrainer、DataCollatorForCompletionOnlyLM |
| peft | ≥ 0.12.0 | LoRA 参数高效微调 |

### 安装

```bash
pip install -r requirements.txt
```

### 可选依赖

| 依赖 | 用途 | 安装 |
|------|------|------|
| `flash-attn` | Flash Attention 显存优化（30-50%）+ 训练加速（20-50%） | 见下方 Flash Attention 章节 |
| `bert-score` | BERTScore 语义相似度评估（eval.bertscore.enabled） | `pip install bert-score` |
| `lm-eval` | C-Eval/MMLU Benchmark 标准化评测 | `pip install lm-eval` |
| `nvidia-ml-py` | GPU SM 利用率/功耗/温度实时监控 | `pip install nvidia-ml-py` |
| `openai` | LLM-as-Judge OpenAI SDK 调用外部裁判模型 | `pip install openai` |

### Flash Attention（显存优化，可选）

Flash Attention 可将注意力层显存降低 30-50%，训练速度提升 20-50%。`attn_implementation: "auto"` 时自动检测并使用最优实现：

| GPU 架构 | SM 版本 | 自动选择 | 代表显卡 |
|----------|---------|----------|---------|
| Blackwell | ≥ 120 | Flash Attention 3 | RTX 5080/5090 |
| Hopper | ≥ 90 | Flash Attention 3 | H100 |
| Ada | ≥ 89 | Flash Attention 2 | RTX 4090 |
| Ampere | ≥ 80 | Flash Attention 2 | RTX 3090, A100 |
| Volta/Turing | < 80 | SDPA | V100, RTX 2080Ti |
| 未安装 flash-attn | — | SDPA（PyTorch 内置） | 任意 |

**安装（推荐）**：

```bash
# Linux / WSL2
pip install flash-attn --no-build-isolation

# Windows：源码编译需要 MSVC + CUDA 开发环境，推荐下载预编译 wheel
# 从 https://github.com/Dao-AILab/flash-attention/releases 下载对应版本
# 例如 CUDA 12.8 + PyTorch 2.8 + Python 3.10：
pip install flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

> **注意**：
> - Flash Attention 3 需要 `flash-attn >= 2.6` + `transformers >= 4.51`
> - 未安装 flash-attn 时自动回退 SDPA，训练正常进行，仅显存占用较高
> - RTX 5080/5090（Blackwell）必须使用预编译 wheel 或 flash-attn >= 2.7（源码编译需 CUDA 12.8+）

> GPU 驱动要求：NVIDIA Driver ≥ 525（CUDA 12 兼容）。推荐 Ampere 及以上架构（3090/4090/5090/A100 等）以支持 BF16。

## 数据集

蒸馏数据集覆盖以下 AI Infra 领域主题：

- **音视频编解码**：FFmpeg 全链路处理、H.264 码流操作
- **流媒体协议**：RTMP/HTTP-FLV/HLS 选型、GB28181 PS 封包与 RTP 打包
- **GPU 编程**：CUDA Stream 并行、零拷贝技术、统一寻址（UVA）
- **网络传输**：WebRTC ICE/STUN/TURN、JitterBuffer 自适应、FEC/NACK QoS
- **屏幕采集**：DXGI Desktop Duplication、多显示器适配
- **大模型推理**：vLLM Continuous Batching、PagedAttention、INT8 量化
- **AI 应用**：RAG 检索优化、Whisper 流式 ASR、CosyVoice2 零样本 TTS
- **系统工程**：P2P 中转服务器设计、共享内存环形缓冲区、设备注册与心跳

### SFT数据集格式

```json
[
  {
    "instruction": "问题...",
    "input": "",
    "thinking": "思考过程...",
    "output": "最终回答..."
  }
]
```

训练使用 Parquet 格式（reasoning-distill schema）。首次使用需转换：

```bash
python poly_distill/json_to_parquet.py --input ./data --output ./data
```

## 使用方法

### 训练

```bash
python scripts/train.py                     # 完整流水线：训练 → 推理对比 → 全量评测
python scripts/train.py --config prod.yaml  # 指定配置文件
python scripts/train.py --skip-eval         # 跳过全量评测（仅做快速推理对比）
python scripts/train.py --eval-only         # 仅评测已有模型（不重新训练，自动探测最新 adapter）
```

### Benchmark 标准化评测

```bash
python scripts/benchmark.py                          # 完整 C-Eval（1,346 题）
python scripts/benchmark.py --limit 20              # 快速验证 20 题
python scripts/benchmark.py --tasks ceval,mmlu       # 多 benchmark 评测
```

### 单独运行评测

```bash
python poly_distill/eval.py                          # 直接运行全量评测
```

### 调整配置

推荐直接编辑 `config.yaml`，无需改动 Python 代码：

```yaml
model_id: "Qwen/Qwen3-0.6B"
training:
  num_train_epochs: 100
  learning_rate: 2.0e-4
  per_device_batch_size: 4
```

### 加载微调模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 按模型规格选择 model_id：
#   TinySage-0.6B: "Qwen/Qwen3-0.6B"
#   TinySage-1.7B: "Qwen/Qwen3-1.7B"
#   TinySage-4B:   "Qwen/Qwen3-4B"
#   TinySage-8B:   "Qwen/Qwen3-8B"
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    torch_dtype="auto",
)
# LoRA adapter 位于 runs/train/exp{N}/ 实验目录下
model = PeftModel.from_pretrained(
    base_model,
    "./runs/train/exp3",  # 替换为实际实验目录
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

### 导出独立模型

LoRA adapter 需配合基座模型使用。如需独立分发，将 LoRA 权重合并为完整模型：

```bash
python scripts/export.py                     # 自动探测最新实验 → ./models/TinySage-{规格}
python scripts/export.py --list              # 列出所有可导出实验
python scripts/export.py --exp exp3          # 导出指定实验
python scripts/export.py --adapter ./runs/train/exp3  # 手动指定 adapter 路径
python scripts/export.py --output ./my-model # 自定义输出路径
```

导出逻辑：
- **自动探测**：搜索 `runs/train/exp{N}/` 下最新包含 `adapter_config.json` 的目录
- **智能命名**：`Qwen/Qwen3-4B` → `TinySage-4B`（正则提取模型规格后缀）
- **输出优先级**：CLI `--output` > `config.yaml` `export.output_dir` > 自动推导

合并后的 TinySage 可直接加载，无需 PEFT：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 按规格切换路径：
model = AutoModelForCausalLM.from_pretrained(
    "./models/TinySage-0.6B",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./models/TinySage-0.6B")
```

### 评估维度配置

6 个评估维度均支持 `config.yaml` 独立开关，按需启用：

```yaml
# config.yaml
eval:
  ppl:
    enabled: true            # PPL 困惑度（计算快，无外部依赖）
  rouge:
    enabled: false           # ROUGE-L 字面匹配（默认关闭，用 BERTScore 替代）
  bertscore:
    enabled: true            # BERTScore 语义相似度（需 pip install bert-score）
  gen_samples:
    enabled: true            # 生成样本 Base vs LoRA 并排对比
  general_ability:
    enabled: true            # 通用能力评估（20 道跨领域问题，检测灾难性遗忘）
  llm_judge:
    enabled: true            # LLM-as-Judge 大模型打分
```

### LLM-as-Judge 质量评估

启用后，外部大模型进行 **5 维度三方对比打分**（准确性 / 相关性 / 完整性 / 清晰度 / 整体质量，1-5 分制）：

- **三方对比**：Base（原始基座）vs LoRA（微调后）vs Teacher（教师参考答案）
- **输出字段**：`improvement_over_baseline`（蒸馏增益）+ `gap_to_teacher`（与教师差距）

```yaml
# config.yaml
eval:
  llm_judge:
    enabled: true                                           # 启用评估
    endpoint: "http://your-llm-server/v1/chat/completions"  # 兼容 OpenAI API 的地址
    model: "gpt-4"                                          # 裁判模型名
    api_key: "sk-xxx"                                       # API Key（留空则读环境变量 LLM_JUDGE_API_KEY）
    max_samples: 50                                          # 最大评估样本数（避免 API 费用过高）
```

评估结果自动写入 `eval_report.md`（含"蒸馏增益分析"和"与教师差距"章节）和 `eval_results.json` 的 `llm_judge` 字段。

> **推荐裁判模型**：Claude 4.5/Opus 4.6、GPT-4o、DeepSeek-V3 等强模型，或本地部署的 Qwen3/Gemma 等兼容 OpenAI API 的服务。

## 输出

训练采用 YOLOv5 风格实验目录 `runs/train/exp{N}/`（自增编号），所有产物归入同一目录：

- `adapter_config.json` — LoRA 配置记录
- `adapter_model.safetensors` — LoRA 权重（仅几 MB）
- `checkpoint-{step}/` — 各 epoch 检查点
- `config_snapshot.yaml` — 训练时配置快照
- `results.csv` — 训练日志（loss/grad_norm/accuracy/lr 每步记录）
- `train.log` — 完整训练日志（含 GPU SM 利用率/功耗/温度）
- `eval_report.md` — 评测报告（自动生成）
- `eval_results.json` — 结构化评测结果（供程序化分析）
- `benchmark_report.md` — Benchmark 标准化评测报告
- `benchmark_results.json` — Benchmark 结构化结果

导出独立模型（`python scripts/export.py`）输出至 `./models/TinySage-{规格}/`。

### 训练日志参数

训练过程中每 `logging_steps` 步输出一行日志，各字段含义：

| 参数 | 含义 | 说明 |
|------|------|------|
| `loss` | 交叉熵损失 | 当前 step 的 language modeling loss，越低拟合越好 |
| `grad_norm` | 梯度范数 | 梯度裁剪前的总梯度 L2 范数，反映参数更新幅度；过大可能震荡，过小可能停滞 |
| `learning_rate` | 学习率 | 当前 step 的实际 LR，随 scheduler 变化（如 cosine 衰减） |
| `num_tokens` | 已处理 token 数 | 训练到目前为止处理的总 token 数 |
| `mean_token_accuracy` | 平均 token 准确率 | 每个 token 预测正确的比例，接近 1.0 表示模型已高度拟合当前数据 |
| `epoch` | 当前 epoch | 小数格式（如 58.4 表示第 58 个 epoch 的 40% 进度） |
| `SM_util` | GPU SM 利用率 | Streaming Multiprocessor 活跃占比（0-100%），反映 GPU 计算资源使用效率 |
| `power` | GPU 功耗 | 当前 GPU 功耗（W），用于评估能效和散热 |
| `temp` | GPU 温度 | 当前 GPU 核心温度（°C） |

### 训练监控要点

- **`loss` 持续下降 + `eval_loss` 持平或上升** → 过拟合信号，考虑早停或增强正则化
- **`grad_norm` 突然飙升** → 梯度爆炸，`max_grad_norm` 已在裁剪但需关注
- **`mean_token_accuracy` 快速接近 1.0** → 模型记忆训练集，验证集性能可能已经开始退化
- **`learning_rate`** → 随 cosine/linear scheduler 逐步衰减到 0
- **`SM_util` 持续 < 60%** → GPU 数据饥饿，考虑增加 num_workers/batch_size 或减少 CPU 瓶颈
- **`power` / `temp` 异常** → 检查散热和功耗限制，可能需要锁定 GPU 频率

## 注意事项

- 默认使用 Hugging Face 镜像 `hf-mirror.com` 加速下载。海外环境可在 `config.yaml` 中设置 `hf_endpoint: "https://huggingface.co"`。
- 单卡训练，已主动禁用所有分布式/DDP 环境变量。
- 训练末尾的推理对比仅用于定性评估，非严谨 benchmark。
