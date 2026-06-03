# TinySage: 多教师蒸馏小语言模型

TinySage 是基于 Qwen3 家族的多规格小模型，使用自研蒸馏训练框架 **PolyDistill** 
统一调度 GPT、Claude、Gemini 等商业 API 教师模型与学生模型的训练。
支持 0.6B / 1.7B 双规格，修改 config.yaml 中 model_id 即可切换。

PolyDistill 核心能力：
- 支持跨架构商业大模型的黑盒蒸馏
- 多教师回答生成与融合机制
- 大规模教师语料的高效流式处理
- 灵活的训练策略（逻辑层蒸馏、指令蒸馏等）

最终产出的 TinySage 集三家之长，体量 0.6B/1.7B 可选，可灵活部署至端侧或服务器。

**训练框架**：PolyDistill（本仓库提供）  
**模型**：TinySage（0.6B / 1.7B 双规格）

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
│   ├── dataset.py         # 数据加载与预处理
│   ├── trainer.py         # LoRA SFT 训练核心逻辑
│   ├── eval.py            # PPL / ROUGE-L / 生成样本评测
│   ├── json_to_parquet.py # JSON → Parquet 格式转换
│   ├── teachers/          # GPT/Claude/Gemini 适配器
│   └── aggregation/       # 多教师知识聚合
├── scripts/
│   └── train.py           # 入口 main：训练 → 推理对比 → 全量评测
├── config.yaml            # YAML 配置文件（覆盖默认值，推荐编辑此文件）
├── requirements.txt       # Python 依赖清单（Python ≥ 3.10）
├── data/                  # 蒸馏数据集（Parquet 格式）
├── models/                # 模型缓存目录
└── img/                   # 流程图
```

5 层工业结构，依赖关系：`config ← dataset ← trainer ← eval ← scripts/train`

## 环境依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| Python | ≥ 3.10 | - |

### 安装

```bash
pip install -r requirements.txt
```

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
python scripts/train.py --eval-only         # 仅评测已有模型（不重新训练）
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

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(
    base_model,
    "./lora_sft_ai_infra_audio_video_output",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

### 导出独立模型

LoRA adapter 需配合基座模型使用。如需独立分发，将 LoRA 权重合并为完整模型：

```bash
python scripts/export.py                        # 默认输出到 ./models/TinySage-0.6B/
python scripts/export.py --output ./my-model    # 自定义输出路径
```

合并后的 TinySage-0.6B 可直接加载，无需 PEFT：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./models/TinySage-0.6B",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./models/TinySage-0.6B")
```

## 输出

训练完成后在输出目录生成：

- `adapter_config.json` — LoRA 配置记录
- `adapter_model.safetensors` — LoRA 权重（仅几 MB）
- `checkpoint-{step}/` — 各 epoch 检查点
- `eval_report.md` — 评测报告（自动生成）
- `eval_results.json` — 结构化评测结果

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

### 训练监控要点

- **`loss` 持续下降 + `eval_loss` 持平或上升** → 过拟合信号，考虑早停或增强正则化
- **`grad_norm` 突然飙升** → 梯度爆炸，`max_grad_norm` 已在裁剪但需关注
- **`mean_token_accuracy` 快速接近 1.0** → 模型记忆训练集，验证集性能可能已经开始退化
- **`learning_rate`** → 随 cosine/linear scheduler 逐步衰减到 0

## 注意事项

- 默认使用 Hugging Face 镜像 `hf-mirror.com` 加速下载。海外环境可在 `config.yaml` 中设置 `hf_endpoint: "https://huggingface.co"`。
- 单卡训练，已主动禁用所有分布式/DDP 环境变量。
- 训练末尾的推理对比仅用于定性评估，非严谨 benchmark。
