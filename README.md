# AI Infra 领域知识蒸馏

使用 LoRA SFT 对 `Qwen2.5-0.5B-Instruct` 进行参数高效微调，将音视频/流媒体/GPU/CUDA 等 AI 基础设施领域的专业知识注入小模型。

## 项目背景

大模型在通用对话上表现优异，但在特定垂直领域（如 AI 基础设施）往往缺乏深度。本项目通过知识蒸馏的方式，用高质量领域 QA 数据微调小参数模型，使其在 AI Infra 技术面试场景下给出专业、结构化的回答。

## 项目结构

```
ai_infra/
├── config.py                      # Config 默认值 + load_config() + 环境初始化
├── config.yaml                    # YAML 配置文件（覆盖默认值，推荐编辑此文件）
├── dataset.py                     # 数据加载与预处理
├── train.py                       # LoRA SFT 训练核心逻辑
├── inference.py                   # 推理对比评估 + 入口 main（支持 --config 参数）
├── requirements.txt               # Python 依赖清单（Python ≥ 3.10）
├── data/
│   └── ai_infra_audio_video.json  # 训练数据集（instruction-output 格式）
├── img/                           # 流程图（LoRA 原理、训练流程等）
├── models/qwen2.5-0.5b/           # 基座模型缓存目录（运行后自动下载）
└── lora_sft_ai_infra_audio_video_output/  # LoRA adapter 输出目录
```

4 层工业结构，依赖关系：`config ← dataset ← train ← inference`

## 环境依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| Python | ≥ 3.10 | - |

### 安装

```bash
# 一键安装所有依赖
pip install -r requirements.txt
```

> GPU 驱动要求：NVIDIA Driver ≥ 525（CUDA 12 兼容）。推荐 Ampere 及以上架构（3090/4090/A100 等）以支持 BF16。
> 若需指定 CUDA 版本的 PyTorch，先单独安装 torch 再执行 `pip install -r requirements.txt`。

## 数据集

`data/ai_infra_audio_video.json` 包含约 18 条高质量中文技术面试问答，覆盖以下主题：

- **音视频编解码**：FFmpeg 全链路处理、H.264 码流操作
- **流媒体协议**：RTMP/HTTP-FLV/HLS 选型、GB28181 PS 封包与 RTP 打包
- **GPU 编程**：CUDA Stream 并行、零拷贝技术、统一寻址（UVA）
- **网络传输**：WebRTC ICE/STUN/TURN、JitterBuffer 自适应、FEC/NACK QoS
- **屏幕采集**：DXGI Desktop Duplication、多显示器适配
- **大模型推理**：vLLM Continuous Batching、PagedAttention、INT8 量化
- **AI 应用**：RAG 检索优化、Whisper 流式 ASR、CosyVoice2 零样本 TTS
- **系统工程**：P2P 中转服务器设计、共享内存环形缓冲区、设备注册与心跳

### 数据格式

```json
[
  {
    "instruction": "面试问题...",
    "input": "",
    "output": "【面试官目的】...【回答思路】...【思考过程】...【最终答案】..."
  }
]
```

## 使用方法

### 训练

```bash
# 确保 ai_infra_audio_video.json 在项目根目录下
python inference.py                     # 自动读取 ./config.yaml
python inference.py --config prod.yaml  # 指定配置文件
```

脚本会自动：
1. 从 Hugging Face 下载 `Qwen2.5-0.5B-Instruct`（首次运行）
2. 应用 LoRA 微调（仅训练 q_proj/v_proj，约 2M 可训练参数）
3. 保存 LoRA adapter 到 `./lora_sft_ai_infra_audio_video_output/`
4. 对比基座模型与微调模型的回答质量

### 调整配置

推荐直接编辑 `config.yaml`，无需改动 Python 代码：

```yaml
# config.yaml
model_id: "Qwen/Qwen2.5-0.5B-Instruct"   # 可替换为其他 Qwen/Llama 模型
training:
  num_train_epochs: 300                   # 调整训练轮次
  learning_rate: 2.0e-4                   # 调整学习率
  per_device_batch_size: 4                # 调整 batch size
```

`config.py` 中的 `Config` class 提供所有字段的默认值，YAML 中未声明的字段自动使用默认值。若未安装 PyYAML 或 YAML 文件不存在，静默回退到 Config 默认值。

## 输出

训练完成后在 `OUTPUT_DIR` 生成：

- `adapter_config.json` — LoRA 配置记录
- `adapter_model.safetensors` — LoRA 权重（仅几 MB）
- `checkpoint-{step}/` — 各 epoch 检查点

### 加载微调模型进行推理

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(
    base_model,
    "./lora_sft_ai_infra_audio_video_output",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# ... 正常使用 model.generate()
```

## 注意事项

- 脚本默认使用 Hugging Face 镜像 `hf-mirror.com`，便于国内网络环境。海外环境可在 `Config.HF_ENDPOINT` 中改为 `https://huggingface.co`。
- 单卡训练，已主动禁用所有分布式/DDP 环境变量，避免误触发多卡通信。
- 脚本末尾的推理对比仅用于定性评估，非严谨 benchmark。
