# TinySage: A Multi-Teacher Distilled Small Language Model

TinySage is a multi-scale SLM based on the Qwen3 family (0.6B / 1.7B / 4B / 8B). Switch by setting `model_id` in `config.yaml`.  
It was trained using **PolyDistill**, a flexible multi-teacher knowledge distillation framework 
that orchestrates API-based teachers (GPT, Claude, Gemini) 
and a local student model under a unified training pipeline.

### Student Model Comparison

| Variant | Params | Layers | hidden_size | BF16 Size | Recommended GPU | Use Case |
|---------|--------|--------|-------------|-----------|-----------------|----------|
| **TinySage-0.6B** | 0.6B | 28 | 1024 | ~1.2 GB | 8GB+ | Edge deployment, fast prototyping |
| **TinySage-1.7B** | 1.7B | 28 | 2048 | ~3.4 GB | 16GB+ | Server deployment, higher quality |
| **TinySage-4B** | 4B | 36 | 2560 | ~7.6 GB | 16GB+ | Balanced quality/speed sweet spot |
| **TinySage-8B** | 8B | 36 | 4096 | ~16 GB | 24GB+ | Near-production quality, strong reasoning |

> Training VRAM is significantly higher than BF16 model size (activations + gradients + optimizer state).
> Estimated training peak: 0.6B~4GB / 1.7B~8GB / 4B~14GB / 8B~22GB.

PolyDistill handles:
- **Black-box distillation** from heterogeneous commercial LLMs
- **Multi-teacher response generation & aggregation**
- **Efficient data streaming** from large teacher corpora
- **Customizable training strategies** (logit-level, feature-level, or instruction-level distillation)

The result: a tiny yet capable model that inherits the collective strengths of 
three state-of-the-art teachers, deployable on edge devices with minimal footprint.

**Framework**: PolyDistill (included in this repo)  
**Model**: TinySage (0.6B / 1.7B / 4B / 8B quad-scale)

---

## Qwen2.5 → Qwen3 Upgrade Notes

PolyDistill has upgraded the student model from Qwen2.5-0.5B to the Qwen3 family. Key improvements:

### Architecture

| Dimension | Qwen2.5 | Qwen3 |
|-----------|---------|-------|
| Attention | Standard QKV bias | **QK-Norm**, QKV bias removed — more stable training |
| Pretraining | 18T tokens | **36T tokens** across 3 stages (general → reasoning → long-context 32K) |
| Multilingual | 29 languages | **119 languages and dialects** |
| Context Length | 32K | 32K (trained with 32K sequences natively) |

### Core Innovation: Dual-Mode Thinking

Qwen3's breakthrough — **a single model supporting both inference modes**:

- **Thinking Mode**: Internal chain-of-thought for complex multi-step reasoning (math proofs, debugging, logic)
- **Non-Thinking Mode**: Fast context-driven responses for standard Q&A, translation, summarization
- **Auto-switching**: Dynamically selects mode based on query complexity — no manual model switching
- **Thinking Budget**: Adaptive compute allocation — spend more tokens on hard problems, less on easy ones

### Training Strategy

| Stage | Qwen2.5 | Qwen3 |
|-------|---------|-------|
| Post-training | SFT + Multi-stage RL | **4-stage**: Long CoT cold-start → Reasoning RL → Thought-mode fusion → General RL |
| Distillation | None built-in | **On-policy + Off-policy dual distillation**, large model teaches small model |
| RL Sample Efficiency | — | **<4,000 questions** for significant reasoning gains |

### Performance

- SOTA on code generation, math reasoning, and agent benchmarks
- **0.6B model rivals larger MoE and some closed-source models**
- Significant improvements over Qwen2.5 across all benchmarks

### Impact on This Project

- Stronger base capability raises the ceiling for distillation fine-tuning
- Thinking mode aligns perfectly with PolyDistill's `thinking` data field for reasoning-chain distillation
- Expanded multilingual coverage enables broader application scenarios

---

## Project Structure

```
PolyDistill/
├── poly_distill/          # Core training framework
│   ├── config.py          # Config defaults + load_config() + env setup
│   ├── dataset.py         # Data loading & preprocessing
│   ├── trainer.py         # LoRA SFT training core
│   ├── eval.py            # PPL / ROUGE-L / generation sample evaluation
│   ├── json_to_parquet.py # JSON → Parquet format conversion
│   ├── teachers/          # GPT/Claude/Gemini adapters
│   └── aggregation/       # Multi-teacher knowledge aggregation
├── scripts/
│   └── train.py           # Main entry: train → compare → eval
├── config.yaml            # YAML config file (overrides defaults)
├── requirements.txt       # Python dependencies (Python ≥ 3.10)
├── data/                  # Distillation dataset (Parquet format)
├── models/                # Model cache directory
└── img/                   # Diagrams
```

5-layer architecture, one-way dependencies: `config ← dataset ← trainer ← eval ← scripts/train`

## Environment

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | ≥ 3.10 | - |

### Install

```bash
pip install -r requirements.txt
```

### Flash Attention (Memory Optimization, Optional)

Flash Attention reduces attention-layer memory by 30-50% and speeds up training by 20-50%. When `attn_implementation: "auto"`, the optimal implementation is auto-detected:

| GPU Arch | SM Version | Auto-Selected | Representative GPU |
|----------|-----------|---------------|-------------------|
| Blackwell | ≥ 120 | Flash Attention 3 | RTX 5080/5090 |
| Hopper | ≥ 90 | Flash Attention 3 | H100 |
| Ada | ≥ 89 | Flash Attention 2 | RTX 4090 |
| Ampere | ≥ 80 | Flash Attention 2 | RTX 3090, A100 |
| Volta/Turing | < 80 | SDPA | V100, RTX 2080Ti |
| flash-attn not installed | — | SDPA (PyTorch built-in) | Any |

**Install (Recommended)**:

```bash
# Linux / WSL2
pip install flash-attn --no-build-isolation

# Windows: source build requires MSVC + CUDA toolkit; pre-built wheels recommended
# Download from https://github.com/Dao-AILab/flash-attention/releases
# Example for CUDA 12.8 + PyTorch 2.8 + Python 3.10:
pip install flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

> **Note**:
> - Flash Attention 3 requires `flash-attn >= 2.6` + `transformers >= 4.51`
> - Falls back to SDPA automatically if flash-attn is not installed — training proceeds normally
> - RTX 5080/5090 (Blackwell) requires pre-built wheel or flash-attn >= 2.7 (source build needs CUDA 12.8+)

> GPU requirement: NVIDIA Driver ≥ 525 (CUDA 12 compatible). Ampere+ architecture recommended (3090/4090/5090/A100) for BF16 support.

## Dataset

The knowledge distillation dataset covers AI Infra domain topics:

- **Audio/Video Codecs**: FFmpeg pipeline, H.264 bitstream operations
- **Streaming Protocols**: RTMP/HTTP-FLV/HLS, GB28181 PS/RTP packaging
- **GPU Programming**: CUDA Stream parallelism, zero-copy, UVA
- **Network Transport**: WebRTC ICE/STUN/TURN, JitterBuffer, FEC/NACK QoS
- **Screen Capture**: DXGI Desktop Duplication, multi-monitor support
- **LLM Inference**: vLLM Continuous Batching, PagedAttention, INT8 quantization
- **AI Applications**: RAG optimization, Whisper streaming ASR, CosyVoice2 zero-shot TTS
- **Systems Engineering**: P2P relay design, shared memory ring buffer, device heartbeat

### Data Format

```json
[
  {
    "instruction": "Interview question...",
    "input": "",
    "thinking": "Reasoning process...",
    "output": "Final answer..."
  }
]
```

Training uses Parquet format (reasoning-distill schema). Convert JSON first:

```bash
python poly_distill/json_to_parquet.py --input ./data --output ./data
```

## Usage

### Training

```bash
python scripts/train.py                     # Full pipeline: train → compare → eval
python scripts/train.py --config prod.yaml  # Specify config file
python scripts/train.py --skip-eval         # Skip full eval (quick compare only)
python scripts/train.py --eval-only         # Evaluate existing model only
```

### Adjust Config

Edit `config.yaml` — no Python code changes needed:

```yaml
model_id: "Qwen/Qwen3-0.6B"
training:
  num_train_epochs: 100
  learning_rate: 2.0e-4
  per_device_batch_size: 4
```

### Load Fine-tuned Model

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

### Export Standalone Model

The LoRA adapter requires the base model. To distribute independently, merge LoRA weights into a full model:

```bash
python scripts/export.py                        # Default output: ./models/TinySage-0.6B/
python scripts/export.py --output ./my-model    # Custom output path
```

The merged TinySage-0.6B can be loaded directly without PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./models/TinySage-0.6B",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./models/TinySage-0.6B")
```

### LLM-as-Judge Evaluation

After training, enable an external LLM to automatically score generation quality across 4 dimensions: **accuracy / relevance / completeness / overall** (1-5 scale), with detailed commentary.

```yaml
# config.yaml
eval:
  llm_judge:
    enabled: true                                         # Enable evaluation
    endpoint: "http://your-llm-server/v1/chat/completions"  # OpenAI-compatible API
    model: "gpt-4"                                        # Judge model name
    api_key: "sk-xxx"                                     # API Key (falls back to env var LLM_JUDGE_API_KEY)
    max_samples: 10                                       # Max samples to evaluate
```

Results are written to `eval_report.md` chapter 4 and `eval_results.json` under `llm_judge`.

> **Recommended judges**: Claude 4.5/Opus 4.6, GPT-4o, DeepSeek-V3, or locally deployed models (Qwen3, Gemma) with OpenAI-compatible APIs.

## Output

After training, the output directory contains:

- `adapter_config.json` — LoRA configuration
- `adapter_model.safetensors` — LoRA weights (a few MB)
- `checkpoint-{step}/` — per-epoch checkpoints
- `eval_report.md` — evaluation report (auto-generated)
- `eval_results.json` — structured results

### Training Log Parameters

During training, a log entry is printed every `logging_steps` steps:

| Parameter | Meaning |
|-----------|---------|
| `loss` | Cross-entropy loss — lower is better fit |
| `grad_norm` | Total gradient L2 norm before clipping — reflects update magnitude |
| `learning_rate` | Current LR after scheduler step (e.g. cosine decay) |
| `num_tokens` | Total tokens processed so far |
| `mean_token_accuracy` | Token-level prediction accuracy — approaching 1.0 signals memorization |
| `epoch` | Current epoch in fractional form (e.g. 58.4 = epoch 58 at 40%)

### Training Monitoring

- **`loss` dropping while `eval_loss` plateaus/rises** → overfitting, consider early stopping
- **`grad_norm` spikes** → gradient explosion, already clipped by `max_grad_norm`
- **`mean_token_accuracy` nearing 1.0** → training set memorization, validation likely degrading
- **`learning_rate`** → decays toward 0 via cosine/linear scheduler

## Notes

- Uses HF mirror `hf-mirror.com` by default for China network. Set `hf_endpoint: "https://huggingface.co"` in `config.yaml` for overseas.
- Single-GPU training; DDP environment variables are explicitly cleared.
- The quick inference comparison at the end of training is qualitative, not a rigorous benchmark.
