# TinySage: A Multi-Teacher Distilled Small Language Model

TinySage is a multi-scale SLM based on the Qwen3 family (0.6B / 1.7B). Switch by setting `model_id` in `config.yaml`.  
It was trained using **PolyDistill**, a flexible multi-teacher knowledge distillation framework 
that orchestrates API-based teachers (GPT, Claude, Gemini) 
and a local student model under a unified training pipeline.

PolyDistill handles:
- **Black-box distillation** from heterogeneous commercial LLMs
- **Multi-teacher response generation & aggregation**
- **Efficient data streaming** from large teacher corpora
- **Customizable training strategies** (logit-level, feature-level, or instruction-level distillation)

The result: a tiny yet capable model that inherits the collective strengths of 
three state-of-the-art teachers, deployable on edge devices with minimal footprint.

**Framework**: PolyDistill (included in this repo)  
**Model**: TinySage (0.6B / 1.7B dual-scale)

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
