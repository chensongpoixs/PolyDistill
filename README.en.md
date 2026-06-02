# TinySage: A Multi-Teacher Distilled Small Language Model

TinySage is a 0.5B small language model (SLM) based on Qwen2.5-0.5B.  
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
**Model**: TinySage-0.5B

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
model_id: "Qwen/Qwen2.5-0.5B-Instruct"
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
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(
    base_model,
    "./lora_sft_ai_infra_audio_video_output",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

### Export Standalone Model

The LoRA adapter requires the base model. To distribute independently, merge LoRA weights into a full model:

```bash
python scripts/export.py                        # Default output: ./models/TinySage-0.5B/
python scripts/export.py --output ./my-model    # Custom output path
```

The merged TinySage-0.5B can be loaded directly without PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./models/TinySage-0.5B",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./models/TinySage-0.5B")
```

## Output

After training, the output directory contains:

- `adapter_config.json` — LoRA configuration
- `adapter_model.safetensors` — LoRA weights (a few MB)
- `checkpoint-{step}/` — per-epoch checkpoints
- `eval_report.md` — evaluation report (auto-generated)
- `eval_results.json` — structured results

## Notes

- Uses HF mirror `hf-mirror.com` by default for China network. Set `hf_endpoint: "https://huggingface.co"` in `config.yaml` for overseas.
- Single-GPU training; DDP environment variables are explicitly cleared.
- The quick inference comparison at the end of training is qualitative, not a rigorous benchmark.
