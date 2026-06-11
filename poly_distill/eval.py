"""
训练质量评估模块。

评估维度（均支持 config.yaml 开关，默认 PPL/样本/Judge 开启，ROUGE 关闭）：
  1. Perplexity (PPL)      — 模型对测试文本的困惑度，越低越好（eval.ppl.enabled）
  2. ROUGE-L               — 最长公共子序列字面重叠度（eval.rouge.enabled，默认关闭）
  3. BERTScore             — BERT 语义向量余弦相似度，弥补 ROUGE-L 对改写不友好的缺陷
  4. 生成样本对比           — Base vs LoRA 实际输出并排展示，供人工抽查（eval.gen_samples.enabled）
  5. 通用能力评估           — 20 道跨领域问题（数学/科学/逻辑/中文/代码），检测灾难性遗忘
  6. LLM-as-Judge          — 外部大模型 5 维度三方对比打分（Base vs LoRA vs Teacher）
                             维度：准确性/相关性/完整性/清晰度/整体质量
                             输出：improvement_over_baseline + gap_to_teacher（eval.llm_judge.enabled）

综合判定：
  基于 PPL / ROUGE-L / BERTScore / 通用能力 多指标联动，自动输出 PASS / WARNING / FAIL。

输出文件：
  - eval_report.md    : 可读的 Markdown 评测报告
  - eval_results.json : 结构化原始数据（供后续程序化分析）
"""

import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

# 将项目根目录加入 Python 搜索路径，确保直接运行 poly_distill/eval.py 时能找到 poly_distill 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from transformers.utils import is_flash_attn_2_available


from poly_distill.config import Config
from poly_distill.llm_client import LLMClient

logger = logging.getLogger(__name__)

# BERTScore — 语义相似度评估（可选依赖，未安装时静默回退）
try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    bert_score_fn = None

# LLM-as-Judge 评分维度
_JUDGE_CRITERIA = {
    "accuracy": "准确性 — 回答内容技术正确、无事实错误",
    "relevance": "相关性 — 回答紧扣问题，不偏离主题",
    "completeness": "完整性 — 回答覆盖问题的关键要点，无重大遗漏",
    "clarity": "清晰度 — 表达清晰、逻辑通顺、结构易于理解",
    "overall": "整体质量 — 综合评判回答的专业性和可用性",
}


# ============================================================
# 0. 数据格式标准化
# ============================================================
def _get_question(example: dict) -> str:
    """从 Parquet 样本中提取用户问题。

    Parquet messages 列表中 role=user 的 content。
    """
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _get_reference(example: dict) -> str:
    """从 Parquet 样本中提取参考答案（response 列）。"""
    return example.get("response", "")


def _build_messages(example: dict, config: Config) -> list:
    """从 Parquet 行构造完整对话消息列表。

    Parquet schema: {messages, thinking, response, system}

    返回统一的 [system, user, assistant] 消息列表。
    """
    system = example.get("system") or config.SYSTEM_PROMPT

    user_content = ""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    thinking = example.get("thinking") or ""
    response = example.get("response", "")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "reasoning_content": thinking, "content": response},
    ]


# ============================================================
# 1. Perplexity 评估
# ============================================================
def evaluate_perplexity(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    samples: list,
    label: str,
) -> dict:
    """计算模型在给定样本集上的困惑度（PPL）。

    PPL = exp(cross_entropy_loss)。
    数值越低，模型对该文本越"不惊讶"，即拟合越好。
    但 PPL 低 ≠ 生成质量高，需结合其他指标综合判断。

    Args:
        model: 待评估模型（base 或 LoRA）。
        tokenizer: 分词器。
        config: 全局配置。
        samples: 样本列表，每条含 instruction / output。
        label: 模型标签（用于输出）。

    Returns:
        {"avg_ppl": float, "avg_loss": float, "num_samples": int}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = len(samples)
    logger.info(f">>> 开始 Perplexity 评估 ，当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    with torch.no_grad():
        for i, example in enumerate(samples):
            # 构造与训练一致的对话格式（兼容 JSON 和 Parquet 两种格式）
            messages = _build_messages(example, config)
            start_time = time.time()  # 用于计算处理速度（tokens/s），反映实际评测效率
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # 前向传播计算 loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

            # 进度（每 10 条或最后一条打印）
            #if (i + 1) % 10 == 0 or i == n - 1:
            logger.info(f"  [{label}] PPL: {i+1}/{n}, {total_tokens / (time.time() - start_time):.2f} tokens/s");
            del inputs, outputs, loss  # 及时删除中间变量，释放显存
        #print("", flush=True)  # 换行
    # 3. 清理 inputs，以防后面变长
    #del inputs
    torch.cuda.synchronize()  # 等待GPU操作完成
    torch.cuda.empty_cache()  # 清空 GPU 缓存，释放显存占用
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    avg_ppl = torch.exp(torch.tensor(avg_loss)).item()

    logger.info("  [%s] Loss=%.4f  PPL=%.2f  (tokens=%d)", label, avg_loss, avg_ppl, total_tokens)
    return {"label": label, "avg_ppl": round(avg_ppl, 2), "avg_loss": round(avg_loss, 4)}


# ============================================================
# 2. ROUGE-L 评估
# ============================================================
def _lcs_length(a: list, b: list) -> int:
    """最长公共子序列（Longest Common Subsequence）长度。

    动态规划，O(m*n)。用于计算 ROUGE-L。
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(reference: str, candidate: str) -> Tuple[float, float, float]:
    """计算 ROUGE-L 分数。

    ROUGE-L 基于最长公共子序列（LCS）计算：
      R_lcs = LCS_len / ref_len   （召回率）
      P_lcs = LCS_len / cand_len  （精确率）
      F_lcs = 2 * R * P / (R + P) （F1）

    对中文文本按字符切分（逐字比较），无需分词。
    """
    ref_chars = list(reference)
    cand_chars = list(candidate)
    lcs_len = _lcs_length(ref_chars, cand_chars)

    if len(ref_chars) == 0 or len(cand_chars) == 0:
        return 0.0, 0.0, 0.0

    recall = lcs_len / len(ref_chars)
    precision = lcs_len / len(cand_chars)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return recall, precision, f1


def evaluate_rouge(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    samples: list,
    label: str,
) -> dict:
    """生成回答并与参考答案计算 ROUGE-L F1。

    Args:
        model: 待评估模型。
        tokenizer: 分词器。
        config: 全局配置。
        samples: 样本列表。
        label: 模型标签。

    Returns:
        {"label": str, "rouge_l_f1": float, "num_samples": int}
    """
    model.eval()
    scores = []
    n = len(samples)

    start_time = time.time()  # 用于计算生成速度（tokens/s），包含前向传播和解码时间，反映实际使用体验
    for i, example in enumerate(samples):
        # 生成
        question = _get_question(example)
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                temperature=config.EVAL_TEMPERATURE,
                do_sample=config.EVAL_TEMPERATURE > 0,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取 assistant 回复部分
        if "<|im_start|>assistant" in generated:
            generated = generated.split("<|im_start|>assistant")[-1].strip()

        reference = _get_reference(example)
        _, _, f1 = rouge_l_score(reference, generated)
        scores.append(f1)

        # 进度
        logger.info(f"  [{label}] ROUGE: {i+1}/{n}, {len(generated) / (time.time() - start_time):.2f} tokens/s");
    # logger.info("", flush=True)  # 换行

    avg_f1 = sum(scores) / len(scores) if scores else 0.0
    logger.info("  [%s] ROUGE-L F1=%.4f  (samples=%d)", label, avg_f1, len(scores))
    return {"label": label, "rouge_l_f1": round(avg_f1, 4), "num_samples": len(scores)}


# ============================================================
# 2.5 BERTScore 评估（语义相似度）
# ============================================================
def evaluate_bertscore(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    samples: list,
    label: str,
) -> dict:
    """使用 BERTScore 评估生成回答与参考答案的语义相似度。

    原理：
      BERTScore 将参考答案和生成文本分别通过预训练 BERT 编码为 token embeddings，
      计算 token 间的余弦相似度，得到 Precision / Recall / F1。

      与 ROUGE-L 的关键区别：
        ROUGE-L 基于最长公共子序列（逐字严格匹配）→ 惩罚改写
        BERTScore 基于语义向量余弦相似度 → 识别"意思相同表述不同"的等价文本

      示例：
        参考答案: "FFmpeg 使用 -c:v libx264 -preset slow 进行 H.264 编码"
        LoRA生成: "可以通过 ffmpeg -c:v libx264 -preset veryslow 启用 x264 编码器"

        ROUGE-L F1  ≈ 0.15  (字面重叠低，误判为差)
        BERTScore F1 ≈ 0.92  (语义高度相似，正确判定)

    依赖: pip install bert-score
    未安装时静默跳过，返回空结果。

    Args:
        model: 待评估模型。
        tokenizer: 分词器。
        config: 全局配置。
        samples: 评估样本列表。
        label: 模型标签。

    Returns:
        {"label": str, "bertscore_f1": float, "precision": float, "recall": float,
         "num_samples": int}
        或 {"label": str, "enabled": False} (bert-score 未安装时)
    """
    if not _BERTSCORE_AVAILABLE:
        logger.warning("  [%s] BERTScore: bert-score 未安装，跳过 (pip install bert-score)", label)
        return {"label": label, "enabled": False}

    model.eval()
    references = []
    candidates = []

    for i, example in enumerate(samples):
        # 生成回答
        question = _get_question(example)
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        start_time = time.time()  # 用于计算生成速度（tokens/s），包含前向传播和解码时间，反映实际使用体验
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                temperature=config.EVAL_TEMPERATURE,
                do_sample=config.EVAL_TEMPERATURE > 0,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in generated:
            generated = generated.split("<|im_start|>assistant")[-1].strip()

        references.append(_get_reference(example))
        candidates.append(generated)

        logger.info(f"  [{label}] BERTScore: {i+1}/{len(samples)}, {len(generated) / (time.time() - start_time):.2f} tokens/s")
        #logging.info(f" [{label}] Gen: {i+1}/{n_show}, {len(generated) / (time.time() - start_time):.2f} tokens/s")
    #logger.info("", flush=True)

    # 调用 bert-score 库（batch 模式，比逐条调用快 ~10×）
    # model_type="bert-base-chinese" — 中文 BERT，单字级别 tokenization
    # lang="zh" — 语言标记（影响 tokenization 行为）
    # batch_size=16 — 批处理，平衡速度与内存
    precision, recall, f1 = bert_score_fn(
        candidates, references,
        model_type="bert-base-chinese",
        lang="zh",
        batch_size=16,
        verbose=False,
    )

    avg_p = precision.mean().item()
    avg_r = recall.mean().item()
    avg_f1 = f1.mean().item()

    logger.info(
        "  [%s] BERTScore P=%.4f  R=%.4f  F1=%.4f  (samples=%d)",
        label, avg_p, avg_r, avg_f1, len(samples),
    )
    return {
        "label": label,
        "bertscore_f1": round(avg_f1, 4),
        "bertscore_precision": round(avg_p, 4),
        "bertscore_recall": round(avg_r, 4),
        "num_samples": len(samples),
    }


# ============================================================
# 2.6 通用能力评估 — 蒸馏前后对比
# ============================================================

# 通用能力基准问题集（覆盖数学、科学、逻辑、中文、代码 5 个维度）
# 用于检测蒸馏过程中的灾难性遗忘（Catastrophic Forgetting）
GENERAL_BENCHMARK_QUESTIONS = [
    # ── 数学推理 ──
    {"question": "计算 123 × 456 的结果。", "type": "math"},
    {"question": "一个长方形的长是 8cm，宽是 5cm，求面积。", "type": "math"},
    {"question": "若 3x + 7 = 22，求 x 的值。", "type": "math"},
    {"question": "一个圆的半径是 10cm，求其周长（取 π=3.14）。", "type": "math"},
    # ── 科学常识 ──
    {"question": "请解释什么是牛顿第二定律。", "type": "science"},
    {"question": "为什么天空是蓝色的？请简要解释。", "type": "science"},
    {"question": "什么是光合作用？它对地球生态系统有何意义？", "type": "science"},
    {"question": "请解释电流、电压和电阻之间的关系（欧姆定律）。", "type": "science"},
    # ── 逻辑推理 ──
    {"question": "如果所有的猫都怕水，Tom 是一只猫，Tom 怕水吗？为什么？", "type": "logic"},
    {"question": "小明比小红高，小红比小刚高，谁最高？请推理。", "type": "logic"},
    {"question": "一个盒子里有 3 个红球和 5 个蓝球，随机摸出一个球，摸到红球的概率是多少？", "type": "logic"},
    {"question": "如果今天是星期三，100 天后是星期几？请推导。", "type": "logic"},
    # ── 中文能力 ──
    {"question": "请用'春风'为题写一首五言绝句。", "type": "chinese"},
    {"question": "请解释成语'画龙点睛'的含义和出处。", "type": "chinese"},
    {"question": "请将以下句子翻译成英文：'学而不思则罔，思而不学则殆。'", "type": "chinese"},
    {"question": "请用不超过 100 字概括《三国演义》的主要故事脉络。", "type": "chinese"},
    # ── 代码能力 ──
    {"question": "用 Python 实现斐波那契数列的前 20 项。", "type": "code"},
    {"question": "用 Python 写一个函数判断一个字符串是否为回文。", "type": "code"},
    {"question": "用 Python 实现二分查找算法。", "type": "code"},
    {"question": "用 Python 读取一个 CSV 文件并计算某一列的平均值。", "type": "code"},
    # -- C++ 代码能力（更接近模型实际训练环境，检测是否遗忘底层编程能力） --
    # {"question": "使用 C++11 实现一个线程安全的阻塞队列，支持多生产者多消费者模型。","type": "code","category": "concurrency","difficulty": "medium"},
    # {
    #     "question": "使用 C++ 实现一个高性能内存池，支持对象复用和自动回收。",
    #     "type": "code",
    #     "category": "memory",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "使用 C++ 实现一个 Reactor 网络模型，并支持 epoll 事件驱动。",
    #     "type": "code",
    #     "category": "network",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "设计并实现一个支持百万连接的 TCP 服务端框架。",
    #     "type": "design",
    #     "category": "architecture",
    #     "difficulty": "hard"
    # },
    # # -- Linux 系统编程能力（检测是否遗忘底层系统编程能力） --
    # {
    #     "question": "实现一个 Linux 版简易 top 命令，统计 CPU、内存和线程信息。",
    #     "type": "code",
    #     "category": "linux",
    #     "difficulty": "medium"
    # },
    # {
    #     "question": "实现一个文件监控服务，使用 inotify 实时监听目录变化。",
    #     "type": "code",
    #     "category": "linux",
    #     "difficulty": "medium"
    # },
    # {
    #     "question": "解释 Linux 零拷贝技术，并使用 sendfile 实现文件传输。",
    #     "type": "code",
    #     "category": "linux",
    #     "difficulty": "hard"
    # },
    # # -- FFmpeg 音视频处理能力（检测是否遗忘底层音视频处理能力） --
    # {
    #     "question": "使用 FFmpeg API 实现 H264 视频解码，并输出 YUV420P 数据。",
    #     "type": "code",
    #     "category": "ffmpeg",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "设计一个支持 RTMP、RTSP、HLS 的流媒体转发服务。",
    #     "type": "design",
    #     "category": "streaming",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "实现一个基于 FFmpeg 的视频转码服务，并支持 GPU 硬件加速。",
    #     "type": "code",
    #     "category": "ffmpeg",
    #     "difficulty": "hard"
    # },
    # # -- WebRTC 音视频处理能力（检测是否遗忘底层音视频处理能力） --
    # {
    #     "question": "实现 RTP 包乱序重组缓冲区，并支持丢包恢复。",
    #     "type": "code",
    #     "category": "webrtc",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "设计 WebRTC SFU 架构，支持万人直播。",
    #     "type": "design",
    #     "category": "webrtc",
    #     "difficulty": "expert"
    # },
    # {
    #     "question": "实现一个简化版 Jitter Buffer，并分析其延迟控制策略。",
    #     "type": "code",
    #     "category": "webrtc",
    #     "difficulty": "hard"
    # },
    # # -- CUDA 并行计算能力（检测是否遗忘底层并行计算能力） --
    # {
    #     "question": "使用 CUDA 实现矩阵乘法，并利用 Shared Memory 优化性能。",
    #     "type": "code",
    #     "category": "cuda",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "实现 CUDA Reduction 算法，并分析 Warp Divergence 问题。",
    #     "type": "code",
    #     "category": "cuda",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "设计一个 GPU 推理服务器，实现动态 Batch 调度。",
    #     "type": "design",
    #     "category": "gpu",
    #     "difficulty": "expert"
    # },
    # # -- LLM 推理能力（检测是否遗忘大模型推理能力） --
    # {
    #     "question": "实现 Transformer 的 Multi-Head Attention，并分析时间复杂度。",
    #     "type": "code",
    #     "category": "llm",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "实现 KV Cache，并分析其对推理性能的影响。",
    #     "type": "code",
    #     "category": "llm",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "设计一个支持 1000 QPS 的大模型推理服务。",
    #     "type": "design",
    #     "category": "llm",
    #     "difficulty": "expert"
    # },
    # {
    #     "question": "分析 vLLM Continuous Batching 的实现原理，并给出简化代码。",
    #     "type": "code",
    #     "category": "vllm",
    #     "difficulty": "expert"
    # },
    # # --   AI Infra
    # {
    #     "question": "设计一个支持 LoRA 热加载的大模型推理平台。",
    #     "type": "design",
    #     "category": "ai_infra",
    #     "difficulty": "expert"
    # },
    # {
    #     "question": "设计一个多租户 GPU 资源调度系统。",
    #     "type": "design",
    #     "category": "ai_infra",
    #     "difficulty": "expert"
    # },
    # {
    #     "question": "实现一个基于 Qdrant 的 RAG 检索系统。",
    #     "type": "code",
    #     "category": "rag",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "设计一个企业级 Agent 平台，支持 Tool Calling、Memory 和 Workflow。",
    #     "type": "design",
    #     "category": "agent",
    #     "difficulty": "expert"
    # },
    # # --    大模型训练
    # {
    #     "question": "实现 LoRA 微调算法，并解释 Rank 参数对训练效果的影响。",
    #     "type": "code",
    #     "category": "training",
    #     "difficulty": "hard"
    # },
    # {
    #     "question": "设计 Teacher-Student 蒸馏流程，将 70B 模型能力迁移到 4B 模型。",
    #     "type": "design",
    #     "category": "distillation",
    #     "difficulty": "expert"
    # },
    # {
    #     "question": "实现 DPO 训练流程，并比较 DPO 与 RLHF 的区别。",
    #     "type": "code",
    #     "category": "alignment",
    #     "difficulty": "expert"
    # },
]


def evaluate_general_ability(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    label: str,
) -> dict:
    """评估模型在通用知识问题上的表现，用于检测蒸馏后的灾难性遗忘。

    对 20 道跨领域（数学/科学/逻辑/中文/代码）通用问题进行推理，
    记录生成长度、截断率等基本统计。

    注意：通用题无参考答案，不做自动评分（ROUGE/BERTScore 需要参考答案）。
    评估结论通过对比 Base 和 LoRA 的生成长度变化和截断率来判断退化程度。

    Args:
        model: 待评估模型。
        tokenizer: 分词器。
        config: 全局配置。
        label: 模型标签。

    Returns:
        {
            "label": str,
            "num_questions": int,
            "avg_length": float,          # 平均生成长度（字符数）
            "min_length": int,
            "max_length": int,
            "truncation_rate": float,     # 截断率（达到 max_new_tokens 的占比）
            "category_stats": dict,       # 按类型分组的平均长度
        }
    """
    model.eval()
    lengths = []
    truncated = 0
    category_lengths = {}  # type → [lengths]

    for i, item in enumerate(GENERAL_BENCHMARK_QUESTIONS):
        q = item["question"]
        q_type = item["type"]

        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        start_time = time.time()  # 用于计算生成速度（tokens/s），包含前向传播和解码时间，反映实际使用体验  
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].size(1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                temperature=config.EVAL_TEMPERATURE,
                do_sample=config.EVAL_TEMPERATURE > 0,
            )

        # 计算实际生成的 token 数
        gen_tokens = outputs.size(1) - input_len
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in generated:
            generated = generated.split("<|im_start|>assistant")[-1].strip()

        gen_len = len(generated)
        lengths.append(gen_len)

        if gen_tokens >= config.EVAL_MAX_NEW_TOKENS * 0.95:
            truncated += 1

        category_lengths.setdefault(q_type, []).append(gen_len)

        logger.info(f"\r  [{label}] General: {i+1}/{len(GENERAL_BENCHMARK_QUESTIONS)}, {len(generated) / (time.time() - start_time):.2f} tokens/s")
        del inputs, outputs  # 及时删除中间变量，释放显存
    #logger.info("", flush=True)
    torch.cuda.synchronize()  # 等待GPU操作完成
    torch.cuda.empty_cache()  # 清空 GPU 缓存，释放显存占用
    n = len(GENERAL_BENCHMARK_QUESTIONS)
    avg_len = sum(lengths) / n if n > 0 else 0.0
    trunc_rate = truncated / n if n > 0 else 0.0

    # 按类别统计平均长度
    category_stats = {
        cat: {
            "avg_length": round(sum(ls) / len(ls), 0) if ls else 0,
            "num_questions": len(ls),
        }
        for cat, ls in category_lengths.items()
    }

    logger.info(
        "  [%s] General: avg_len=%.0f chars, trunc_rate=%.1f%%",
        label, avg_len, trunc_rate * 100,
    )

    return {
        "label": label,
        "num_questions": n,
        "avg_length": round(avg_len, 0),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "truncation_rate": round(trunc_rate, 4),
        "category_stats": category_stats,
    }


# ============================================================
# 3. 生成样本收集
# ============================================================
def collect_generation_samples(
    model,
    tokenizer: PreTrainedTokenizer,
    config: Config,
    samples: list,
    label: str,
    n_show: int = 5,
) -> list:
    """对少量样本生成回答，用于报告中并排对比。

    Args:
        n_show: 在报告中展示的样本数量（从 samples 中取前 N 条）。

    Returns:
        [{"instruction": str, "reference": str, "generated": str}, ...]
    """
    model.eval()
    results = []
    # show_samples= [];
    # 随机选择 samples 中的 n_show 条进行展示，保证每次评测报告的样本多样性；如果样本数量不足 n_show，则展示全部样本。
    # if len(samples) > n_show:
    #     show_samples = random.sample(samples, n_show);
    #     #show_samples = random.sample(samples, n_show)
    # else:        show_samples = samples[:n_show]

   # gc.collect()  # 手动触发 Python 垃圾回收，释放未使用的内存
    #torch.cuda.empty_cache()  # 清空 PyTorch GPU 缓存，释放显存占用
    logger.info(f">>> 收集 {label} 模型生成样本 ，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    for i, example in enumerate(samples):
        question = _get_question(example)
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if config.EVAL_SHOW_SAMPLES:
            logging.info(f"  [{label}] Generating sample {i+1}/{len(samples)}: {question}");
        # start_time 用于计算生成速度（tokens/s），包含前向传播和解码时间，反映实际使用体验
        start_time = time.time();
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                temperature=config.EVAL_TEMPERATURE,
                do_sample=config.EVAL_TEMPERATURE > 0,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|im_start|>assistant" in generated:
            generated = generated.split("<|im_start|>assistant")[-1].strip()

        results.append({
            "instruction": question,
            "reference": _get_reference(example),
            "generated": generated,
        })
        if config.EVAL_SHOW_SAMPLES:
            logging.info(f"  [{label}] Generated sample {i+1}/{len(samples)} (len={len(generated)}) : {generated} ");
        logging.info(f" [{label}] Gen: {i+1}/{n_show}, {len(generated) / (time.time() - start_time):.2f} tokens/s")
        del inputs, outputs  # 及时删除中间变量，释放显存
    # print("", flush=True)  # 换行
    # 3. 清理 inputs，以防后面变长
    #del inputs
    torch.cuda.synchronize()  # 等待GPU操作完成
    torch.cuda.empty_cache()  # 清空 GPU 缓存，释放显存占用
    return results


# ============================================================
# 4. LLM-as-Judge 评估（大模型打分）
# ============================================================
# 蒸馏评测专用 prompt（Base vs LoRA vs Teacher 三方对比）
DISTILLATION_EVAL_PROMPT = """你是一位严谨的 AI Infra 领域技术评测专家。你的任务是对一个”学生模型”的回答进行多维度质量评估，并通过对比”基类模型”和”教师模型”的输出，判断蒸馏训练是否带来了正向增益。

评估时请严格遵循以下原则：
1. “教师模型”的回答代表高质量标准，但它不是唯一正确的表达方式。学生模型的回答若语义等价、技术正确但措辞不同，不应扣分。
2. “基类模型”的回答代表模型在蒸馏/微调前的初始能力，用于判断学生模型是否真正取得了进步。
3. 你需要对每个维度独立打分，然后给出综合分和对比分析。
4. 最终只输出 JSON，不要包含任何额外文字或 markdown 标记。

【输入信息】
- 问题：{question}
- 基类模型回答（蒸馏前）：{baseline}
- 教师模型回答（高质量标准）：{reference}
- 学生模型回答（蒸馏后）：{generated}

【评分维度与标准】

1. **accuracy（技术准确性）**：回答中的技术事实、概念、命令、参数、架构等是否正确。
   - 5分：完全正确，无任何事实错误或误导性陈述。
   - 4分：主体正确，但有极个别不影响整体理解的轻微瑕疵。
   - 3分：大部分正确，但存在一处事实性错误或模糊表述。
   - 2分：多处事实性错误，或存在关键性错误。
   - 1分：几乎全错，或与问题完全无关。

2. **relevance（相关性）**：回答是否紧扣问题核心，不跑题、不包含无关信息。
   - 5分：完全命中问题要点，回答紧密围绕提问，无任何偏离。
   - 4分：基本相关，存在极少量的次要无关内容。
   - 3分：部分相关，但包含了明显的无关信息，或未直击问题核心。
   - 2分：大部分内容与问题关联度低。
   - 1分：答非所问，或完全偏离主题。

3. **completeness（关键点覆盖度）**：与教师模型回答相比，是否覆盖了需要解决该问题所必需的关键要点、步骤或论据。
   - 5分：覆盖了教师回答中的所有关键点（允许不同的组织方式），无重要遗漏。
   - 4分：覆盖了绝大多数关键点，仅缺一个次要细节。
   - 3分：覆盖了主要关键点，但缺失了部分重要信息。
   - 2分：只触及少量关键点，存在重大遗漏。
   - 1分：几乎没有覆盖任何关键点。

4. **clarity（清晰度与可读性）**：表达是否清晰、逻辑是否通顺、结构是否易于理解。
   - 5分：表达极其清晰，逻辑严密，结构优秀，易于阅读和理解。
   - 4分：清晰，逻辑通顺，但不完美（如存在小部分啰嗦）。
   - 3分：基本清晰，但部分表述费解，或结构有些混乱。
   - 2分：多处表达含糊不清，影响理解。
   - 1分：严重混乱，几乎不可读。

5. **overall（综合质量）**：综合以上四个维度，以及考虑回答在实际场景中的有用性，给出的整体评分。
   - 注意：综合分不是简单的平均，需要你根据专业判断权衡各维度的重要性。
   - 5分：各方面均优秀，可直接采纳为高质量答案。
   - 4分：良好，有一处小瑕疵，但仍然很有用。
   - 3分：勉强可用，但存在明显不足。
   - 2分：较差，实用价值低。
   - 1分：完全不可用。

【对比分析】
除了对以上维度打分外，你还需要生成两个辅助字段：
- “improvement_over_baseline”：用一句话说明相较于基类模型，学生模型有何改进或退化。
- “gap_to_teacher”：用一句话说明学生模型与教师模型相比，还有哪些主要差距。

【输出格式】
请严格按照下面的 JSON 结构输出，只输出 JSON，不要包括 ```json、解释性文字或 markdown 标记：

{{
  “accuracy”: <1-5整数>,
  “relevance”: <1-5整数>,
  “completeness”: <1-5整数>,
  “clarity”: <1-5整数>,
  “overall”: <1-5整数>,
  “comment”: “<基于各维度的一句话中文点评，指出学生回答的主要优点和可改进之处>”,
  “improvement_over_baseline”: “<中文，描述相对基类模型的进步或不足>”,
  “gap_to_teacher”: “<中文，描述与教师模型的主要差距>”
}}
"""


def _build_judge_prompt(question: str, reference: str, generated: str, baseline: str = "") -> str:
    """构造给裁判模型的三方对比评估 prompt。

    Args:
        question: 用户问题。
        reference: 教师模型回答（高质量标准）。
        generated: 学生模型回答（蒸馏后/待评估）。
        baseline: 基类模型回答（蒸馏前），空字符串表示无基线对比。
    """
    return DISTILLATION_EVAL_PROMPT.format(
        question=question,
        baseline=baseline or "（无基类模型输出，请仅对比学生模型与教师模型）",
        reference=reference,
        generated=generated,
    )


def _call_llm_judge(
    endpoint: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout: int = 600,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    seed: int = 42,
    max_retries: int = 2,
) -> dict:
    """调用外部 LLM API 进行质量评分。

    使用 LLMClient 公共类，兼容 OpenAI Chat Completions API。
    """
    client = LLMClient(endpoint, model, api_key, timeout, max_retries)
    return client.chat_json(
        prompt,
        system="You are a technical quality evaluator. Always respond in Chinese JSON format.",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
    )


def evaluate_with_llm_judge(
    config: Config,
    base_samples: list,
    lora_samples: list,
) -> dict:
    """使用外部 LLM 对学生模型生成质量进行多维度评分（Base vs LoRA vs Teacher 三方对比）。

    Args:
        config: 全局配置（含 LLM Judge 连接参数）。
        base_samples: Base 模型生成样本 [{"instruction", "reference", "generated"}, ...]。
        lora_samples: LoRA 模型生成样本。

    Returns:
        {
            "enabled": bool,
            "model": str,
            "num_samples": int,
            "lora_scores": {"accuracy": avg, ...},
            "base_scores": {...},
            "improvement_summary": [str, ...],   # 相对基类提升摘要
            "gap_summary": [str, ...],           # 相对教师差距摘要
            "details": [...],
        }
    """
    if not config.EVAL_LLM_JUDGE_ENABLED:
        logger.info(">>> LLM-as-Judge: 未启用（设置 eval.llm_judge.enabled=true 开启）")
        return {"enabled": False}

    # API Key 优先级：config 值 > 环境变量
    api_key = config.EVAL_LLM_JUDGE_API_KEY or os.environ.get("LLM_JUDGE_API_KEY", "")
    if not api_key:
        logger.warning(">>> LLM-as-Judge: 已启用但未配置 api_key，跳过")
        return {"enabled": False, "error": "no_api_key"}

    logger.info(">>> LLM-as-Judge: 开始三方对比评估 (Base vs LoRA vs Teacher)")
    logger.info("  Endpoint: %s", config.EVAL_LLM_JUDGE_ENDPOINT)
    logger.info("  Model:    %s", config.EVAL_LLM_JUDGE_MODEL)
    logger.info("  Samples:  %d (max)", config.EVAL_LLM_JUDGE_MAX_SAMPLES)

    n_samples = min(config.EVAL_LLM_JUDGE_MAX_SAMPLES, len(lora_samples))

    samples_to_score = [];
    # 随机选择 n_samples 条 LoRA 样本进行评估，保证每次评测报告的样本多样性；如果样本数量不足 n_samples，则评估全部样本。
    if len(lora_samples) > n_samples:
        samples_to_score = random.sample(lora_samples, n_samples);
        #samples_to_score = random.sample(lora_samples, n_samples)
    else:        samples_to_score = lora_samples[:n_samples];
    # samples_to_score = lora_samples[:n_samples]
    # 匹配 base 样本
    base_map = {}
    for b in base_samples:
        base_map[b["instruction"]] = b["generated"]

    all_lora_scores = {key: [] for key in _JUDGE_CRITERIA}
    all_base_scores = {key: [] for key in _JUDGE_CRITERIA}
    improvement_summary = []
    gap_summary = []
    details = []

    for i, sample in enumerate(samples_to_score):
        question = sample["instruction"]
        reference = sample["reference"]
        lora_gen = sample["generated"]
        base_gen = base_map.get(question, "")

        logger.info("  [%d/%d] 评估: %s", i + 1, n_samples, question[:80])

        # ── 三方对比：LoRA（学生） + Base（基线） + Teacher（标准） ──
        prompt_lora = _build_judge_prompt(
            question, reference, lora_gen, baseline=base_gen,
        )
        lora_result = _call_llm_judge(
            config.EVAL_LLM_JUDGE_ENDPOINT,
            config.EVAL_LLM_JUDGE_MODEL,
            api_key,
            prompt_lora,
            timeout=config.EVAL_LLM_JUDGE_TIMEOUT,
            temperature=config.EVAL_LLM_JUDGE_TEMPERATURE,
            max_tokens=config.EVAL_LLM_JUDGE_MAX_TOKENS,
            top_p=config.EVAL_LLM_JUDGE_TOP_P,
            seed=config.EVAL_LLM_JUDGE_SEED,
            max_retries=config.EVAL_LLM_JUDGE_MAX_RETRIES,
        )

        ##time.sleep(0.5)  # 避免 API 限流

        # ── Base 单独评估（无基线对比，Base 本身就是基线） ──
        prompt_base = _build_judge_prompt(
            question, reference, base_gen, baseline="",
        )
        base_result = _call_llm_judge(
            config.EVAL_LLM_JUDGE_ENDPOINT,
            config.EVAL_LLM_JUDGE_MODEL,
            api_key,
            prompt_base,
            timeout=config.EVAL_LLM_JUDGE_TIMEOUT,
            temperature=config.EVAL_LLM_JUDGE_TEMPERATURE,
            max_tokens=config.EVAL_LLM_JUDGE_MAX_TOKENS,
            top_p=config.EVAL_LLM_JUDGE_TOP_P,
            seed=config.EVAL_LLM_JUDGE_SEED,
            max_retries=config.EVAL_LLM_JUDGE_MAX_RETRIES,
        )

        # 收集 5 维度分数
        for key in _JUDGE_CRITERIA:
            if isinstance(lora_result.get(key), (int, float)):
                all_lora_scores[key].append(lora_result[key])
            if isinstance(base_result.get(key), (int, float)):
                all_base_scores[key].append(base_result[key])

        # 收集对比分析
        lora_improvement = lora_result.get("improvement_over_baseline", "")
        lora_gap = lora_result.get("gap_to_teacher", "")
        if lora_improvement:
            improvement_summary.append(lora_improvement)
        if lora_gap:
            gap_summary.append(lora_gap)

        details.append({
            "instruction": question[:200],
            "reference": reference[:300],
            "lora_generated": lora_gen[:300],
            "base_generated": base_gen[:300],
            "lora_scores": {k: lora_result.get(k) for k in _JUDGE_CRITERIA},
            "base_scores": {k: base_result.get(k) for k in _JUDGE_CRITERIA},
            "lora_comment": lora_result.get("comment", ""),
            "base_comment": base_result.get("comment", ""),
            "improvement_over_baseline": lora_improvement,
            "gap_to_teacher": lora_gap,
        })

    # 计算平均分
    lora_avg = {k: round(sum(v) / len(v), 2) if v else 0.0 for k, v in all_lora_scores.items()}
    base_avg = {k: round(sum(v) / len(v), 2) if v else 0.0 for k, v in all_base_scores.items()}

    logger.info("  LoRA 均分: %s", json.dumps(lora_avg, ensure_ascii=False))
    logger.info("  Base 均分: %s", json.dumps(base_avg, ensure_ascii=False))

    return {
        "enabled": True,
        "model": config.EVAL_LLM_JUDGE_MODEL,
        "endpoint": config.EVAL_LLM_JUDGE_ENDPOINT,
        "num_samples": n_samples,
        "lora_scores": lora_avg,
        "base_scores": base_avg,
        "improvement_summary": improvement_summary,
        "gap_summary": gap_summary,
        "details": details,
    }


# ============================================================
# 5. 报告生成
# ============================================================
def generate_report(
    config: Config,
    ppl_results: list,
    rouge_results: list,
    base_samples: list,
    lora_samples: list,
    llm_judge_results: dict = None,
    bertscore_results: list = None,
    general_results: list = None,
) -> str:
    """生成 Markdown 格式的评测报告字符串。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- PPL 对比表 ----
    ppl_rows = ""
    ppl_disabled_msg = ""
    if ppl_results:
        for r in ppl_results:
            ppl_rows += f"| {r['label']} | {r['avg_loss']:.4f} | {r['avg_ppl']:.2f} |\n"
    else:
        ppl_disabled_msg = (
            "> PPL 已禁用（`eval.ppl.enabled: false`）。"
            "如需启用，在 config.yaml 中设为 true。\n\n"
        )

    # ---- ROUGE 对比表 ----
    rouge_rows = ""
    rouge_disabled_msg = ""
    if rouge_results:
        for r in rouge_results:
            rouge_rows += f"| {r['label']} | {r['rouge_l_f1']:.4f} | {r['num_samples']} |\n"
    else:
        rouge_disabled_msg = (
            "> ROUGE-L 已禁用（`eval.rouge.enabled: false`）。"
            "如需启用，在 config.yaml 中设为 true。\n"
            "> 替代方案：BERTScore（2.5 节）基于语义向量相似度，"
            "更适合评估 LLM 改写输出。\n\n"
        )

    # ---- BERTScore 对比表 ----
    bertscore_rows = ""
    bertscore_header = ""
    if bertscore_results:
        # 仅在有 enabled=True 的结果时才生成 BERTScore 章节
        enabled_results = [r for r in bertscore_results if r.get("enabled", True)]
        if enabled_results:
            bertscore_header = "| 模型 | F1 | Precision | Recall | 样本数 |\n|------|-----|-----------|--------|--------|\n"
            for r in bertscore_results:
                if r.get("enabled") is False:
                    bertscore_rows += f"| {r['label']} | ⚠ 未安装bert-score | — | — | — |\n"
                else:
                    bertscore_rows += (
                        f"| {r['label']} | {r['bertscore_f1']:.4f} | "
                        f"{r['bertscore_precision']:.4f} | {r['bertscore_recall']:.4f} | "
                        f"{r['num_samples']} |\n"
                    )

    # ---- 生成样本并排对比 ----
    sample_blocks = ""
    for i, (bs, ls) in enumerate(zip(base_samples, lora_samples)):
        # 截断过长的文本
        ref_short = bs["reference"][:300] + ("..." if len(bs["reference"]) > 300 else "")
        base_short = bs["generated"][:500] + ("..." if len(bs["generated"]) > 500 else "")
        lora_short = ls["generated"][:500] + ("..." if len(ls["generated"]) > 500 else "")

        sample_blocks += f"""### 样本 {i + 1}

**问题**: {bs["instruction"]}

| 角色 | 内容 |
|------|------|
| 📝 参考答案 | {ref_short} |
| 🔴 Base 模型 | {base_short} |
| 🟢 LoRA 模型 | {lora_short} |

---

"""

    # ---- LLM-as-Judge 评估结果 ----
    llm_judge_section = ""
    if llm_judge_results and llm_judge_results.get("enabled"):
        jr = llm_judge_results
        criteria_names = list(_JUDGE_CRITERIA.keys())
        # 综合均分对比表
        header = "| 维度 | Base | LoRA | Δ |\n|------|------|------|-----|\n"
        rows = ""
        for k in criteria_names:
            b = jr.get("base_scores", {}).get(k, "-")
            l = jr.get("lora_scores", {}).get(k, "-")
            if isinstance(b, (int, float)) and isinstance(l, (int, float)):
                delta = l - b
                delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            else:
                delta_str = "-"
            rows += f"| {k} | {b} | {l} | {delta_str} |\n"

        # 逐条详情
        detail_blocks = ""
        for i, d in enumerate(jr.get("details", [])):
            lora_comment = d.get("lora_comment", "")
            base_comment = d.get("base_comment", "")
            detail_blocks += f"""### Judge 样本 {i + 1}

**问题**: {d["instruction"]}

| 维度 | Base | LoRA |
|------|------|------|
"""
            for k in criteria_names:
                bs = d.get("base_scores", {}).get(k, "-")
                ls = d.get("lora_scores", {}).get(k, "-")
                detail_blocks += f"| {k} | {bs} | {ls} |\n"
            improvement = d.get("improvement_over_baseline", "")
            gap = d.get("gap_to_teacher", "")
            detail_blocks += f"""
> 🔴 **Base 点评**: {base_comment}

> 🟢 **LoRA 点评**: {lora_comment}
"""
            if improvement:
                detail_blocks += f"\n> 📈 **相对基类改进**: {improvement}\n"
            if gap:
                detail_blocks += f"\n> 🎯 **与教师差距**: {gap}\n"
            detail_blocks += "\n---\n"

        # 汇总蒸馏增益分析
        improvement_list = ""
        for imp in jr.get("improvement_summary", [])[:5]:
            improvement_list += f"- 📈 {imp}\n"
        gap_list = ""
        for g in jr.get("gap_summary", [])[:5]:
            gap_list += f"- 🎯 {g}\n"

        llm_judge_section = f"""## 4. LLM-as-Judge 质量评估（三方对比）

> 裁判模型：{jr.get("model", "N/A")}
> 评估样本数：{jr.get("num_samples", 0)}
> 评估模式：**Base（基类/蒸馏前） vs LoRA（学生/蒸馏后） vs Teacher（教师/高质量标准）**
> 评分维度：准确性、相关性、完整性、清晰度、整体质量（1-5 分，5 维度）

### 综合均分对比

{header}{rows}

> **Δ (LoRA - Base)**: 正值为提升，负值为退化。

### 蒸馏增益分析

{improvement_list}

### 与教师差距

{gap_list}

### 逐条详细评分

{detail_blocks}

---

"""

    report = f"""# AI Infra LoRA SFT 训练评测报告

> 生成时间：{now}
> 基座模型：{config.MODEL_ID}
> LoRA 输出：{config.OUTPUT_DIR}
> 评估样本数：{config.EVAL_NUM_SAMPLES}

---

## 1. Perplexity (PPL) 对比

PPL 衡量模型对文本的"惊讶程度"，越低越好。
但 PPL 低 ≠ 生成质量高，需结合 ROUGE 和人工评估。

{ppl_disabled_msg}| 模型 | Avg Loss | Avg PPL |
|------|----------|---------|
{ppl_rows}

> **解读**: 若 LoRA 模型 PPL 远低于 Base，说明模型已学到数据分布特征。
> 若 LoRA 模型 PPL 接近于 1，可能是过拟合信号。

---

## 2. ROUGE-L F1 对比

ROUGE-L 基于最长公共子序列（LCS），衡量生成答案与参考答案的字符级重叠度。
F1 越高，生成结果越接近参考答案。

{rouge_rows}{rouge_disabled_msg}

> **解读**: ROUGE-L 对开放式长回答不太敏感（答案可以不同但都正确）。
> F1 > 0.3 通常表示内容有实质重叠。配合人工抽查判断质量。

---

## 2.5 BERTScore 语义相似度对比

BERTScore 基于 BERT 语义向量余弦相似度，衡量生成答案与参考答案的语义等价性。
与 ROUGE-L 不同，BERTScore 能识别"意思相同但表述不同"的改写——这对 LLM 评估尤为重要。

{bertscore_header}{bertscore_rows}

> **解读**:
> - BERTScore F1 > 0.85: 生成回答与参考答案语义高度一致
> - BERTScore F1 0.70-0.85: 语义基本一致，有一定改写
> - BERTScore F1 < 0.70: 语义偏离较大，需关注内容准确性
> - BERTScore 依赖 `bert-base-chinese` 模型（~440MB），首次运行会自动下载
> - 若未安装 bert-score 库：`pip install bert-score`

---

## 3. 生成样本对比

{'以下对比 Base 模型与 LoRA 模型对相同问题的回答（前 5 条）。' if sample_blocks else '> 生成样本对比已禁用（`eval.gen_samples.enabled: false`）。'}

{sample_blocks}
{llm_judge_section}"""

    # ── 4.5 通用能力对比表 ──
    general_section = ""
    if general_results and len(general_results) == 2:
        base_gen = general_results[0]
        lora_gen = general_results[1]

        # 构建类别对比表
        all_cats = sorted(set(
            list(base_gen.get("category_stats", {}).keys()) +
            list(lora_gen.get("category_stats", {}).keys())
        ))
        cat_header = "| 类别 | Base 平均长度 | LoRA 平均长度 | Δ |\n|------|-------------|-------------|-----|\n"
        cat_rows = ""
        for cat in all_cats:
            b_avg = base_gen.get("category_stats", {}).get(cat, {}).get("avg_length", "-")
            l_avg = lora_gen.get("category_stats", {}).get(cat, {}).get("avg_length", "-")
            if isinstance(b_avg, (int, float)) and isinstance(l_avg, (int, float)):
                delta = l_avg - b_avg
                delta_str = f"+{delta:.0f}" if delta > 0 else f"{delta:.0f}"
                pct = (delta / b_avg * 100) if b_avg > 0 else 0
                delta_str += f" ({pct:+.1f}%)"
            else:
                delta_str = "-"
            cat_rows += f"| {cat} | {b_avg} | {l_avg} | {delta_str} |\n"

        general_section = f"""## 4.5 通用能力评估（灾难性遗忘检测）

> 基准问题集：{base_gen.get("num_questions", 20)} 道跨领域问题（数学/科学/逻辑/中文/代码）
> 无参考答案，仅对比生成长度和截断率变化以检测退化

### 总体对比

| 指标 | Base | LoRA | Δ |
|------|------|------|-----|
| 平均生成长度 | {base_gen["avg_length"]:.0f} | {lora_gen["avg_length"]:.0f} | {lora_gen["avg_length"] - base_gen["avg_length"]:+.0f} ({(lora_gen["avg_length"] - base_gen["avg_length"]) / max(base_gen["avg_length"], 1) * 100:+.1f}%) |
| 截断率 | {base_gen["truncation_rate"]:.1%} | {lora_gen["truncation_rate"]:.1%} | {lora_gen["truncation_rate"] - base_gen["truncation_rate"]:+.1%} |

### 按类别对比

{cat_header}{cat_rows}

> **解读**:
> - 生成长度变化 < 20%: 正常波动范围
> - 生成长度显著下降（> 30%）: 可能存在灾难性遗忘，模型失去了通用知识
> - 截断率显著上升: 模型可能倾向于生成过短回答（"偷懒"行为）
> - 数学/逻辑类变化大: 蒸馏可能影响推理能力，需特别关注

---

"""

    # ── 5. 蒸馏效果综合判断 ──
    # 提取数据用于自动判断
    ppl_base = next((r["avg_ppl"] for r in ppl_results if r["label"] == "Base"), None)
    ppl_lora = next((r["avg_ppl"] for r in ppl_results if r["label"] == "LoRA"), None)
    rouge_lora = next((r["rouge_l_f1"] for r in rouge_results if r["label"] == "LoRA"), None)
    bert_base = next((r.get("bertscore_f1") for r in (bertscore_results or []) if r["label"] == "Base" and r.get("enabled", True)), None)
    bert_lora = next((r.get("bertscore_f1") for r in (bertscore_results or []) if r["label"] == "LoRA" and r.get("enabled", True)), None)
    gen_len_delta_pct = 0.0
    gen_trunc_delta = 0.0
    if general_results and len(general_results) == 2:
        base_avg = general_results[0]["avg_length"]
        lora_avg = general_results[1]["avg_length"]
        gen_len_delta_pct = (lora_avg - base_avg) / max(base_avg, 1) * 100
        gen_trunc_delta = general_results[1]["truncation_rate"] - general_results[0]["truncation_rate"]

    # 判断逻辑
    checks = []
    if ppl_base is not None and ppl_lora is not None:
        if ppl_lora < ppl_base * 0.95:
            checks.append(("✅", "PPL 显著下降", f"LoRA PPL={ppl_lora:.1f} < Base PPL={ppl_base:.1f}"))
        elif ppl_lora < ppl_base:
            checks.append(("✅", "PPL 小幅下降", f"LoRA PPL={ppl_lora:.1f} < Base PPL={ppl_base:.1f}"))
        else:
            checks.append(("⚠️", "PPL 未下降", f"LoRA PPL={ppl_lora:.1f} >= Base PPL={ppl_base:.1f}，模型未学到数据分布"))
    else:
        checks.append(("❓", "PPL 数据缺失", "无法评估"))

    if rouge_lora is not None:
        if rouge_lora > 0.3:
            checks.append(("✅", "ROUGE-L > 0.3", f"LoRA ROUGE-L={rouge_lora:.4f}，与参考答案有实质重叠"))
        elif rouge_lora > 0.15:
            checks.append(("⚠️", "ROUGE-L 偏低", f"LoRA ROUGE-L={rouge_lora:.4f}，内容重叠较少"))
        else:
            checks.append(("❌", "ROUGE-L 过低", f"LoRA ROUGE-L={rouge_lora:.4f}，可能未学到内容"))
    else:
        checks.append(("❓", "ROUGE-L 数据缺失", "无法评估"))

    if bert_base is not None and bert_lora is not None:
        bert_delta = bert_lora - bert_base
        if bert_delta > 0.05:
            checks.append(("✅", "BERTScore 显著提升", f"Δ={bert_delta:+.4f}，语义一致性增强"))
        elif bert_delta > -0.03:
            checks.append(("✅", "BERTScore 基本持平", f"Δ={bert_delta:+.4f}，语义一致性维持"))
        else:
            checks.append(("⚠️", "BERTScore 下降", f"Δ={bert_delta:+.4f}，语义一致性退化"))

    if general_results and len(general_results) == 2:
        if abs(gen_len_delta_pct) < 20:
            checks.append(("✅", "通用能力基本维持", f"生成长度变化={gen_len_delta_pct:+.1f}%，在正常范围"))
        elif gen_len_delta_pct > -30:
            checks.append(("⚠️", "通用能力轻微下降", f"生成长度变化={gen_len_delta_pct:+.1f}%"))
        else:
            checks.append(("❌", "通用能力严重退化", f"生成长度变化={gen_len_delta_pct:+.1f}%，疑似灾难性遗忘"))
        if gen_trunc_delta > 0.1:
            checks.append(("⚠️", "截断率上升", f"LoRA 截断率增加 {gen_trunc_delta:+.1%}，模型可能倾向过短回答"))

    # 生成判定结论
    has_error = any(c[0] == "❌" for c in checks)
    has_warning = any(c[0] == "⚠️" for c in checks)
    if has_error:
        verdict = "❌ FAIL — 存在严重问题，建议检查训练数据和参数后重训"
    elif has_warning:
        verdict = "⚠️ PASS WITH WARNINGS — 蒸馏基本成功，但有些指标需要关注"
    else:
        verdict = "✅ PASS — 领域能力显著提升，通用能力无明显退化，蒸馏成功"

    check_rows = ""
    for icon, name, detail in checks:
        check_rows += f"| {icon} {name} | {detail} |\n"

    conclusion_section = f"""{general_section}## 5. 蒸馏效果综合判断

### 逐项检查

| 检查项 | 详情 |
|--------|------|
{check_rows}

### 综合判定

**{verdict}**

### 下一步建议

"""
    if has_error:
        conclusion_section += """1. ❌ ROUGE-L 过低或 PPL 不降 → 检查 DataCollator（labels 是否全 mask）、chat_template 格式
2. ❌ 通用能力严重退化 → 降低学习率（1e-4）、增大 dropout（0.15）、减少 LoRA rank（4-8）
3. ❌ 尝试混合少量通用数据到训练集中（如 5% 通用 SFT 数据）以减轻遗忘
"""
    elif has_warning:
        conclusion_section += """1. ⚠️ 关注 PPL 趋势 → 若持续不降，考虑增加 epoch 或调整 lr
2. ⚠️ ROUGE-L 偏低 → 检查教师回答是否过长（>4096 tokens），考虑增加 max_response_length
3. ⚠️ 通用能力波动 → 正常现象，下次训练时用 benchmark 定量确认
4. ℹ️ 所有建议请结合人工抽查 3-5 条生成样本做最终判断
"""
    else:
        conclusion_section += """1. ✅ 蒸馏效果良好，可考虑导出模型：`python scripts/export.py`
2. ✅ 建议用 Benchmark 工具（P0-2）做正式通用能力验证
3. ✅ 可尝试增大数据集或训练更多 epoch 进一步提升领域能力
"""

    conclusion_section += """
---

*报告由 eval.py 自动生成*
"""
    report += conclusion_section
    return report


# ============================================================
# 5. 主评估流程
# ============================================================
def _detect_adapter_dir(config: Config) -> str:
    """自动探测最新训练的 LoRA adapter 目录。

    查找优先级：
      1. config.OUTPUT_DIR（如果包含 adapter_config.json）
      2. runs/train/exp{N}/ 下最新包含 adapter_config.json 的目录（按修改时间）
      3. 报错退出

    Returns:
        str: LoRA adapter 目录路径。

    Raises:
        FileNotFoundError: 未找到任何有效的 adapter 目录。
    """
    # 优先级 1: config.yaml 中的 output_dir
    output_dir = Path(config.OUTPUT_DIR)
    if (output_dir / "adapter_config.json").exists():
        return str(output_dir)

    # 优先级 2: 从实验目录中找最新的
    runs_dir = Path(config.RUNS_DIR)
    if runs_dir.is_dir():
        exp_dirs = sorted(
            [d for d in runs_dir.iterdir()
             if d.is_dir() and d.name.startswith("exp") and (d / "adapter_config.json").exists()],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if exp_dirs:
            latest = exp_dirs[0]
            logger.info("自动检测到最新实验目录: %s", latest.resolve())
            return str(latest)

    # 都没找到
    raise FileNotFoundError(
        f"未找到任何有效的 LoRA adapter 目录！\n"
        f"请先运行训练: python scripts/train.py\n"
        f"或手动将 adapter 放到: {output_dir}\n"
        f"已搜索:\n"
        f"  - {output_dir} (config.output_dir)\n"
        f"  - runs/train/exp{{N}}/ (实验目录)\n"
        f"\n提示: 训练时 output_dir 会自动重定向到 runs/train/exp{{N}}/，\n"
        f"      如需 --eval-only，请确保 config.yaml 中 output_dir 指向有效 adapter。"
    )


def run_evaluation(config: Config, tokenizer: PreTrainedTokenizer) -> None:
    """执行完整评估流水线，输出 eval_report.md 和 eval_results.json。

    步骤：
      1. 加载数据集并选取评估样本
      2. 加载 Base 模型 → 计算 PPL / ROUGE / 收集生成样本
      3. 加载 LoRA 模型 → 计算 PPL / ROUGE / 收集生成样本
      4. 生成 Markdown 报告 + JSON 结果
    """
    logger.info("=" * 60)
    logger.info("  训练质量评估")
    logger.info("=" * 60)

    # ---- 加载评估数据（仅 Parquet 格式） ----
    data_dir = Path(config.DATA_DIR)
    data_files = sorted(data_dir.glob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(
            f"数据目录下未找到 .parquet 文件: {data_dir}\n"
            f"请先运行: python poly_distill/json_to_parquet.py"
        )
    logger.info("评估数据: Parquet, 文件数: %d", len(data_files))

    datasets_list = []
    for f in data_files:
        ds = load_dataset("parquet", data_files=str(f), split="train")
        datasets_list.append(ds)
    dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    n_total = len(dataset)

    # 选取子集（随机但固定种子，保证可复现）
    n_eval = config.EVAL_NUM_SAMPLES if config.EVAL_NUM_SAMPLES > 0 else n_total
    n_eval = min(n_eval, n_total)
    rng = random.Random(42)  # 固定种子，结果可复现
    indices = rng.sample(range(n_total), n_eval)
    eval_samples = [dataset[i] for i in indices]
    logger.info("评估数据集: %d / %d 条样本（随机种子=42）", n_eval, n_total)
    logger.info(f">>> 选取的评估样本示例: {_get_question(eval_samples[0])}");
    # ---- 加载 Base 模型 ----
    logger.info("--- 加载 Base 模型: %s ---", config.MODEL_ID)
    logger.info(f">>> 加载Base 模型 ，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        # device_map="auto",
        device_map="cuda:0",  # 不要用 auto！
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # ← 这行最关键
    )
    base_model = base_model.to('cuda')
    logger.info("\n=== 显存与加速诊断 ===")
    logger.info(f"Flash Attention 2 已安装: {is_flash_attn_2_available()}")
    logger.info(f"模型注意力实现: {base_model.config._attn_implementation}")

    # 检查参数是否都在 GPU
    on_cpu = sum(1 for p in base_model.parameters() if p.device.type == 'cpu')
    on_cuda = sum(1 for p in base_model.parameters() if p.device.type == 'cuda')
    # 打印模型所有参数的设备分布
    logger.info(f"参数分布: CPU = {on_cpu}, GPU = {on_cuda}")
    # 打印模型所有参数的设备分布详情
    for i, (name, param) in enumerate(base_model.named_parameters()):
        logger.info(f"i={i}, 参数: {name}, 设备: {param.device}, 大小: {param.numel() * param.element_size() / 1024**2:.2f} MB");

    # 哪些参数是冻结的基础模型参数，哪些是 LoRA adapter 参数，并打印它们的设备分布详情，确认所有参数都在 GPU 上
    # total_params = 0
    # adapter_params = 0
    # for name, param in base_model.named_parameters():
    #     total_params += param.numel()
    #     if "lora" in name.lower():
    #         adapter_params += param.numel()
    #         logger.info(f"LoRA 参数: {name}, 设备: {param.device}, 大小: {param.numel() * param.element_size() / 1024**2:.2f} MB")
    #     else:
    #         logger.info(f"基础模型参数: {name}, 设备: {param.device}, 大小: {param.numel() * param.element_size() / 1024**2:.2f} MB")
    # logger.info(f"总参数量: {total_params:,}, LoRA 参数量: {adapter_params:,} ({adapter_params / total_params * 100:.2f}%)")    

 
    #print(f"加载Base 模型完成，当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    #logger.info(f">>> 加载Base 模型完成 ，当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    # 打印此时剩余的显存（单位：GB）
    # allocated = torch.cuda.memory_allocated() / 1024**3
    # reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f">>> 加载Base 模型完成 ，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    #print(f"释放后显存占用: 已分配 {allocated:.2f} GB, 预分配 {reserved:.2f} GB")
    #time.sleep(1)  # 确保显存占用稳定
    # 
    show_samples= [];
    _n_show = max(config.EVAL_LLM_JUDGE_MAX_SAMPLES, 5) if config.EVAL_LLM_JUDGE_ENABLED else 5
    # 随机选择 samples 中的 n_show 条进行展示，保证每次评测报告的样本多样性；如果样本数量不足 n_show，则展示全部样本。
    if len(eval_samples) > _n_show:
        show_samples = random.sample(eval_samples, _n_show);
        #show_samples = random.sample(eval_samples, _n_show)
    else:        show_samples = eval_samples[:_n_show]
    #logger.info(f"-----------------------------------------------------------------------------------------------")
    # 打印选中的展示样本的 instruction 字段，确认它们是随机且多样的
    logger.info(f"------------------------------展示样本（共 {len(show_samples)} 条）--------------------------")
    for i, s in enumerate(show_samples):
        # instr = s.get("instruction", "N/A")
        instr = _get_question(s);
        logger.info(f"样本 {i + 1}: {instr[:50]}{'...' if len(instr) > 50 else ''}")
    logger.info(f"-------------------------------------------------------------------------------------------")
    # ---- Base 评估 ----
    ppl_results = []
    if config.EVAL_PPL_ENABLED:
        logger.info(">>> Perplexity 评估")
        ppl_results.append(
            evaluate_perplexity(base_model, tokenizer, config, eval_samples, "Base")
        )
        logger.info(f">>> Perplexity 评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> Perplexity 评估: 已禁用 (eval.ppl.enabled=false)")
        
    rouge_results = []
    if config.EVAL_ROUGE_ENABLED:
        logger.info(">>> ROUGE-L 评估")
        rouge_results.append(
            evaluate_rouge(base_model, tokenizer, config, eval_samples, "Base")
        )
        logger.info(f">>> ROUGE-L 评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> ROUGE-L 评估: 已禁用 (eval.rouge.enabled=false)")
    # logger.info(f" 当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    bertscore_results = []
    if config.EVAL_BERTSCORE_ENABLED:
        logger.info(">>> BERTScore 评估")
        bertscore_results.append(
            evaluate_bertscore(base_model, tokenizer, config, eval_samples, "Base")
        )
        logger.info(f">>> BERTScore 评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> BERTScore 评估: 已禁用 (eval.bertscore.enabled=false)")
    # logger.info(f" 当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    base_gen_samples = []
    _need_gen = config.EVAL_GEN_SAMPLES_ENABLED or config.EVAL_LLM_JUDGE_ENABLED
    if _need_gen:
        #_n_show = max(config.EVAL_LLM_JUDGE_MAX_SAMPLES, 5) if config.EVAL_LLM_JUDGE_ENABLED else 5
        logger.info(">>> 收集 Base 模型生成样本 (n=%d)%s",
                    _n_show, " (供 LLM-as-Judge)" if not config.EVAL_GEN_SAMPLES_ENABLED else "")
        base_gen_samples = collect_generation_samples(
            base_model, tokenizer, config, show_samples, "Base", n_show=_n_show
        )
        logger.info(f">>> 收集 Base 模型生成样本 ，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> 生成样本收集: 已禁用 (eval.gen_samples.enabled=false)") 
    general_results = []
    if config.EVAL_GENERAL_ABILITY_ENABLED:
        logger.info(">>> 通用能力评估 (Base)")
        general_results.append(
            evaluate_general_ability(base_model, tokenizer, config, "Base")
        )
        logger.info(f">>>  通用能力评估 (Base) 完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> 通用能力评估: 已禁用 (eval.general_ability.enabled=false)")

    # ---- 释放 Base 模型显存 ----
    logger.info(f"当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    del base_model
    gc.collect()  # 手动触发 Python 垃圾回收
    torch.cuda.synchronize()  # 确保所有 GPU 操作完成
    torch.cuda.empty_cache()
    logger.info(f"释放 Base 模型显存后，当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    # ---- 加载 LoRA 模型 ----
    adapter_dir = _detect_adapter_dir(config)
    logger.info("--- 加载 LoRA 模型 ---")
    logger.info("  基座模型: %s", config.MODEL_ID)
    logger.info("  Adapter : %s", adapter_dir)
    lora_base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        # device_map="auto",
        device_map="cuda:0",  # 不要用 auto！
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # ← 这行最关键
    )
    logger.info(f"加载 LoRA 基座模型完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    lora_model = PeftModel.from_pretrained(lora_base, adapter_dir)
    logger.info(f"加载 LoRA 模型完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    # # 强制把所有参数移到 GPU（有些模型可能默认在 CPU 上，导致评估时频繁数据迁移）
    lora_model = lora_model.to('cuda')
    # 哪些参数是冻结的基础模型参数，哪些是 LoRA adapter 参数，并打印它们的设备分布详情，确认所有参数都在 GPU 上
    total_params = 0
    adapter_params = 0
    for name, param in lora_model.named_parameters():
        total_params += param.numel()
        if "lora" in name.lower():
            adapter_params += param.numel()
            logger.info(f"LoRA 参数: {name}, 设备: {param.device}, 大小: {param.numel() * param.element_size() / 1024**2:.2f} MB")
        else:
            logger.info(f"基础模型参数: {name}, 设备: {param.device}, 大小: {param.numel() * param.element_size() / 1024**2:.2f} MB")
    logger.info(f"总参数量: {total_params:,}, LoRA 参数量: {adapter_params:,} ({adapter_params / total_params * 100:.2f}%)")
    logger.info(f"当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    # ---- LoRA 评估 ----
    if config.EVAL_PPL_ENABLED:
        logger.info(">>> Perplexity 评估")
        ppl_results.append(
            evaluate_perplexity(lora_model, tokenizer, config, eval_samples, "LoRA")
        )
        logger.info(f">>> Perplexity 评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> Perplexity 评估: 已禁用 (eval.ppl.enabled=false)")
    
    if config.EVAL_ROUGE_ENABLED:
        logger.info(">>> ROUGE-L 评估")
        rouge_results.append(
            evaluate_rouge(lora_model, tokenizer, config, eval_samples, "LoRA")
        )
        logger.info(f">>> ROUGE-L 评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> ROUGE-L 评估: 已禁用 (eval.rouge.enabled=false)")

    if config.EVAL_BERTSCORE_ENABLED:
        logger.info(">>> BERTScore 评估")
        bertscore_results.append(
            evaluate_bertscore(lora_model, tokenizer, config, eval_samples, "LoRA")
        )
        logger.info(f">>> BERTScore 评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> BERTScore 评估: 已禁用 (eval.bertscore.enabled=false)")

    lora_gen_samples = []
    if _need_gen:
        logger.info(">>> 收集 LoRA 模型生成样本 (n=%d)%s",
                    _n_show, " (供 LLM-as-Judge)" if not config.EVAL_GEN_SAMPLES_ENABLED else "")
        lora_gen_samples = collect_generation_samples(
            lora_model, tokenizer, config, show_samples, "LoRA", n_show=_n_show
        )
        logger.info(f">>>  LoRA 模型生成样本收集 完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> 生成样本收集: 已禁用 (eval.gen_samples.enabled=false)")

    if config.EVAL_GENERAL_ABILITY_ENABLED:
        logger.info(">>> 通用能力评估 (LoRA)")
        general_results.append(
            evaluate_general_ability(lora_model, tokenizer, config, "LoRA")
        )
        logger.info(f">>> 通用能力评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.info(">>> 通用能力评估: 已禁用 (eval.general_ability.enabled=false)")
    del lora_base  # 只保留 LoRA 模型，释放基座模型占用的显存
    gc.collect()  # 手动触发 Python 垃圾回收
    torch.cuda.synchronize()  # 确保所有 GPU 操作完成
    torch.cuda.empty_cache()
    logger.info(f"释放 LORA  模型显存后，当前 GPU 显存已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    # ---- LLM-as-Judge 评估 ----
    llm_judge_results = evaluate_with_llm_judge(config, base_gen_samples, lora_gen_samples)
    logger.info(f">>> LLM-as-Judge 评估完成，当前 GPU 显存 已分配 {torch.cuda.memory_allocated() / 1024**3:.2f} GB, 预分配 {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # ---- 生成报告 ----
    report = generate_report(config, ppl_results, rouge_results, base_gen_samples, lora_gen_samples, llm_judge_results, bertscore_results, general_results)

    report_path = Path(config.EVAL_REPORT_PATH)
    report_path.write_text(report, encoding="utf-8")
    logger.info("评测报告已保存: %s", report_path.resolve())

    # ---- 保存 JSON 结构化结果 ----
    json_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": config.MODEL_ID,
        "lora_output": config.OUTPUT_DIR,
        "eval_num_samples": n_eval,
        "perplexity": ppl_results,
        "rouge_l": rouge_results,
        "generation_samples": [
            {"instruction": s["instruction"], "reference": s["reference"][:200],
             "base": s["generated"][:200], "lora": l["generated"][:200]}
            for s, l in zip(base_gen_samples, lora_gen_samples)
        ],
        "llm_judge": llm_judge_results,
        "bertscore": bertscore_results,
        "general_ability": general_results,
    }
    json_path = Path(config.EVAL_JSON_PATH)
    json_path.write_text(
        json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结构化结果已保存: %s", json_path.resolve())

    # ---- 终端摘要 ----
    logger.info("=" * 60)
    logger.info("  评估摘要")
    logger.info("=" * 60)
    if ppl_results:
        for r in ppl_results:
            logger.info("  PPL  [%s]: %.2f  (loss=%.4f)", r["label"], r["avg_ppl"], r["avg_loss"])
    else:
        logger.info("  PPL: 已禁用")
    if rouge_results:
        for r in rouge_results:
            logger.info("  ROUGE [%s]: %.4f", r["label"], r["rouge_l_f1"])
    else:
        logger.info("  ROUGE: 已禁用")
    if bertscore_results:
        for r in bertscore_results:
            if r.get("enabled") is False:
                logger.info("  BERTScore [%s]: 未安装 (pip install bert-score)", r["label"])
            else:
                logger.info("  BERTScore [%s]: F1=%.4f  P=%.4f  R=%.4f",
                            r["label"], r["bertscore_f1"], r["bertscore_precision"], r["bertscore_recall"])
    else:
        logger.info("  BERTScore: 已禁用")
    if general_results:
        for r in general_results:
            logger.info("  General [%s]: avg_len=%.0f chars, trunc_rate=%.1f%%",
                        r["label"], r["avg_length"], r["truncation_rate"] * 100)
    else:
        logger.info("  General: 已禁用")
    if llm_judge_results and llm_judge_results.get("enabled") is not False:
        logger.info("  LLM-Judge [LoRA]: %s", json.dumps(llm_judge_results.get("lora_scores", {}), ensure_ascii=False))
        logger.info("  LLM-Judge [Base]: %s", json.dumps(llm_judge_results.get("base_scores", {}), ensure_ascii=False))
    else:
        logger.info("  LLM-Judge: 已禁用")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    # 确保项目根目录在搜索路径中（python poly_distill/eval.py 运行时需要）
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from poly_distill.config import load_config, setup_environment

    cfg = load_config()
    setup_environment(cfg)

    # 仅推理评估，不重新训练——需要先加载 tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_ID, use_fast=True, trust_remote_code=True, cache_dir=cfg.CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_evaluation(cfg, tokenizer)
