"""
PolyDistill — 多教师知识蒸馏训练框架。

统一调度 GPT-4o、Claude 3.5 Sonnet、Gemini 1.5 Pro 等商业 API 教师模型，
将集体知识蒸馏到本地学生模型。
"""

from poly_distill.config import Config, get_logger, load_config, setup_environment, setup_logging
from poly_distill.dataset import load_and_prepare_data
from poly_distill.trainer import train
from poly_distill.eval import run_evaluation

__all__ = [
    "Config",
    "get_logger",
    "load_config",
    "setup_environment",
    "setup_logging",
    "load_and_prepare_data",
    "train",
    "run_evaluation",
]
