"""
PolyDistill — 多教师知识蒸馏训练框架。

统一调度 GPT、Claude、Gemini 等商业 API 教师模型，
将集体知识蒸馏到本地学生模型。
"""

from poly_distill.config import Config, get_logger, load_config, setup_environment, setup_logging

__all__ = [
    "Config",
    "get_logger",
    "load_config",
    "setup_environment",
    "setup_logging",
]
