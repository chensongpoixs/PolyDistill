"""
全局配置与运行环境初始化。

配置加载优先级：config.yaml 覆盖值 > Config class 默认值。
未安装 PyYAML 或 YAML 文件不存在时，静默回退到 Config 默认值。
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# ============================================================
# 1. 全局配置（默认值）
# ============================================================
class Config:
    """训练与推理的集中配置。

    所有可调参数在此统一声明默认值。
    运行时会尝试从 config.yaml 加载覆盖值。
    修改超参数推荐直接编辑 config.yaml，无需改动此 class。
    """

    # ---- 模型 ----
    MODEL_ID: str = "Qwen/Qwen3-0.6B"

    # ---- 数据 ----
    DATA_DIR: str = "./data"  # 数据目录，自动读取目录下所有 .json 文件并合并

    # ---- 路径 ----
    CACHE_DIR: str = "./models/qwen3-0.6b"
    OUTPUT_DIR: str = "./lora_sft_ai_infra_audio_video_output"

    # ---- LoRA 参数 ----
    LORA_R: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.1
    LORA_TARGET_MODULES: list = None  # 由 __init__ 设置默认值

    # ---- 训练超参数 ----
    PER_DEVICE_BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 8
    LEARNING_RATE: float = 2e-4
    WARMUP_RATIO: float = 0.03
    LR_SCHEDULER_TYPE: str = "cosine"
    NUM_TRAIN_EPOCHS: int = 100
    # 正则化
    WEIGHT_DECAY: float = 0.01
    MAX_GRAD_NORM: float = 1.0
    NEFTUNE_NOISE_ALPHA: int = 5
    # 早停
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_THRESHOLD: float = 0.001
    # 评估分割
    EVAL_SPLIT_RATIO: float = 0.1
    # 梯度检查点（显存不足时开启，以 ~20% 速度换 ~50% 显存节省）
    GRADIENT_CHECKPOINTING: bool = False
    # 日志与保存
    LOGGING_STEPS: int = 10
    SAVE_STRATEGY: str = "best"
    SAVE_TOTAL_LIMIT: int = 3
    METRIC_FOR_BEST_MODEL: str = "eval_loss"
    LOAD_BEST_MODEL_AT_END: bool = True

    # ---- 对话模板 ----
    RESPONSE_TEMPLATE: str = "<|im_start|>assistant\n"
    SYSTEM_PROMPT: str = "You are a helpful assistant."

    # ---- 设备 ----
    CUDA_VISIBLE_DEVICES: str = "0"

    # ---- 镜像加速（国内环境） ----
    HF_ENDPOINT: str = "https://hf-mirror.com"

    # ---- 应用日志 ----
    APP_LOG_LEVEL: str = "INFO"
    APP_LOG_FILE: str = "./train.log"
    APP_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # ---- 评估参数 ----
    EVAL_NUM_SAMPLES: int = 50
    EVAL_MAX_NEW_TOKENS: int = 512
    EVAL_TEMPERATURE: float = 0.1
    EVAL_REPORT_PATH: str = "./eval_report.md"
    EVAL_JSON_PATH: str = "./eval_results.json"

    def __init__(self, **kwargs):
        """用关键字参数覆盖任意默认值。

        用法:
            cfg = Config(LEARNING_RATE=1e-4, NUM_TRAIN_EPOCHS=100)
        """
        # 设置 LORA_TARGET_MODULES 的默认值（list 不能直接放在 class 属性中）
        if self.LORA_TARGET_MODULES is None:
            self.LORA_TARGET_MODULES = ["q_proj", "v_proj"]

        # 应用关键字参数覆盖
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ============================================================
# 2. YAML 配置加载
# ============================================================
# 字段映射：config.yaml 中的小写下划线名 → Config class 中的大写下划线名
_FIELD_MAP = {
    # ---- 模型 ----
    "model_id": "MODEL_ID",
    # ---- 数据 ----
    "data_dir": "DATA_DIR",
    # ---- 路径 ----
    "cache_dir": "CACHE_DIR",
    "output_dir": "OUTPUT_DIR",
    # ---- LoRA ----
    "lora.r": ("LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGET_MODULES"),
    # ---- 训练 ----
    "training.per_device_batch_size": "PER_DEVICE_BATCH_SIZE",
    "training.gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
    "training.learning_rate": "LEARNING_RATE",
    "training.warmup_ratio": "WARMUP_RATIO",
    "training.lr_scheduler_type": "LR_SCHEDULER_TYPE",
    "training.num_train_epochs": "NUM_TRAIN_EPOCHS",
    "training.weight_decay": "WEIGHT_DECAY",
    "training.max_grad_norm": "MAX_GRAD_NORM",
    "training.neftune_noise_alpha": "NEFTUNE_NOISE_ALPHA",
    "training.early_stopping_patience": "EARLY_STOPPING_PATIENCE",
    "training.early_stopping_threshold": "EARLY_STOPPING_THRESHOLD",
    "training.eval_split_ratio": "EVAL_SPLIT_RATIO",
    "training.gradient_checkpointing": "GRADIENT_CHECKPOINTING",
    "training.logging_steps": "LOGGING_STEPS",
    "training.save_strategy": "SAVE_STRATEGY",
    "training.save_total_limit": "SAVE_TOTAL_LIMIT",
    "training.metric_for_best_model": "METRIC_FOR_BEST_MODEL",
    "training.load_best_model_at_end": "LOAD_BEST_MODEL_AT_END",
    # ---- 对话模板 ----
    "response_template": "RESPONSE_TEMPLATE",
    "system_prompt": "SYSTEM_PROMPT",
    # ---- 设备 ----
    "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
    # ---- 镜像 ----
    "hf_endpoint": "HF_ENDPOINT",
    # ---- 日志 ----
    "log.level": "APP_LOG_LEVEL",
    "log.file": "APP_LOG_FILE",
    "log.format": "APP_LOG_FORMAT",
    # ---- 评估 ----
    "eval.num_samples": "EVAL_NUM_SAMPLES",
    "eval.max_new_tokens": "EVAL_MAX_NEW_TOKENS",
    "eval.temperature": "EVAL_TEMPERATURE",
    "eval.report_path": "EVAL_REPORT_PATH",
    "eval.json_path": "EVAL_JSON_PATH",
}


def _nested_get(d: dict, dotted_key: str, default=None):
    """从嵌套字典中用点分隔键取值。例如 d['lora']['r']。"""
    keys = dotted_key.split(".")
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def load_config(yaml_path: Optional[str] = None) -> Config:
    """加载配置：YAML 覆盖值 > Config 默认值。

    优先级：
      1. 命令行指定的 yaml_path（最高）
      2. 项目根目录下的 config.yaml
      3. Config class 默认值（兜底）

    Args:
        yaml_path: YAML 配置文件路径。为 None 时自动查找 ./config.yaml。

    Returns:
        Config: 合并后的配置对象。

    若无 PyYAML 依赖或 YAML 文件不存在，静默回退到 Config 默认值。
    """
    logger = logging.getLogger(__name__)

    # 尝试导入 yaml
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML 未安装，使用 Config 默认值。安装: pip install pyyaml")
        return Config()

    # 确定 YAML 文件路径
    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "config.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        logger.warning("未找到 %s，使用 Config 默认值", yaml_path)
        return Config()

    # 读取 YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if yaml_data is None:
        return Config()

    logger.info("已加载配置: %s", yaml_path)

    # 将 YAML 数据映射为 Config 关键字参数
    overrides = {}

    # ---- 扁平字段 ----
    for yaml_key, config_key in _FIELD_MAP.items():
        # 跳过 lora 嵌套块的批量映射
        if yaml_key == "lora.r":
            continue
        value = _nested_get(yaml_data, yaml_key)
        if value is not None:
            overrides[config_key] = value

    # ---- LoRA 嵌套块 ----
    lora_block = yaml_data.get("lora")
    if isinstance(lora_block, dict):
        if "r" in lora_block:
            overrides["LORA_R"] = lora_block["r"]
        if "alpha" in lora_block:
            overrides["LORA_ALPHA"] = lora_block["alpha"]
        if "dropout" in lora_block:
            overrides["LORA_DROPOUT"] = lora_block["dropout"]
        if "target_modules" in lora_block:
            overrides["LORA_TARGET_MODULES"] = lora_block["target_modules"]

    return Config(**overrides)


# ============================================================
# 3. 环境初始化
# ============================================================
def setup_environment(config: Config) -> None:
    """配置运行环境：镜像源、CUDA 设备、禁用分布式。

    为什么禁用分布式（DDP）？
      - 本脚本面向单 GPU 微调场景（0.6B 小模型 + LoRA），无需多卡。
      - 显式清理 DDP 环境变量可防止误触发 torch.distributed 初始化，
        避免出现 "Address already in use" 或端口冲突等报错。
    """
    # Hugging Face 国内镜像加速
    os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT

    # 固定 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

    # 彻底禁用分布式训练
    _reset_ddp_env()

    # 禁止 tokenizer 多进程并行（避免与 DataLoader 多进程冲突）
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 初始化日志系统（环境就绪后、业务逻辑之前）
    setup_logging(config)


def _reset_ddp_env() -> None:
    """清理可能残留的 PyTorch 分布式环境变量。

    PyTorch 分布式训练依赖以下环境变量建立进程间通信。
    在单卡场景下，这些变量若被提前设置（如从其他脚本继承），
    会导致 SFTTrainer / accelerate 误认为处于分布式模式，进而报错。
    此处逐一删除以恢复到纯净的单机单卡状态。
    """
    ddp_keys = [
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "NODE_RANK",
    ]
    for key in ddp_keys:
        os.environ.pop(key, None)


# ============================================================
# 4. 日志系统
# ============================================================
def setup_logging(config: Config) -> None:
    """初始化全局日志系统。

    配置 root logger，同时输出到：
      - 控制台（stdout）
      - 文件（config.APP_LOG_FILE）

    所有模块通过 logging.getLogger(__name__) 获取 logger，
    日志自动继承此配置。

    调用时机：在 setup_environment() 之后、业务逻辑之前。
    """
    level = getattr(logging, config.APP_LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # 清除已有的 handler（避免重复添加）
    root.handlers.clear()

    # 格式
    fmt = logging.Formatter(config.APP_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 文件 handler
    file_handler = logging.FileHandler(config.APP_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 抑制第三方库的 DEBUG 日志
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("trl").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root.info(f"日志系统已初始化 (level={config.APP_LOG_LEVEL}, file={config.APP_LOG_FILE})")


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger 的便捷函数。等价于 logging.getLogger(name)。"""
    return logging.getLogger(name)
