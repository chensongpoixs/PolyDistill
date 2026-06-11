"""
全局配置与运行环境初始化。

配置加载优先级：config.yaml 覆盖值 > Config class 默认值。
未安装 PyYAML 或 YAML 文件不存在时，静默回退到 Config 默认值。
"""

import logging
import os
import sys
import warnings
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
    # 注意力实现: "auto" | "flash_attention_2" | "sdpa" | "eager"
    #   auto                — 自动: FA3(Blackwell) > FA2(Ampere) > SDPA(回退)
    #   flash_attention_2   — Flash Attention，HF 内部根据 GPU 自动路由 FA3/FA2
    #   sdpa                — PyTorch 2.0+ 内置 scaled_dot_product_attention
    #   eager               — 原生实现，兼容性最好但显存最大
    # 注意: HF 的 attn_implementation 参数不支持 "flash_attention_3"，
    #       传 "flash_attention_2" 即可在 Blackwell 上自动使用 FA3 kernel。
    ATTN_IMPLEMENTATION: str = "auto"

    # ---- 数据 ----
    DATA_DIR: str = "./data"  # 数据目录，自动读取目录下所有 .json 文件并合并

    # ---- 路径 ----
    CACHE_DIR: str = "./models/qwen3-0.6b"
    OUTPUT_DIR: str = "./lora_sft_ai_infra_audio_video_output"
    # YOLOv5 风格实验目录：每次训练自动创建 runs/train/exp{N}/ 并将所有产物归入
    RUNS_DIR: str = "./runs/train"

    # ---- 模型导出 ----
    EXPORT_DIR: str = ""  # 默认留空，代码按 MODEL_ID 自动推导（如 TinySage-4B）

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
    # DataLoader 优化（缓解 GPU 数据饥饿）
    DATALOADER_NUM_WORKERS: int = 0  # 0=单进程；建议 8-16 以加速数据预处理
    DATALOADER_PIN_MEMORY: bool = True  # 加速 CPU→GPU 数据传输
    DATALOADER_PREFETCH_FACTOR: int = 4  # 每个 worker 预取 batch 数，缓解 GPU 数据饥饿
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
    APP_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s"

    # ---- 评估参数 ----
    EVAL_NUM_SAMPLES: int = 50
    EVAL_MAX_NEW_TOKENS: int = 10240
    EVAL_TEMPERATURE: float = 0.1
    EVAL_REPORT_PATH: str = "./eval_report.md"
    EVAL_JSON_PATH: str = "./eval_results.json"
    # 评估维度开关（默认全部开启，可按需关闭）
    EVAL_PPL_ENABLED: bool = True
    EVAL_ROUGE_ENABLED: bool = False  # 默认关闭，字面匹配对 LLM 改写不友好，用 BERTScore 替代
    EVAL_BERTSCORE_ENABLED: bool = True
    EVAL_GEN_SAMPLES_ENABLED: bool = True
    EVAL_GENERAL_ABILITY_ENABLED: bool = True
    EVAL_SHOW_SAMPLES: bool = True  # 是否在评测报告中展示生成样本（仅 eval_gen_samples_enabled=True 时生效）
    # ---- LLM-as-Judge（大模型打分评估） ----
    EVAL_LLM_JUDGE_ENABLED: bool = True
    EVAL_LLM_JUDGE_ENDPOINT: str = "http://localhost:8000/v1/chat/completions"
    EVAL_LLM_JUDGE_MODEL: str = "gpt-4"
    EVAL_LLM_JUDGE_API_KEY: str = ""
    EVAL_LLM_JUDGE_MAX_SAMPLES: int = 10
    EVAL_LLM_JUDGE_TIMEOUT: int = 600  # 请求超时秒数（默认 10 分钟）
    EVAL_LLM_JUDGE_TEMPERATURE: float = 0.0  # 评估用低温，力求确定性
    EVAL_LLM_JUDGE_MAX_TOKENS: int = 4096  # 最大生成 token 数
    EVAL_LLM_JUDGE_TOP_P: float = 1.0  # nucleus sampling
    EVAL_LLM_JUDGE_SEED: int = 42  # 随机种子，评估结果可复现
    EVAL_LLM_JUDGE_MAX_RETRIES: int = 2  # 失败自动重试次数
    

    # ---- 数据质量过滤 ----
    QUALITY_FILTER_ENABLED: bool = True
    QUALITY_FILTER_MIN_RESPONSE_LENGTH: int = 50
    QUALITY_FILTER_MAX_RESPONSE_LENGTH: int = 4096
    QUALITY_FILTER_SKIP_EMPTY: bool = True
    QUALITY_FILTER_SKIP_DUPLICATES: bool = True

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
    "attn_implementation": "ATTN_IMPLEMENTATION",
    # ---- 数据 ----
    "data_dir": "DATA_DIR",
    # ---- 路径 ----
    "cache_dir": "CACHE_DIR",
    "runs_dir": "RUNS_DIR",
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
    "training.dataloader_num_workers": "DATALOADER_NUM_WORKERS",
    "training.dataloader_pin_memory": "DATALOADER_PIN_MEMORY",
    "training.dataloader_prefetch_factor": "DATALOADER_PREFETCH_FACTOR",
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
    "eval.ppl.enabled": "EVAL_PPL_ENABLED",
    "eval.rouge.enabled": "EVAL_ROUGE_ENABLED",
    "eval.bertscore.enabled": "EVAL_BERTSCORE_ENABLED",
    "eval.gen_samples.enabled": "EVAL_GEN_SAMPLES_ENABLED",
    "eval.general_ability.enabled": "EVAL_GENERAL_ABILITY_ENABLED",
    "eval.show_samples": "EVAL_SHOW_SAMPLES", # EVAL_SHOW_SAMPLES 仅控制评测报告中是否展示生成样本，不影响 eval_gen_samples_enabled 的评测流程和结果
    # ---- LLM-as-Judge ----
    "eval.llm_judge.enabled": "EVAL_LLM_JUDGE_ENABLED",
    "eval.llm_judge.endpoint": "EVAL_LLM_JUDGE_ENDPOINT",
    "eval.llm_judge.model": "EVAL_LLM_JUDGE_MODEL",
    "eval.llm_judge.api_key": "EVAL_LLM_JUDGE_API_KEY",
    "eval.llm_judge.max_samples": "EVAL_LLM_JUDGE_MAX_SAMPLES",
    "eval.llm_judge.timeout": "EVAL_LLM_JUDGE_TIMEOUT",
    "eval.llm_judge.temperature": "EVAL_LLM_JUDGE_TEMPERATURE",
    "eval.llm_judge.max_tokens": "EVAL_LLM_JUDGE_MAX_TOKENS",
    "eval.llm_judge.top_p": "EVAL_LLM_JUDGE_TOP_P",
    "eval.llm_judge.seed": "EVAL_LLM_JUDGE_SEED",
    "eval.llm_judge.max_retries": "EVAL_LLM_JUDGE_MAX_RETRIES",
    # ---- 数据质量过滤 ----
    "distillation.quality_filter.enabled": "QUALITY_FILTER_ENABLED",
    "distillation.quality_filter.min_response_length": "QUALITY_FILTER_MIN_RESPONSE_LENGTH",
    "distillation.quality_filter.max_response_length": "QUALITY_FILTER_MAX_RESPONSE_LENGTH",
    "distillation.quality_filter.skip_empty": "QUALITY_FILTER_SKIP_EMPTY",
    "distillation.quality_filter.skip_duplicates": "QUALITY_FILTER_SKIP_DUPLICATES",
    # ---- 模型导出 ----
    "export.output_dir": "EXPORT_DIR",
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

    # Windows 上 flash-attn.ops.triton 会 import triton，但 triton 是 Linux 专属包。
    # 在 import 链触发前注入 mock triton 模块，防止 ModuleNotFoundError 中断整个流程。
    # flash-attn 的核心 CUDA ops 不依赖 triton，注入 mock 不影响 FA2/FA3 正常使用。
    #
    # 双重注入策略（解决 Windows multiprocessing spawn 子进程问题）：
    #   1. 运行时注入: sys.meta_path + sys.modules → 保护当前进程
    #   2. .pth 文件注入: site-packages 下安装 _triton_win32_mock.pth
    #      → 保护 DataLoader spawn 子进程（子进程不执行 setup_environment，
    #         但 site.py 会处理 .pth 文件，早于任何 import）
    if sys.platform == "win32":
        _inject_triton_mock_if_needed()
        _install_triton_mock_pth()

    # 静默 TensorFlow 日志（非 TF 项目，但依赖可能间接导入 TF）。
    # TF_CPP_MIN_LOG_LEVEL: 0=ALL 1=INFO 2=WARNING 3=ERROR
    # 用 os.environ["..."]= 直接覆盖（非 setdefault），确保生效。
    # TF 用内部 absl/tf_logging，Python warnings.filterwarnings 无法拦截；
    # 设置 "3" 只显示 ERROR，完全静默 WARNING/INFO。
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    # 抑制 TF/Keras 通过 Python warnings 模块发出的 deprecation
    warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")
    warnings.filterwarnings("ignore", message=".*oneDNN custom operations.*")

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


def _inject_triton_mock_if_needed() -> None:
    """Windows 上 triton 不可用时，通过 sys.meta_path 拦截所有 triton.* 导入。

    flash-attn / transformers / torch._dynamo / torch._inductor 等多层库
    在不同路径以 import 语句或属性访问方式使用 triton。逐一在 sys.modules
    预注册所有子模块不可行（torch._inductor 深层依赖 triton.backends.compiler
    等未知子包）。

    解决方案：在 sys.meta_path 注册一个自定义查找器，拦截所有以 "triton" /
    "triton." 开头的导入请求，自动创建 mock 模块注入 sys.modules。
    flash-attn 的 CUDA kernel (FA2/FA3) 不依赖 triton 运行时，mock 不影响训练。
    """
    try:
        import triton  # noqa: F401
        return  # triton 已安装，无需 mock
    except ImportError:
        pass

    import types
    from importlib.abc import Loader, MetaPathFinder
    from importlib.machinery import ModuleSpec

    class _TritonFinder(MetaPathFinder):
        """自动创建 triton.* 的 mock 模块。"""

        def find_spec(self, fullname, path, target=None):
            # 仅拦截 triton / triton.xxx / triton.xxx.yyy ...
            if fullname != "triton" and not fullname.startswith("triton."):
                return None  # 不处理，交给下一个查找器

            # 如果已经被注册了（例如 triton / triton.language），复用
            if fullname in sys.modules:
                mod = sys.modules[fullname]
                return mod.__spec__ if hasattr(mod, "__spec__") else None

            return ModuleSpec(fullname, _TritonLoader(), origin="mock")

    class _TritonLoader(Loader):
        """创建 mock 模块，处理属性访问和包结构。"""

        def create_module(self, spec):
            mod = _AnyAttrModule(spec.name)
            mod.__spec__ = spec
            mod.__path__ = []       # 标记为包（允许 triton.xxx.zzz 子导入）
            mod.__file__ = "mock"
            mod.__loader__ = self
            mod.__package__ = spec.name
            return mod

        def exec_module(self, module):
            # 特殊处理：triton 根模块
            if module.__name__ == "triton":
                module.__version__ = "0.0.0.win32.mock"
                module.jit = _make_triton_jit()
                module.autotune = _make_triton_decorator()
                module.heuristics = _make_triton_decorator()
                # 确保 triton.backends 可访问
                if "triton.language" not in sys.modules:
                    self._ensure("triton.language")

            # 特殊处理：triton.backends（torch._inductor 依赖回退路径）
            if module.__name__ == "triton.backends":
                if "triton.backends.compiler" not in sys.modules:
                    self._ensure("triton.backends.compiler")

        def _ensure(self, name):
            """确保某个子模块已注册到 sys.modules。"""
            if name not in sys.modules:
                spec = ModuleSpec(name, _TritonLoader(), origin="mock")
                mod = _AnyAttrModule(name)
                mod.__spec__ = spec
                mod.__path__ = []
                mod.__file__ = "mock"
                mod.__loader__ = self
                mod.__package__ = name
                sys.modules[name] = mod

    class _AnyAttrModule(types.ModuleType):
        """对任意属性访问返回合法的 mock 值，不抛 AttributeError。"""

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            child = _AnyAttrModule(f"{self.__name__}.{name}")
            child.__spec__ = ModuleSpec(child.__name__, loader=None, origin="mock")
            child.__path__ = []
            child.__file__ = "mock"
            child.__loader__ = None
            child.__package__ = child.__name__
            setattr(self, name, child)
            return child

    def _make_triton_jit():
        def jit(*args, **kwargs):
            def decorator(fn):
                return fn
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator
        return jit

    def _make_triton_decorator():
        return lambda *args, **kwargs: lambda fn: fn

    # ---- 注册 meta_path 查找器（最高优先级） ----
    sys.meta_path.insert(0, _TritonFinder())

    # ---- 注册 triton 根（meta_path 不触发已有的 sys.modules 条目） ----
    spec = ModuleSpec("triton", _TritonLoader(), origin="mock")
    mock = _AnyAttrModule("triton")
    mock.__spec__ = spec
    mock.__path__ = []
    mock.__file__ = "mock"
    mock.__loader__ = _TritonLoader()
    mock.__package__ = "triton"
    mock.__version__ = "0.0.0.win32.mock"
    mock.jit = _make_triton_jit()
    mock.autotune = _make_triton_decorator()
    mock.heuristics = _make_triton_decorator()
    sys.modules["triton"] = mock

    import logging
    logging.getLogger(__name__).info(
        "Windows 环境: 已注入 triton mock 导入钩子 (flash-attn CUDA ops 不受影响)"
    )


def _install_triton_mock_pth() -> None:
    """在 site-packages 下安装 .pth 文件，使 triton mock 在 spawn 子进程中也生效。

    Windows multiprocessing 使用 spawn 模式，DataLoader worker 是全新 Python 进程，
    不继承父进程的 sys.meta_path / sys.modules 注入。但所有 Python 进程都会在启动时
    由 site.py 处理 site-packages 下的 .pth 文件，因此在此安装一个 .pth 文件可确保
    子进程在 import 任何模块之前已完成 triton mock 注入。

    .pth 文件格式：一行 "import <module>" 会被 site.py 执行。
    同时写入 _triton_win32_mock.py 作为独立的 mock 模块。
    """
    try:
        import site
        site_dirs = site.getsitepackages()
        if not site_dirs:
            return
        target_dir = site_dirs[0]
    except Exception:
        return

    mock_module_path = os.path.join(target_dir, "_triton_win32_mock.py")

    # 写入自包含的 triton mock 模块。
    # 此模块无外部依赖, 不引用 poly_distill 的任何代码,
    # 确保在任意 Python 进程中都能独立运行。
    _TRITON_MOCK_CODE = r'''
# Auto-generated by PolyDistill. Do not edit manually.
# Purpose: mock triton on Windows so flash-attn + transformers import chains don't break
# in multiprocessing spawn child processes (DataLoader workers).
#
# This file is self-contained (zero external dependencies) so it works in ANY Python
# process, including spawned subprocesses that don't import poly_distill at all.
#
# The mock is a no-op on Linux or if real triton is installed.
import sys
import os

if sys.platform != "win32":
    pass  # not needed on Linux
else:
    try:
        import triton  # noqa: F401
        # real triton is installed, no mock needed
    except ImportError:
        import types
        from importlib.abc import Loader, MetaPathFinder
        from importlib.machinery import ModuleSpec

        # Prevent double injection
        if any(getattr(f, "__name__", "") == "_TritonWin32Finder"
               for f in sys.meta_path):
            pass  # already injected
        else:
            class _TritonWin32Finder(MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname != "triton" and not fullname.startswith("triton."):
                        return None
                    if fullname in sys.modules:
                        mod = sys.modules[fullname]
                        return mod.__spec__ if hasattr(mod, "__spec__") else None
                    return ModuleSpec(fullname, _TritonWin32Loader(), origin="mock")

            class _TritonWin32Loader(Loader):
                def create_module(self, spec):
                    mod = _AnyAttrModuleWin32(spec.name)
                    mod.__spec__ = spec
                    mod.__path__ = []
                    mod.__file__ = "mock"
                    mod.__loader__ = self
                    mod.__package__ = spec.name
                    return mod

                def exec_module(self, module):
                    if module.__name__ == "triton":
                        module.__version__ = "0.0.0.win32.mock"
                        module.jit = _make_triton_jit_win32()
                        module.autotune = lambda *a, **kw: lambda fn: fn
                        module.heuristics = lambda *a, **kw: lambda fn: fn
                        if "triton.language" not in sys.modules:
                            self._ensure("triton.language")
                    if module.__name__ == "triton.backends":
                        if "triton.backends.compiler" not in sys.modules:
                            self._ensure("triton.backends.compiler")

                def _ensure(self, name):
                    if name not in sys.modules:
                        spec = ModuleSpec(name, _TritonWin32Loader(), origin="mock")
                        mod = _AnyAttrModuleWin32(name)
                        mod.__spec__ = spec
                        mod.__path__ = []
                        mod.__file__ = "mock"
                        mod.__loader__ = self
                        mod.__package__ = name
                        sys.modules[name] = mod

            class _AnyAttrModuleWin32(types.ModuleType):
                def __getattr__(self, name):
                    if name.startswith("_"):
                        raise AttributeError(name)
                    child = _AnyAttrModuleWin32(self.__name__ + "." + name)
                    child.__spec__ = ModuleSpec(child.__name__, loader=None, origin="mock")
                    child.__path__ = []
                    child.__file__ = "mock"
                    child.__loader__ = None
                    child.__package__ = child.__name__
                    setattr(self, name, child)
                    return child

            def _make_triton_jit_win32():
                def jit(*args, **kwargs):
                    def decorator(fn):
                        return fn
                    if len(args) == 1 and callable(args[0]):
                        return args[0]
                    return decorator
                return jit

            sys.meta_path.insert(0, _TritonWin32Finder())
            spec_root = ModuleSpec("triton", _TritonWin32Loader(), origin="mock")
            root = _AnyAttrModuleWin32("triton")
            root.__spec__ = spec_root
            root.__path__ = []
            root.__file__ = "mock"
            root.__loader__ = _TritonWin32Loader()
            root.__package__ = "triton"
            root.__version__ = "0.0.0.win32.mock"
            root.jit = _make_triton_jit_win32()
            root.autotune = lambda *a, **kw: lambda fn: fn
            root.heuristics = lambda *a, **kw: lambda fn: fn
            sys.modules["triton"] = root
'''

    # 写入 mock 模块（仅在内容变化时写入，避免无谓的磁盘 I/O）
    try:
        with open(mock_module_path, "r", encoding="utf-8") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""
    if existing != _TRITON_MOCK_CODE:
        with open(mock_module_path, "w", encoding="utf-8") as f:
            f.write(_TRITON_MOCK_CODE)

    # 写入 .pth 文件（site.py 会在启动时执行其中的 import 行）
    pth_path = os.path.join(target_dir, "_triton_win32_mock.pth")
    pth_content = "import _triton_win32_mock\n"
    try:
        with open(pth_path, "r", encoding="utf-8") as f:
            existing_pth = f.read()
    except FileNotFoundError:
        existing_pth = ""
    if existing_pth != pth_content:
        with open(pth_path, "w", encoding="utf-8") as f:
            f.write(pth_content)

    import logging
    logging.getLogger(__name__).info(
        "Windows 环境: 已安装 triton mock .pth 到 %s (保护 DataLoader spawn 子进程)",
        target_dir,
    )


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
