"""
OpenAI-compatible LLM 客户端公共类。

封装 OpenAI Python SDK，提供统一的聊天补全接口。
支持任意兼容 OpenAI Chat Completions API 的后端（vLLM、Ollama、LM Studio 等）。

用法:
    from poly_distill.llm_client import LLMClient

    client = LLMClient(
        endpoint="http://localhost:8000/v1",
        model="qwen3-4b",
        api_key="sk-xxx",
    )
    response = client.chat("你好，请介绍一下自己")
    # 或传入 messages 列表
    response = client.chat([{"role": "user", "content": "你好"}])
"""

import json
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class LLMClient:
    """兼容 OpenAI Chat Completions API 的通用 LLM 客户端。

    Attributes:
        endpoint: API 端点地址（如 http://host:port/v1）。
        model:   模型名称。
        api_key: API Key（可为空字符串，兼容本地无需鉴权的服务）。
        timeout: 请求超时秒数。
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout: int = 600,
        max_retries: int = 0,
    ):
        # 从完整路径提取 base_url（去除 /chat/completions 后缀）
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url.rsplit("/chat/completions", 1)[0]

        self.endpoint = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

        from openai import OpenAI
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(
        self,
        messages: Union[str, List[dict]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """发送聊天请求并返回模型回复文本。

        Args:
            messages:    用户消息字符串，或完整的 messages 列表。
            system:      系统提示词（仅当 messages 为字符串时生效）。
            temperature: 采样温度（0.0 = 贪婪解码）。
            max_tokens:  最大生成 token 数。
            top_p:       nucleus 采样。
            seed:        随机种子（None = 不设置）。

        Returns:
            模型回复的文本内容。

        Raises:
            RuntimeError: API 调用失败时抛出。
        """
        # 构建 messages 列表
        if isinstance(messages, str):
            msg_list = []
            if system:
                msg_list.append({"role": "system", "content": system})
            msg_list.append({"role": "user", "content": messages})
        else:
            msg_list = messages

        try:
            extra = {}
            if seed is not None:
                extra["seed"] = seed
            response = self._client.chat.completions.create(
                model=self.model,
                messages=msg_list,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **extra,
            )
            logger.debug("LLM API 调用成功, model=%s, tokens=%s",
                         self.model, response.usage)
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(
                f"LLM API 调用失败 (endpoint={self.endpoint}, model={self.model}): {e}"
            ) from e

    def chat_json(
        self,
        messages: Union[str, List[dict]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> dict:
        """发送聊天请求并解析 JSON 返回。

        自动处理 markdown 代码块包裹（```json ... ```）以及
        响应中混入的非 JSON 文本。

        Args:
            同 chat()。

        Returns:
            解析后的 dict。解析失败时返回 {"error": ..., "raw": ...}。
        """
        content = self.chat(messages, system, temperature, max_tokens, top_p, seed)
        content = content.strip()

        # 去除 markdown 代码块标记
        if content.startswith("```"):
            first_nl = content.find("\n")
            if first_nl != -1:
                content = content[first_nl + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        # 提取 JSON 对象
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError as e:
                logger.warning("LLM 返回 JSON 解析失败: %s", e)
                return {"error": "json_parse_error", "raw": content[:500]}

        logger.warning("LLM 返回非 JSON 格式:\n%s", content[:500])
        return {"error": "non_json_response", "raw": content[:500]}
