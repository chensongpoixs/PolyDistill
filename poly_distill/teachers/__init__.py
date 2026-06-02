"""
多教师适配器层。

为每个商业 LLM API 提供统一接口：
  - generate(prompt) → response
  - 内置重试、速率限制、token 统计
"""
