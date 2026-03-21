from __future__ import annotations

import time
from typing import Any, Optional

import litellm
from loguru import logger

_NON_RETRYABLE = {
    "AuthenticationError",
    "BadRequestError",
    "NotFoundError",
    "ContentPolicyViolationError",
}


class LLMMixin:
    """通过 *litellm* 进行 LLM 交互的基础 Mixin。

    核心方法 :meth:`generate` **已经实现** —— 它调用
    ``litellm.completion()`` 并返回提取后的文本。
    子类通过以下方式定制行为：

    * 构造参数 – *model*、*provider*、*temperature* 等
    * :meth:`prepare_messages` – 发送前预处理消息列表
    * :meth:`parse_response` – 从原始 LLM 响应中提取/转换内容
    * :meth:`_build_completion_params` – 调整参数字典

    使用协作式 ``__init__`` —— 始终接收并转发 ``**kwargs``。
    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        llm_tag: str = "",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._provider = provider
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._llm_tag = llm_tag
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._extra_params: dict[str, Any] = {}
        self._last_raw_response: Any = None

    # ------------------------------------------------------------------
    # 模型标识
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """litellm 可识别的完整模型字符串（例如 ``openai/gpt-4o``）。"""
        if self._provider:
            return f"{self._provider}/{self._model}"
        return self._model

    # ------------------------------------------------------------------
    # 生成 - 通用实现
    # ------------------------------------------------------------------

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """调用 LLM 生成回复，失败时自动重试。

        处理流程::

            prepare_messages → litellm.completion (+ 重试) → parse_response

        认证、参数、模型不存在等不可恢复的错误会立即抛出；
        速率限制、网络超时、服务不可用等瞬时错误会按指数退避重试。

        额外的 *kwargs* 会转发给 ``litellm.completion()``，
        **仅在本次调用中** 覆盖已配置的默认参数。
        """
        prepared = self.prepare_messages(messages)
        params = self._build_completion_params()
        params.update(kwargs)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=prepared,
                    **params,
                )
                self._last_raw_response = response
                return self.parse_response(response)
            except Exception as exc:
                if type(exc).__name__ in _NON_RETRYABLE:
                    raise
                last_error = exc
                logger.warning(
                    f"LLM 调用失败 "
                    f"(第 {attempt + 1}/{self._max_retries} 次): "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_base_delay * (2 ** attempt))

        raise RuntimeError(
            f"LLM 在 {self._max_retries} 次重试后仍然失败"
        ) from last_error

    # ------------------------------------------------------------------
    # 钩子方法 - 子类可覆写以定制行为
    # ------------------------------------------------------------------

    def prepare_messages(
        self, messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """钩子：在发送给 LLM 之前对消息列表进行变换。

        可覆写以添加校验、格式转换、Token 预算截断等逻辑。
        默认实现为直接透传。
        """
        return messages

    def parse_response(self, response: Any) -> str:
        """钩子：从 litellm 的响应中提取所需内容。

        可覆写以实现结构化输出解析、工具调用处理、多选项选择等。
        """
        content: Any = response.choices[0].message.content
        return str(content) if content is not None else ""

    # ------------------------------------------------------------------
    # 参数组装
    # ------------------------------------------------------------------

    def _build_completion_params(self) -> dict[str, Any]:
        """组装 ``litellm.completion()`` 的参数字典。"""
        params: dict[str, Any] = {}
        if self._temperature is not None:
            params["temperature"] = self._temperature
        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens
        if self._top_p is not None:
            params["top_p"] = self._top_p
        params.update(self._extra_params)
        return params

    def set_extra_param(self, key: str, value: Any) -> None:
        """设置一个额外参数，将应用于后续所有调用。"""
        self._extra_params[key] = value

    # ------------------------------------------------------------------
    # 内省
    # ------------------------------------------------------------------

    @property
    def last_raw_response(self) -> Any:
        """最近一次 :meth:`generate` 调用返回的完整 litellm
        ``ModelResponse`` 对象（首次调用前为 ``None``）。"""
        return self._last_raw_response

    @property
    def llm_identifier(self) -> str:
        """当前 LLM 配置的可读标识符。"""
        if self._llm_tag:
            return f"{self.model_name}:{self._llm_tag}"
        return self.model_name
