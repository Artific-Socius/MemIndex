"""
ExampleAgent - 示例 Agent 实现

用于演示如何实现自定义 Agent。
"""

from __future__ import annotations

import asyncio
import datetime

import requests

from .base_agent import BaseAgent


class ExampleAgent(BaseAgent):
    """
    示例 Agent 实现

    通过 HTTP API 与外部服务交互。
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000/conversation"):
        """
        初始化 Example Agent

        Args:
            api_url: 外部服务 API URL
        """
        super().__init__("Example Agent")
        self.api_url = api_url

    async def send_message(self, message: str) -> str:
        """
        发送消息并获取回复

        Args:
            message: 用户消息

        Returns:
            API 回复
        """
        response = await asyncio.to_thread(
            requests.post,
            self.api_url,
            json={
                "text": message,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            }
        )
        response_data = response.json()

        # 更新对话历史
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response_data["data"]})

        return response_data["data"]



