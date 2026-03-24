"""统一环境配置加载。

从固定路径 ``env_config.yaml``（UniversalBenchmark 根目录）读取配置，
为 Agent 和 Memory 提供自定义参数，无需通过命令行指定配置文件。

配置文件示例::

    memory:
      memecho:
        custom_headers:
          X-Mark: dusk
      buffer: {}
      mem0: {}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger
from ruamel.yaml import YAML

_CONFIG_FILENAME = "env_config.yaml"
_cached_config: Optional[dict[str, Any]] = None


def _resolve_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / _CONFIG_FILENAME


def load_env_config(*, reload: bool = False) -> dict[str, Any]:
    """加载并缓存环境配置。

    配置文件路径固定为 ``UniversalBenchmark/env_config.yaml``。
    文件不存在时返回空字典，不会报错。
    """
    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config

    config_path = _resolve_config_path()
    if not config_path.exists():
        logger.debug(f"环境配置文件不存在: {config_path}，使用默认空配置")
        _cached_config = {}
        return _cached_config

    yaml = YAML()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.load(f)

    _cached_config = dict(raw) if isinstance(raw, dict) else {}
    logger.info(f"已加载环境配置: {config_path}")
    return _cached_config


def get_memory_config(memory_name: str) -> dict[str, Any]:
    """获取指定 Memory 类型的配置段。

    Parameters
    ----------
    memory_name:
        Memory 类型的标识名，如 ``"memecho"``、``"buffer"``、``"mem0"``。

    Returns
    -------
    dict[str, Any]
        该 Memory 类型在配置文件 ``memory:`` 段下的键值对。
        配置不存在时返回空字典。
    """
    config = load_env_config()
    memory_section = config.get("memory", {})
    if not isinstance(memory_section, dict):
        return {}
    result = memory_section.get(memory_name, {})
    return dict(result) if isinstance(result, dict) else {}


def get_agent_config() -> dict[str, Any]:
    """获取全局 Agent 配置段。"""
    config = load_env_config()
    agent_section = config.get("agent", {})
    return dict(agent_section) if isinstance(agent_section, dict) else {}
