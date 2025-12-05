"""
PromptManager - 提示词管理器

提供提示词的加载和管理功能，支持继承机制。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set

from ruamel.yaml import YAML
from loguru import logger


# 模块根目录
PROMPTS_DIR = Path(__file__).parent

# 继承关键字
EXTENDS_KEY = "_extends"


def _is_prompt_debug() -> bool:
    """检查是否启用 prompt 调试日志"""
    return os.environ.get("PROMPT_DEBUG_LOG", "").lower() in ("1", "true", "yes", "on")


def _debug_log(message: str) -> None:
    """仅在 PROMPT_DEBUG_LOG 环境变量启用时打印调试日志"""
    if _is_prompt_debug():
        logger.info(f"[PROMPT_DEBUG] {message}")


class CircularInheritanceError(Exception):
    """循环继承错误"""
    pass


class PromptManager:
    """
    提示词管理器
    
    负责加载和管理 prompts.yaml 中的提示词配置。
    支持继承机制：通过 _extends 字段指定父配置，子配置会继承父配置的所有字段，
    并可以覆盖特定字段。
    """
    
    def __init__(self, prompts_file: str = None):
        """
        初始化提示词管理器
        
        Args:
            prompts_file: 提示词配置文件路径，默认为 prompts/prompts.yaml
        """
        if prompts_file is None:
            prompts_file = PROMPTS_DIR / "prompts.yaml"
        
        self.prompts_file = Path(prompts_file)
        self.yaml = YAML()
        self._prompts_data: dict = {}
        self._defaults: dict = {}
        self._eval_templates: dict = {}  # 评估提示词公共模板
        self._chat_prompts_raw: dict = {}  # 原始配置（含 _extends）
        self._eval_prompts_raw: dict = {}  # 原始配置（含 _extends）
        self._chat_prompts: dict = {}  # 解析后的配置（已展开继承）
        self._eval_prompts: dict = {}  # 解析后的配置（已展开继承）
        
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """加载提示词配置文件"""
        if not self.prompts_file.exists():
            logger.warning(f"Prompts file not found: {self.prompts_file}")
            self._set_default_prompts()
            return
        
        try:
            with self.prompts_file.open("r", encoding="utf-8") as f:
                self._prompts_data = self.yaml.load(f) or {}
            
            self._defaults = self._prompts_data.get("defaults", {})
            self._eval_templates = self._prompts_data.get("eval_templates", {})
            self._chat_prompts_raw = self._prompts_data.get("chat", {})
            self._eval_prompts_raw = self._prompts_data.get("eval", {})
            
            # 解析继承关系
            self._chat_prompts = self._resolve_inheritance(self._chat_prompts_raw, "chat")
            self._eval_prompts = self._resolve_inheritance(self._eval_prompts_raw, "eval")
            
            logger.debug(f"Loaded prompts from {self.prompts_file}")
            logger.debug(f"Available chat prompt keys: {list(self._chat_prompts.keys())}")
            logger.debug(f"Available eval prompt keys: {list(self._eval_prompts.keys())}")
            
            _debug_log(f"Prompts loaded from: {self.prompts_file}")
            _debug_log(f"Default chat key: {self._defaults.get('chat', 'default')}")
            _debug_log(f"Default eval key: {self._defaults.get('eval', 'default')}")
            _debug_log(f"Available chat keys: {list(self._chat_prompts.keys())}")
            _debug_log(f"Available eval keys: {list(self._eval_prompts.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            self._set_default_prompts()
    
    def _resolve_inheritance(self, prompts_raw: dict, category: str) -> dict:
        """
        解析继承关系，展开所有继承
        
        Args:
            prompts_raw: 原始配置字典
            category: 配置类别（chat 或 eval）
            
        Returns:
            展开继承后的配置字典
        """
        resolved = {}
        
        for key in prompts_raw:
            try:
                resolved[key] = self._resolve_single(key, prompts_raw, category, set())
            except CircularInheritanceError as e:
                logger.error(f"Circular inheritance detected in {category}.{key}: {e}")
                # 回退到不使用继承的原始配置
                raw_config = dict(prompts_raw[key])
                raw_config.pop(EXTENDS_KEY, None)
                resolved[key] = raw_config
        
        return resolved
    
    def _resolve_single(
        self, 
        key: str, 
        prompts_raw: dict, 
        category: str,
        visited: Set[str]
    ) -> dict:
        """
        解析单个配置的继承关系
        
        Args:
            key: 配置 key
            prompts_raw: 原始配置字典
            category: 配置类别
            visited: 已访问的 key 集合（用于检测循环）
            
        Returns:
            展开继承后的配置
        """
        if key in visited:
            raise CircularInheritanceError(f"Circular inheritance chain: {' -> '.join(visited)} -> {key}")
        
        if key not in prompts_raw:
            logger.warning(f"{category}.{key} not found")
            return {}
        
        raw_config = prompts_raw[key]
        if not isinstance(raw_config, dict):
            return {}
        
        extends_key = raw_config.get(EXTENDS_KEY)
        
        if extends_key is None:
            # 没有继承，直接返回（去除 _extends 字段）
            result = dict(raw_config)
            result.pop(EXTENDS_KEY, None)
            _debug_log(f"Resolved {category}.{key} (no inheritance)")
            return result
        
        # 有继承关系
        visited = visited | {key}
        
        # 递归解析父配置
        parent_config = self._resolve_single(extends_key, prompts_raw, category, visited)
        
        # 合并：父配置 + 子配置覆盖
        result = dict(parent_config)
        for k, v in raw_config.items():
            if k != EXTENDS_KEY:
                result[k] = v
        
        _debug_log(f"Resolved {category}.{key} (extends: {extends_key})")
        
        return result
    
    def _set_default_prompts(self) -> None:
        """设置默认提示词（当配置文件不存在或加载失败时使用）"""
        self._defaults = {
            "chat": "default",
            "eval": "default",
        }
        
        # 评估提示词公共模板
        self._eval_templates = {
            "data_format": """
{question_section}
评价标准/标准答案:
<ground-truth>
{ground}
</ground-truth>

目标回答:
<target>
{target}
</target>
""",
            "question_format": """
本轮问答的问题:
<question>
{question}
</question>
""",
            "output_binary": """
你的回答必须遵守以下JSON格式:
{{"reason": string, "answer": boolean}}
""",
            "output_score": """
你的回答必须遵守以下JSON格式:
{{"reason": string, "score": float}}

其中 score 必须是 0 到 1 之间的数字。
""",
            "output_multi_score": """
你的回答必须遵守以下JSON格式:
[
    {{"reason": string, "score": float}}
]

***注意: 所有子项得分的总和必须为 1.0***
""",
        }
        
        self._chat_prompts = {
            "default": {
                "system_prompt": "你是一个有帮助的AI助手。",
                "memory_context_prefix": "以下是与当前对话相关的历史记忆：\n",
                "delete_request_check": "你是一个意图分类器。判断用户的消息是否是要求删除/忘记/移除某些记忆或信息的请求。只回答 YES 或 NO。",
                "select_memories_to_delete": "你是一个记忆管理助手。根据用户的删除请求，从记忆列表中选择需要删除的记忆。\n只输出需要删除的记忆ID，每行一个。如果没有需要删除的记忆，输出 NONE。\n不要输出任何其他内容。",
            }
        }
        
        # 评估提示词（只包含任务描述，模板会自动拼接）
        self._eval_prompts = {
            "default": {
                "binary_evaluation": """你是一个答案评估模型，你需要根据<ground-truth>中的标准答案或者评估标准评估<target>中的目标答案是否正确。
并且返回json格式的评估结果，其中要包含因为目标中的什么符合要求，什么不符合要求，给出理由。
""",
                "multi_score_evaluation": """你是一个答案评估模型，你需要根据<ground-truth>中的标准答案或者评估标准评估<target>中的目标答案的每个小目标是否符合要求。
整个评估的任务的总分为1.0，请根据要求与评估目标给出每个小目标的得分和理由。
""",
                "score_evaluation": """你是一个答案评估模型，你需要根据<ground-truth>中的标准答案或评估标准，对<target>中的目标答案进行评分。

评分规则：
- 给出一个 0 到 1 之间的分数（可以是小数，如 0.5, 0.75 等）
- 0 表示完全不正确
- 1 表示完全正确
- 中间分数表示部分正确的程度
""",
                "post_process": """你是一个强大的处理模型，你需要按照要求<require>中的要求处理<target>中的文本，并且只返回处理的结果
如下是数据:

处理要求:
<require>
{require}
</require>

目标数据:
<target>
{target}
</target>""",
            }
        }
        
        self._chat_prompts_raw = self._chat_prompts
        self._eval_prompts_raw = self._eval_prompts
    
    @property
    def default_chat_key(self) -> str:
        """获取默认的 chat prompt key"""
        return self._defaults.get("chat", "default")
    
    @property
    def default_eval_key(self) -> str:
        """获取默认的 eval prompt key"""
        return self._defaults.get("eval", "default")
    
    @property
    def available_chat_keys(self) -> list[str]:
        """获取所有可用的 chat prompt keys"""
        return list(self._chat_prompts.keys())
    
    @property
    def available_eval_keys(self) -> list[str]:
        """获取所有可用的 eval prompt keys"""
        return list(self._eval_prompts.keys())
    
    def get_inheritance_info(self, category: str, key: str) -> str | None:
        """
        获取指定配置的继承信息
        
        Args:
            category: 配置类别（chat 或 eval）
            key: 配置 key
            
        Returns:
            继承的父 key，如果没有继承则返回 None
        """
        raw_prompts = self._chat_prompts_raw if category == "chat" else self._eval_prompts_raw
        if key in raw_prompts and isinstance(raw_prompts[key], dict):
            return raw_prompts[key].get(EXTENDS_KEY)
        return None
    
    def get_chat_prompt(self, key: str = None, prompt_name: str = None) -> str:
        """
        获取 chat 提示词
        
        Args:
            key: prompt key，默认使用 defaults.chat
            prompt_name: 具体的提示词名称（如 system_prompt, memory_context_prefix 等）
            
        Returns:
            提示词字符串
        """
        original_key = key
        if key is None:
            key = self.default_chat_key
        
        prompts = self._chat_prompts.get(key)
        if prompts is None:
            logger.warning(f"Chat prompt key '{key}' not found, using default")
            prompts = self._chat_prompts.get(self.default_chat_key, {})
            key = self.default_chat_key
        
        if prompt_name is None:
            _debug_log(f"get_chat_prompt(key={original_key}) -> resolved to key='{key}', returning all prompts")
            return prompts
        
        prompt = prompts.get(prompt_name)
        if prompt is None:
            # 回退到默认 key
            default_prompts = self._chat_prompts.get(self.default_chat_key, {})
            prompt = default_prompts.get(prompt_name, "")
            if prompt:
                logger.debug(f"Chat prompt '{prompt_name}' not found in '{key}', using default")
                _debug_log(f"get_chat_prompt(key={original_key}, prompt_name={prompt_name}) -> FALLBACK to default")
        
        # 调试日志：打印获取的 prompt 预览
        if prompt:
            preview = prompt[:100].replace('\n', '\\n') + ('...' if len(prompt) > 100 else '')
            _debug_log(f"get_chat_prompt(key={key}, prompt_name={prompt_name}) -> '{preview}'")
        else:
            _debug_log(f"get_chat_prompt(key={key}, prompt_name={prompt_name}) -> EMPTY/NOT FOUND")
        
        return prompt or ""
    
    def get_eval_prompt(self, key: str = None, prompt_name: str = None) -> str:
        """
        获取 eval 提示词
        
        Args:
            key: prompt key，默认使用 defaults.eval
            prompt_name: 具体的提示词名称（如 binary_evaluation, multi_score_evaluation 等）
            
        Returns:
            提示词字符串
        """
        original_key = key
        if key is None:
            key = self.default_eval_key
        
        prompts = self._eval_prompts.get(key)
        if prompts is None:
            logger.warning(f"Eval prompt key '{key}' not found, using default")
            prompts = self._eval_prompts.get(self.default_eval_key, {})
            key = self.default_eval_key
        
        if prompt_name is None:
            _debug_log(f"get_eval_prompt(key={original_key}) -> resolved to key='{key}', returning all prompts")
            return prompts
        
        prompt = prompts.get(prompt_name)
        if prompt is None:
            # 回退到默认 key
            default_prompts = self._eval_prompts.get(self.default_eval_key, {})
            prompt = default_prompts.get(prompt_name, "")
            if prompt:
                logger.debug(f"Eval prompt '{prompt_name}' not found in '{key}', using default")
                _debug_log(f"get_eval_prompt(key={original_key}, prompt_name={prompt_name}) -> FALLBACK to default")
        
        # 调试日志：打印获取的 prompt 预览
        if prompt:
            preview = prompt[:100].replace('\n', '\\n') + ('...' if len(prompt) > 100 else '')
            _debug_log(f"get_eval_prompt(key={key}, prompt_name={prompt_name}) -> '{preview}'")
        else:
            _debug_log(f"get_eval_prompt(key={key}, prompt_name={prompt_name}) -> EMPTY/NOT FOUND")
        
        return prompt or ""
    
    def get_chat_system_prompt(self, key: str = None) -> str:
        """获取 chat 系统提示词"""
        return self.get_chat_prompt(key, "system_prompt")
    
    def get_chat_memory_context_prefix(self, key: str = None) -> str:
        """获取记忆上下文前缀"""
        return self.get_chat_prompt(key, "memory_context_prefix")
    
    def get_chat_delete_request_check(self, key: str = None) -> str:
        """获取删除请求判断提示词"""
        return self.get_chat_prompt(key, "delete_request_check")
    
    def get_chat_select_memories_to_delete(self, key: str = None) -> str:
        """获取选择要删除记忆的提示词"""
        return self.get_chat_prompt(key, "select_memories_to_delete")
    
    def _compose_eval_prompt(
        self, 
        key: str, 
        prompt_name: str, 
        output_template_name: str,
        question: Optional[str] = None,
    ) -> str:
        """
        组合评估提示词：任务描述 + 数据格式模板 + 输出格式模板
        
        Args:
            key: prompt key
            prompt_name: 提示词名称（如 binary_evaluation）
            output_template_name: 输出格式模板名称（如 output_binary）
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）
            
        Returns:
            组合后的完整提示词模板
        """
        # 获取任务描述部分
        task_description = self.get_eval_prompt(key, prompt_name)
        
        # 获取公共模板
        data_format = self._eval_templates.get("data_format", "")
        output_format = self._eval_templates.get(output_template_name, "")
        
        # 处理可选的 question 部分
        if question:
            question_format = self._eval_templates.get("question_format", "")
            question_section = question_format.format(question=question)
        else:
            question_section = ""
        
        # 先填充 question_section
        data_format = data_format.replace("{question_section}", question_section)
        
        # 组合：任务描述 + 数据格式 + 输出格式
        composed = task_description + data_format + output_format
        
        _debug_log(f"_compose_eval_prompt(key={key}, prompt_name={prompt_name}, has_question={bool(question)}) -> composed with templates")
        
        return composed
    
    def get_eval_binary_prompt(self, key: str = None, question: Optional[str] = None) -> str:
        """
        获取二元评估提示词（已组合模板）
        
        Args:
            key: prompt key
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）
        
        返回：任务描述 + 数据格式模板 + 二元输出格式模板
        """
        return self._compose_eval_prompt(key, "binary_evaluation", "output_binary", question)
    
    def get_eval_multi_score_prompt(self, key: str = None, question: Optional[str] = None) -> str:
        """
        获取多分数评估提示词（已组合模板）
        
        Args:
            key: prompt key
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）
        
        返回：任务描述 + 数据格式模板 + 多分数输出格式模板
        """
        return self._compose_eval_prompt(key, "multi_score_evaluation", "output_multi_score", question)
    
    def get_eval_post_process_prompt(self, key: str = None) -> str:
        """获取后处理提示词（不使用公共模板）"""
        return self.get_eval_prompt(key, "post_process")
    
    def get_eval_score_prompt(self, key: str = None, question: Optional[str] = None) -> str:
        """
        获取分数评估提示词（已组合模板，0-1连续分数）
        
        Args:
            key: prompt key
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）
        
        返回：任务描述 + 数据格式模板 + 分数输出格式模板
        """
        return self._compose_eval_prompt(key, "score_evaluation", "output_score", question)


# 全局单例
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(prompts_file: str = None) -> PromptManager:
    """
    获取提示词管理器单例
    
    Args:
        prompts_file: 提示词配置文件路径
        
    Returns:
        PromptManager 实例
    """
    global _prompt_manager
    
    if _prompt_manager is None or (prompts_file is not None and str(_prompt_manager.prompts_file) != prompts_file):
        _prompt_manager = PromptManager(prompts_file)
    
    return _prompt_manager


def reset_prompt_manager() -> None:
    """重置提示词管理器单例"""
    global _prompt_manager
    _prompt_manager = None
