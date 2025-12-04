"""
轻量级国际化(i18n)模块

支持中英双语切换，自动检测系统语言
"""
from __future__ import annotations

import locale
import os
import platform
from typing import Optional

# 当前语言
_current_lang: str = "zh"

# 翻译字典
_translations: dict[str, dict[str, str]] = {}


def set_language(lang: str) -> None:
    """设置当前语言"""
    global _current_lang
    if lang in ("zh", "en"):
        _current_lang = lang
    else:
        raise ValueError(f"Unsupported language: {lang}. Use 'zh' or 'en'.")


def get_language() -> str:
    """获取当前语言"""
    return _current_lang


def t(key: str, **kwargs) -> str:
    """
    翻译函数
    
    Args:
        key: 翻译键（使用中文原文作为键）
        **kwargs: 格式化参数
        
    Returns:
        翻译后的字符串
    """
    if _current_lang == "zh":
        # 中文直接返回key
        text = key
    else:
        # 英文查找翻译
        text = _translations.get(key, key)
    
    # 格式化
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    
    return text


def register_translations(translations: dict[str, str]) -> None:
    """注册翻译"""
    _translations.update(translations)


def detect_system_language() -> str:
    """
    自动检测系统语言
    
    检测顺序:
    1. 环境变量 BOOLQ_LANG (显式设置)
    2. Windows: GetUserDefaultUILanguage API
    3. Unix/macOS: LANG/LC_ALL 环境变量
    4. locale.getdefaultlocale() 作为fallback
    
    Returns:
        "zh" 或 "en"
    """
    # 1. 首先检查显式设置的环境变量
    explicit_lang = os.environ.get("BOOLQ_LANG", "").lower()
    if explicit_lang:
        if explicit_lang in ("en", "english"):
            return "en"
        elif explicit_lang in ("zh", "chinese", "zh_cn", "zh_tw"):
            return "zh"
    
    # 2. 根据操作系统获取系统语言
    system = platform.system()
    lang_code = None
    
    if system == "Windows":
        lang_code = _get_windows_language()
    else:
        lang_code = _get_unix_language()
    
    # 3. Fallback: 使用locale
    if not lang_code:
        lang_code = _get_locale_language()
    
    # 4. 判断是否为中文
    if lang_code:
        lang_lower = lang_code.lower()
        # 中文语言代码: zh, zh_cn, zh_tw, zh_hk, chinese
        if lang_lower.startswith("zh") or "chinese" in lang_lower:
            return "zh"
    
    # 默认返回英文（国际化默认）
    return "en"


def _get_windows_language() -> Optional[str]:
    """获取Windows系统UI语言"""
    try:
        import ctypes
        # GetUserDefaultUILanguage 返回语言ID (LANGID)
        # 0x0804 = 简体中文, 0x0404 = 繁体中文, 0x0409 = 英语(美国)
        windll = ctypes.windll.kernel32
        lang_id = windll.GetUserDefaultUILanguage()
        
        # 中文语言ID范围
        # 简体中文: 0x0804 (2052), 0x0004
        # 繁体中文: 0x0404 (1028), 0x0C04, 0x1004, 0x1404
        chinese_lang_ids = {0x0804, 0x0004, 0x0404, 0x0C04, 0x1004, 0x1404}
        
        if lang_id in chinese_lang_ids or (lang_id & 0xFF) == 0x04:
            return "zh_CN"
        
        # 也可以尝试获取更详细的locale名称
        try:
            # GetUserDefaultLocaleName 返回locale名称字符串
            get_locale_name = windll.GetUserDefaultLocaleName
            buf = ctypes.create_unicode_buffer(85)
            get_locale_name(buf, 85)
            return buf.value
        except Exception:
            pass
        
        return "en_US"
    except Exception:
        return None


def _get_unix_language() -> Optional[str]:
    """获取Unix/macOS系统语言"""
    # 检查语言相关的环境变量
    for var in ("LANG", "LC_ALL", "LC_MESSAGES", "LANGUAGE"):
        value = os.environ.get(var)
        if value:
            # 格式可能是 "zh_CN.UTF-8" 或 "en_US.UTF-8"
            return value.split(".")[0]
    return None


def _get_locale_language() -> Optional[str]:
    """使用locale模块获取语言（fallback）"""
    try:
        # 尝试获取默认locale
        lang, _ = locale.getdefaultlocale()
        return lang
    except Exception:
        pass
    
    try:
        # 备用方法
        lang = locale.getlocale()[0]
        return lang
    except Exception:
        pass
    
    return None


# 初始化时自动检测系统语言
_current_lang = detect_system_language()

# 导入翻译
from .translations import TRANSLATIONS
register_translations(TRANSLATIONS)

