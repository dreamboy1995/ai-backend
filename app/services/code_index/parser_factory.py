"""
Parser 工厂（S4 第 31-32 天）

负责：
1. 根据文件后缀自动检测语言
2. 返回对应语言的 tree-sitter Parser

设计要点：
- 语言模块延迟导入：单个语言包缺失不影响整体服务，仅该语言返回 None
- 兼容 tree-sitter 0.21+ API（language 对象直接传入 Parser）
- 不支持的语言或加载失败时返回 None，由调用方决定降级策略
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# 支持的语言及其文件后缀映射
# 每个语言对应 (语言标识, 语言模块名, Parser 构造)
LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "go": {".go"},
}

# 后缀 -> 语言标识 的反向映射
_EXT_TO_LANG = {}
for _lang, _exts in LANGUAGE_EXTENSIONS.items():
    for _ext in _exts:
        _EXT_TO_LANG[_ext] = _lang

# 缓存已加载的语言对象，避免重复导入
_language_cache: dict = {}


def detect_language(file_path: str) -> str:
    """
    根据文件后缀检测语言

    Args:
        file_path: 文件路径

    Returns:
        语言标识（python / javascript / typescript / java / go），
        未识别时返回 "unknown"
    """
    _, ext = os.path.splitext(file_path)
    return _EXT_TO_LANG.get(ext.lower(), "unknown")


def _load_language(language: str):
    """
    延迟加载 tree-sitter 语言对象

    tree-sitter 0.25 中，各语言包的 `language()` 返回 PyCapsule，
    需用 `tree_sitter.Language(capsule)` 包装为 Language 对象。
    同时兼容旧版（0.21-0.24）中 `language` 直接为 Language 对象的情况。

    Returns:
        tree-sitter Language 对象，加载失败返回 None
    """
    if language in _language_cache:
        return _language_cache[language]

    module_map = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "java": "tree_sitter_java",
        "go": "tree_sitter_go",
    }

    module_name = module_map.get(language)
    if module_name is None:
        _language_cache[language] = None
        return None

    try:
        import importlib
        from tree_sitter import Language
        mod = importlib.import_module(module_name)
    except ImportError as e:
        logger.warning(f"[ParserFactory] 语言包 {module_name} 未安装，{language} 将无法解析: {e}")
        _language_cache[language] = None
        return None

    # 各语言包暴露 language 的方式不同：
    # - python/javascript/java/go: language()
    # - typescript: language_typescript() / language_tsx()
    lang_attr = getattr(mod, "language", None)
    if lang_attr is None and language == "typescript":
        lang_attr = getattr(mod, "language_typescript", None)
    if lang_attr is None:
        logger.warning(f"[ParserFactory] {module_name} 未提供 language 属性")
        _language_cache[language] = None
        return None

    try:
        # 新版 (0.25+): language 是函数，返回 PyCapsule，需包装为 Language
        if callable(lang_attr):
            capsule = lang_attr()
            lang_obj = Language(capsule)
        else:
            # 旧版 (0.21-0.24): language 直接是 Language 对象
            lang_obj = lang_attr
    except Exception as e:
        logger.warning(f"[ParserFactory] 构造 {language} Language 对象失败: {e}")
        _language_cache[language] = None
        return None

    _language_cache[language] = lang_obj
    return lang_obj


def get_parser(language: str):
    """
    获取指定语言的 tree-sitter Parser

    Args:
        language: 语言标识

    Returns:
        tree_sitter.Parser 实例，不支持/加载失败时返回 None
    """
    if language == "unknown":
        return None

    lang_obj = _load_language(language)
    if lang_obj is None:
        return None

    try:
        from tree_sitter import Parser
        # tree-sitter 0.25 不支持 Parser(lang) 构造，需先创建再设置 language
        parser = Parser()
        parser.language = lang_obj
        return parser
    except Exception as e:
        logger.warning(f"[ParserFactory] 创建 {language} Parser 失败: {e}")
        return None


def get_parser_for_file(file_path: str):
    """
    根据文件路径直接获取对应的 Parser

    Args:
        file_path: 文件路径

    Returns:
        (parser, language) 元组；解析失败时 parser 为 None，language 可能为 "unknown"
    """
    language = detect_language(file_path)
    parser = get_parser(language)
    return parser, language


def is_supported(file_path: str) -> bool:
    """判断文件是否在支持的语言列表中"""
    return detect_language(file_path) != "unknown"
