"""
S9 第 81-82 天：智能错误解析器（Error Parser）

这是 Agent 自修复能力的"眼睛"——它把终端报错日志转化为结构化的错误对象，
让后续的自修复循环（第 83-84 天）能够精准定位问题、生成修复方案。

核心能力：
1. 多语言/工具链报错解析：
   - Python Traceback
   - JavaScript/TypeScript Error + Stack Trace
   - 编译器（gcc/clang/tsc）格式
2. 降级策略：无法匹配时截取最后 200 字符（通常包含最核心的错误描述）
3. 返回统一的 ParsedError 结构，供自修复 Prompt 直接使用

设计要点（兼顾 S9 关键接口与风险预警）：
1. ParsedError 结构与 SSE 中 repair_attempt 事件的 error 字段对齐：
   { type, message, file, line } —— 插件端 Builder 面板可直接消费
2. 每个语言解析器独立为纯函数，返回 Optional[ParsedError]，
   按优先级依次尝试，第一个命中即返回（避免正则互相干扰）
3. file_path 统一转换为相对于 cwd 的路径，便于前端定位跳转
4. 风险预警"大模型幻觉修复"：ParsedError 附带 parse_strategy 字段，
   后续自修复循环可根据解析策略的置信度（编译器 > Traceback > JS > fallback）
   决定是否需要额外验证

Sprint_9.md 验收标准：
  输入：一段 Python NameError: name 'x' is not defined 的完整 Traceback
  输出：{file_path: 'main.py', line: 5, error_type: 'NameError', message: "name 'x' is not defined"}
"""

import logging
import os
import re
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构：ParsedError
# ============================================================
# 与 S9 关键接口中 repair_attempt 事件的 error 字段对齐：
# { type: string, message: string, file: string, line: number }
# 额外字段（language, column, code_snippet, parse_strategy）供后端内部使用

class ParsedError(BaseModel):
    """
    解析后的结构化错误对象。

    所有字段都兼容 JSON 序列化，便于：
      - 通过 SSE repair_attempt 事件推送给插件端
      - 直接注入自修复 Prompt
      - 写入审计日志
    """

    error_type: str = Field(
        default="Unknown",
        description="错误类型（如 NameError / TypeError / SyntaxError）",
    )
    error_message: str = Field(
        default="",
        description="错误的核心描述（如 'name x is not defined'）",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="错误发生的文件路径（相对 cwd）",
    )
    line_number: Optional[int] = Field(
        default=None,
        description="错误发生的行号（1-based）",
    )
    column: Optional[int] = Field(
        default=None,
        description="错误发生的列号（主要用于 JS/TS 和编译器）",
    )
    code_snippet: Optional[str] = Field(
        default=None,
        description="错误位置附近的代码片段（用于自修复 Prompt）",
    )
    language: Optional[str] = Field(
        default=None,
        description="检测到的语言：python / javascript / typescript / c / unknown",
    )
    parse_strategy: str = Field(
        default="fallback",
        description="解析策略：python_traceback / js_stack / ts_compiler / gcc / fallback",
    )

    def to_sse_dict(self) -> dict:
        """转换为 SSE repair_attempt 事件中 error 字段的格式"""
        return {
            "type": self.error_type,
            "message": self.error_message,
            "file": self.file_path,
            "line": self.line_number,
        }

    def __str__(self) -> str:
        """人类可读的错误摘要，用于日志和 Prompt"""
        parts = [f"[{self.error_type}] {self.error_message}"]
        if self.file_path and self.line_number:
            parts.append(f"at {self.file_path}:{self.line_number}")
        elif self.file_path:
            parts.append(f"at {self.file_path}")
        return " ".join(parts)


# ============================================================
# 公共工具函数
# ============================================================

def _normalize_path(path: str, cwd: str) -> str:
    """
    将文件路径规范化为相对 cwd 的路径。

    策略：
      1. 输入是相对路径（如 "main.py" / "src/utils.py"）→ 直接保留原样
         （不做 abspath 解析，避免在不同平台变成不同的绝对路径）
      2. 输入是绝对路径且在 cwd 下 → 转为相对 cwd 的路径
      3. 输入是绝对路径但不在 cwd 下 → 保留绝对路径

    Windows 反斜杠统一转为正斜杠。
    同时兼容 Unix 风格绝对路径（/usr/lib/...）——即使在 Windows 上解析错误日志时也能识别。
    """
    if not path:
        return path
    try:
        # Windows 特判：以 / 开头的 Unix 风格路径（来自 Unix 环境的错误日志）
        # os.path.isabs 在 Windows 上对 /xxx 返回 False，需要手动识别
        unix_style_abs = path.startswith("/") or path.startswith("\\")

        # 在当前平台判断是否绝对路径
        is_absolute = os.path.isabs(path) or unix_style_abs

        if not is_absolute:
            return path.replace("\\", "/")

        # 绝对路径：尝试转为相对 cwd
        # Windows 上 path 和 cwd 可能是不同风格（path 来自 Unix 日志是 /usr/...，
        # cwd 是 Windows 风格如 G:/workspace），直接字符串比较不可靠。
        # 这里用正斜杠统一后做 startswith 比较。
        path_norm = path.replace("\\", "/")
        cwd_norm = (cwd or "").replace("\\", "/").rstrip("/")

        if cwd_norm and path_norm.startswith(cwd_norm + "/"):
            rel = path_norm[len(cwd_norm) + 1:]
            return rel
        elif cwd_norm and path_norm == cwd_norm:
            return ""

        # 不在 cwd 下，保留绝对路径
        return path_norm
    except Exception:
        return path.replace("\\", "/")


def _extract_code_snippet(raw: str, file_path: Optional[str], line_number: Optional[int], cwd: str) -> Optional[str]:
    """
    从原始错误日志中提取错误位置附近的代码片段。

    Python Traceback 自带代码行（缩进后的那一行），JS stack trace 不带。
    这里简单处理：如果 Traceback 中有源码行（缩进 4+ 空格的非空行），取最后一段。
    更精确的实现可以在后续版本用 read_file 工具读取实际文件。
    """
    if not raw or line_number is None:
        return None

    lines = raw.splitlines()
    code_lines = []

    # Python Traceback 的源码行缩进 4 个空格以上
    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("    ") and not stripped.startswith("    File"):
            code_lines.append(stripped.strip())

    if code_lines:
        # 取最后 3 行（最靠近错误位置）
        return "\n".join(code_lines[-3:])

    return None


# ============================================================
# 1. Python Traceback 解析器
# ============================================================
# 典型格式：
# Traceback (most recent call last):
#   File "main.py", line 5, in <module>
#     x = y
# NameError: name 'y' is not defined
#
# 或者嵌套调用：
# Traceback (most recent call last):
#   File "runner.py", line 10, in run
#     func()
#   File "main.py", line 5, in func
#     x = y
# NameError: name 'y' is not defined
#
# 关键：取最后一个 File 行（innermost），错误行是 Traceback 的最后一行

# 匹配 Traceback 头
_PY_TRACEBACK_HEADER = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)

# 匹配 File 行：  File "path", line N, in function_name
_PY_FILE_LINE = re.compile(
    r'^\s*File\s+"([^"]+)",\s*line\s+(\d+),\s*in\s+(.+)$',
    re.MULTILINE,
)

# 匹配最后的错误行：ErrorType: message
_PY_ERROR_LINE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9_]*Error(?:\.[A-Z][A-Za-z0-9_]*)?)\s*:\s*(.+)$",
    re.MULTILINE,
)


def _parse_python_traceback(raw: str, cwd: str) -> Optional[ParsedError]:
    """
    尝试解析 Python Traceback 格式的错误。

    Returns:
        ParsedError 或 None（非 Python Traceback）
    """
    if not raw:
        return None

    # 先检查是否有 Traceback header
    if not _PY_TRACEBACK_HEADER.search(raw):
        return None

    # 提取所有 File 行，取最后一个（最内层错误）
    file_matches = list(_PY_FILE_LINE.finditer(raw))
    if not file_matches:
        # 没有 File 行，但有 Traceback header —— 罕见情况，降级
        pass

    # 提取错误类型和消息（Traceback 的最后一行）
    error_matches = list(_PY_ERROR_LINE.finditer(raw))
    if not error_matches:
        # Python 错误必须有 ErrorType: message 格式，否则不是完整 Traceback
        # 但如果有 Traceback header，我们还是尽力提取
        if not file_matches:
            return None

    error_match = error_matches[-1] if error_matches else None
    file_match = file_matches[-1] if file_matches else None

    error_type = error_match.group(1) if error_match else "UnknownError"
    error_message = error_match.group(2).strip() if error_match else raw.strip()[:200]

    file_path = None
    line_number = None
    if file_match:
        file_path = _normalize_path(file_match.group(1), cwd)
        line_number = int(file_match.group(2))

    snippet = _extract_code_snippet(raw, file_path, line_number, cwd)

    return ParsedError(
        error_type=error_type,
        error_message=error_message,
        file_path=file_path,
        line_number=line_number,
        code_snippet=snippet,
        language="python",
        parse_strategy="python_traceback",
    )


# ============================================================
# 2. JavaScript / TypeScript Stack Trace 解析器
# ============================================================
# 典型格式（Node.js / 浏览器）：
# ReferenceError: x is not defined
#     at foo (/home/user/app.js:10:5)
#     at bar (/home/user/app.js:5:10)
#     at Object.<anonymous> (/home/user/index.js:1:1)
#
# 带 TypeScript source map 时：
# Error: something went wrong
#     at functionName (main.ts:10:5)
#     at main (main.ts:20:3)

# 匹配 JS 错误头：
#   "ReferenceError: x is not defined"
#   "TypeError: boom"
#   "Error: something went wrong"  ← 纯 Error 也要匹配
_JS_ERROR_HEADER = re.compile(
    r"^(Error|[A-Z][A-Za-z0-9_]*Error)\s*:\s*(.+)?$",
    re.MULTILINE,
)

# 匹配 at 行：  at functionName (file:line:col)  或  at file:line:col
_JS_STACK_LINE = re.compile(
    r"^\s*at\s+(?:.+?\s+\()?(.+?):(\d+):(\d+)\)?$",
    re.MULTILINE,
)


def _parse_js_stack(raw: str, cwd: str) -> Optional[ParsedError]:
    """
    尝试解析 JavaScript/TypeScript 堆栈格式的错误。

    Returns:
        ParsedError 或 None（非 JS 堆栈）
    """
    if not raw:
        return None

    # 必须有 at 行（JS 堆栈的标志）
    stack_matches = list(_JS_STACK_LINE.finditer(raw))
    if not stack_matches:
        return None

    # 错误头是第一行匹配 ErrorType: message
    error_matches = list(_JS_ERROR_HEADER.finditer(raw))
    if not error_matches:
        # 有堆栈但没错误头（罕见），用第一行堆栈的文件名当 fallback
        first_stack = stack_matches[0]
        file_path = _normalize_path(first_stack.group(1), cwd)
        line_number = int(first_stack.group(2))
        column = int(first_stack.group(3))
        return ParsedError(
            error_type="Error",
            error_message=raw.strip().splitlines()[0][:200],
            file_path=file_path,
            line_number=line_number,
            column=column,
            language="javascript",
            parse_strategy="js_stack",
        )

    # 取最后一个堆栈行（错误源头）——与 Python 相反，JS 的堆栈从上到下是从调用者到被调用者
    # 但实际应该看第一个有文件路径的堆栈（最内层），这里我们取第一个
    stack = stack_matches[0]
    file_path = _normalize_path(stack.group(1), cwd)
    line_number = int(stack.group(2))
    column = int(stack.group(3))

    # 根据文件扩展名判断 JS 还是 TS
    ext = os.path.splitext(file_path)[1].lower() if file_path else ""
    language = "typescript" if ext in (".ts", ".tsx") else "javascript"

    error_type = error_matches[0].group(1)
    error_message = (error_matches[0].group(2) or "").strip()

    return ParsedError(
        error_type=error_type,
        error_message=error_message,
        file_path=file_path,
        line_number=line_number,
        column=column,
        language=language,
        parse_strategy="js_stack",
    )


# ============================================================
# 3. 编译器错误解析器（gcc / clang / tsc）
# ============================================================
# gcc/clang 格式：
# main.c:10:5: error: expected ';' before '}'
#    10 |     }
#       |     ^
#
# 或没有位置：
# main.c:10: error: expected ';' before '}'
#
# TypeScript tsc 格式：
# main.ts(10,5): error TS2307: Cannot find module 'foo'
#
# Rustc 格式：
# error[E0425]: cannot find value `x` in this scope
#  --> main.rs:5:10
#   |
# 5 |     let y = x;
#   |          ^
#

# gcc/clang 格式：
#   file:line[:col]: error: message  （标准格式）
#   file:line: error message         （无冒号，gcc make 的某些错误）
_COMPILER_GCC = re.compile(
    r"^\s*(.+?):(\d+)(?::(\d+))?:\s*error:?\s*(.+)$",
    re.MULTILINE,
)

# TypeScript tsc 格式：file(line,col): error TSXXXX: message
_TSC_FORMAT = re.compile(
    r'^\s*(.+?)\((\d+),(\d+)\):\s*error\s+TS\d+:\s*(.+)$',
    re.MULTILINE,
)

# Rustc 格式：error[EXXXX]: message 然后 --> file:line:col
_RUSTC_ERROR = re.compile(
    r"^\s*error(?:\[(E\d+)\])?:\s*(.+)$",
    re.MULTILINE,
)
_RUSTC_LOCATION = re.compile(
    r"^\s*-->\s*(.+?):(\d+):(\d+)$",
    re.MULTILINE,
)


def _parse_compiler_error(raw: str, cwd: str) -> Optional[ParsedError]:
    """
    尝试解析编译器格式的错误（gcc/clang/tsc/rustc）。

    Returns:
        ParsedError 或 None（非编译器格式）
    """
    if not raw:
        return None

    # TypeScript tsc
    tsc_matches = list(_TSC_FORMAT.finditer(raw))
    if tsc_matches:
        m = tsc_matches[-1]  # 取最后一个错误
        return ParsedError(
            error_type="CompileError",
            error_message=m.group(4).strip(),
            file_path=_normalize_path(m.group(1), cwd),
            line_number=int(m.group(2)),
            column=int(m.group(3)),
            language="typescript",
            parse_strategy="ts_compiler",
        )

    # gcc/clang
    gcc_matches = list(_COMPILER_GCC.finditer(raw))
    if gcc_matches:
        m = gcc_matches[-1]  # 取最后一个错误
        col = int(m.group(3)) if m.group(3) else None
        file_ext = os.path.splitext(m.group(1))[1].lower()
        lang_map = {".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp"}
        language = lang_map.get(file_ext, "c")
        return ParsedError(
            error_type="CompileError",
            error_message=m.group(4).strip(),
            file_path=_normalize_path(m.group(1), cwd),
            line_number=int(m.group(2)),
            column=col,
            language=language,
            parse_strategy="gcc_compiler",
        )

    # Rustc
    rustc_error_matches = list(_RUSTC_ERROR.finditer(raw))
    rustc_loc_matches = list(_RUSTC_LOCATION.finditer(raw))
    if rustc_error_matches and rustc_loc_matches:
        em = rustc_error_matches[0]
        lm = rustc_loc_matches[-1]
        return ParsedError(
            error_type=f"RustError{em.group(1) or ''}",
            error_message=em.group(2).strip(),
            file_path=_normalize_path(lm.group(1), cwd),
            line_number=int(lm.group(2)),
            column=int(lm.group(3)),
            language="rust",
            parse_strategy="rustc_compiler",
        )

    return None


# ============================================================
# 4. 通用错误（Python Error 但不是完整 Traceback）
# ============================================================
# 有时候 Python 会直接输出一行错误，没有完整 Traceback：
# SyntaxError: invalid syntax (main.py, line 5)
#
# 或者类似 pytest 的简短错误：
# FAILED test_main.py::test_example - AssertionError: expected 5 but got 3

_PY_DIRECT_ERROR = re.compile(
    r"^\s*([A-Z][A-Za-z0-9_]*Error)\s*:\s*(.+?)\s*(?:\((.+?),\s*line\s+(\d+)\))?\s*$",
    re.MULTILINE,
)

# pytest 格式：FAILED test_main.py::test_example - AssertionError: expected 3 == 4
# 或：AssertionError: expected 3 == 4
# 或：test_main.py::test_example FAILED - AssertionError: expected 3 == 4
_PYTEST_ERROR = re.compile(
    r"-\s*([A-Z][A-Za-z0-9_]*Error)\s*:\s*(.+)$",
    re.MULTILINE,
)


def _parse_generic_python_error(raw: str, cwd: str) -> Optional[ParsedError]:
    """
    尝试解析 Python 直接输出的简短错误（没有完整 Traceback）。

    覆盖格式：
      - SyntaxError: invalid syntax (script.py, line 42)
      - FAILED test_main.py::test_example - AssertionError: expected 3 == 4  (pytest)

    Returns:
        ParsedError 或 None
    """
    if not raw:
        return None

    # 已经被 _parse_python_traceback 处理过的有完整 Traceback 的不走这里
    if _PY_TRACEBACK_HEADER.search(raw):
        return None

    # 先尝试 pytest 格式（FAILED ... - ErrorType: message）
    for m in _PYTEST_ERROR.finditer(raw):
        return ParsedError(
            error_type=m.group(1),
            error_message=m.group(2).strip(),
            language="python",
            parse_strategy="pytest_error",
        )

    # 再尝试通用 ErrorType: message 格式（可能附带 (file, line N)）
    for m in _PY_DIRECT_ERROR.finditer(raw):
        error_type = m.group(1)
        error_message = m.group(2).strip()
        file_path = None
        line_number = None
        if m.group(3):
            file_path = _normalize_path(m.group(3), cwd)
        if m.group(4):
            line_number = int(m.group(4))
        return ParsedError(
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            line_number=line_number,
            language="python",
            parse_strategy="python_direct",
        )

    return None


# ============================================================
# 5. 降级策略（Fallback）
# ============================================================
# 所有正则都没匹配上时，从最后 200 字符中找包含错误关键词的行。
# 解决 npm install 大量日志、编译警告刷屏等场景——只喂核心错误给自修复 Prompt。

_FALLBACK_MAX_CHARS = 200
# 错误关键词（不区分大小写）：fallback 会优先选包含这些关键词的行
# 按优先级排序（靠前的匹配优先返回）——核心错误通常出现在解决方案提示之前
_ERROR_KEYWORDS = ("error", "failed", "fatal", "exception", "traceback")


def _fallback_parse(raw: str) -> ParsedError:
    """
    降级解析：从最后 200 字符中找最核心的错误消息。

    策略：
      1. 截取最后 200 字符（避免 npm install 数千行日志）
      2. 按行拆分，优先选包含核心错误关键词的行（倒序遍历，返回第一个匹配）
      3. 没有任何关键词匹配则取第一行（最接近错误根源的）

    应对 S9 风险预警中"npm install 报错日志太长"的场景。
    """
    if not raw:
        return ParsedError(
            error_type="Unknown",
            error_message="(无错误输出)",
            parse_strategy="fallback",
        )

    # 截取最后 200 字符
    tail = raw.strip()[-_FALLBACK_MAX_CHARS:]

    # 按行拆分
    lines = [l.strip() for l in tail.split("\n") if l.strip()]
    if not lines:
        return ParsedError(
            error_type="Unknown",
            error_message=tail.strip() or "(空输出)",
            parse_strategy="fallback",
        )

    # 优先选包含核心错误关键词的行（倒序——最后一个关键词行通常是最核心的）
    for line in reversed(lines):
        line_lower = line.lower()
        for kw in _ERROR_KEYWORDS:
            # 简单子串匹配："error" 能匹配 "npm ERR!"（ERR 是 error 的缩写）
            if kw in line_lower:
                return ParsedError(
                    error_type="Unknown",
                    error_message=line[:200],
                    parse_strategy="fallback",
                )

    # 没有关键词匹配，取第一行（最靠近错误根源的）
    return ParsedError(
        error_type="Unknown",
        error_message=lines[0][:200],
        parse_strategy="fallback",
    )


# ============================================================
# 主入口：parse_error
# ============================================================

# 解析器尝试顺序：置信度从高到低
_PARSER_CHAIN = [
    ("python_traceback", _parse_python_traceback),
    ("js_stack", _parse_js_stack),
    ("compiler", _parse_compiler_error),
    ("python_direct", _parse_generic_python_error),
]


def parse_error(raw_stderr: str, cwd: str = "") -> ParsedError:
    """
    智能错误解析主入口。

    按优先级依次尝试各语言解析器，第一个命中即返回；
    全部未命中时走 fallback（最后 200 字符）。

    Args:
        raw_stderr: 命令执行的原始 stderr 输出（可能包含数千行日志）
        cwd:        命令执行时的工作目录，用于将绝对路径转换为相对路径

    Returns:
        ParsedError 结构化错误对象。
        永远不会返回 None——最不济也会走 fallback 返回最后 200 字符。

    示例：
        >>> raw = '''Traceback (most recent call last):
        ...   File "main.py", line 5, in <module>
        ...     x = y
        ... NameError: name 'y' is not defined
        ... '''
        >>> err = parse_error(raw, cwd="/workspace")
        >>> err.file_path
        'main.py'
        >>> err.line_number
        5
        >>> err.error_type
        'NameError'
        >>> err.error_message
        "name 'y' is not defined"
    """
    if not raw_stderr or not raw_stderr.strip():
        return _fallback_parse(raw_stderr or "")

    for strategy_name, parser in _PARSER_CHAIN:
        try:
            result = parser(raw_stderr, cwd)
            if result is not None:
                logger.debug(
                    f"[ErrorParser] 命中策略: {strategy_name}, "
                    f"type={result.error_type}, file={result.file_path}:{result.line_number}"
                )
                return result
        except Exception as e:
            # 解析器不应抛异常，某个解析器异常时继续尝试下一个
            logger.warning(
                f"[ErrorParser] {strategy_name} 解析异常: {e}", exc_info=True
            )
            continue

    # 全部未命中 → 降级策略
    result = _fallback_parse(raw_stderr)
    logger.debug(
        f"[ErrorParser] 所有策略未命中，降级到 fallback: msg={result.error_message[:80]}"
    )
    return result
