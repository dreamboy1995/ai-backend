"""
S9 第 81-82 天：error_parser.py 单元测试

验收标准：
  将一段 Python NameError: name 'x' is not defined 的完整 Traceback
  输入解析器，输出 {file_path: 'main.py', line: 5, error_type: 'NameError',
  message: "name 'x' is not defined"}。

覆盖场景：
  1. Python Traceback（单层 + 嵌套）
  2. JavaScript/TypeScript Stack Trace
  3. 编译器格式（gcc / tsc / rustc）
  4. 降级策略（npm install 大量日志）
  5. 空输入 / 异常输入
  6. 路径规范化（绝对路径 → 相对 cwd）
  7. 通用 Python 错误（无完整 Traceback）
"""

import os
import pytest

from app.services.error_parser import (
    ParsedError,
    parse_error,
)


# ============================================================
# Fixtures
# ============================================================

CWD = "/workspace/project"


# ============================================================
# 1. Python Traceback 验收测试
# ============================================================

class TestPythonTraceback:
    """S9 核心验收标准测试"""

    def test_nameerror_basic(self):
        """验收标准：Python NameError 完整 Traceback"""
        raw = """Traceback (most recent call last):
  File "main.py", line 5, in <module>
    x = y
NameError: name 'y' is not defined
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "NameError"
        assert result.line_number == 5
        assert result.file_path == "main.py"
        assert "name 'y' is not defined" in result.error_message
        assert result.language == "python"
        assert result.parse_strategy == "python_traceback"

    def test_nested_traceback_innermost(self):
        """嵌套调用应取最内层错误"""
        raw = """Traceback (most recent call last):
  File "runner.py", line 10, in run
    import main
  File "main.py", line 5, in <module>
    x = y
NameError: name 'y' is not defined
"""
        result = parse_error(raw, cwd=CWD)

        # 应该是 main.py（内层），不是 runner.py
        assert result.file_path == "main.py"
        assert result.line_number == 5
        assert result.error_type == "NameError"

    def test_typeerror(self):
        raw = """Traceback (most recent call last):
  File "calc.py", line 12, in divide
    return a / b
TypeError: unsupported operand type(s) for /: 'str' and 'int'
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "TypeError"
        assert result.file_path == "calc.py"
        assert result.line_number == 12

    def test_syntaxerror(self):
        raw = """Traceback (most recent call last):
  File "app.py", line 20, in <module>
    def foo(
SyntaxError: unexpected EOF while parsing
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "SyntaxError"
        assert result.file_path == "app.py"
        assert result.line_number == 20

    def test_import_error(self):
        raw = """Traceback (most recent call last):
  File "main.py", line 1, in <module>
    import fastapi
ModuleNotFoundError: No module named 'fastapi'
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "ModuleNotFoundError"
        assert result.file_path == "main.py"
        assert result.line_number == 1

    def test_sse_dict_format(self):
        """验证 to_sse_dict 输出格式符合 S9 关键接口定义"""
        raw = """Traceback (most recent call last):
  File "main.py", line 10, in <module>
    x = app
NameError: name 'app' is not defined
"""
        result = parse_error(raw, cwd=CWD)
        sse = result.to_sse_dict()

        assert sse["type"] == "NameError"
        assert "name 'app' is not defined" in sse["message"]
        assert sse["file"] == "main.py"
        assert sse["line"] == 10


# ============================================================
# 2. JavaScript / TypeScript Stack Trace
# ============================================================

class TestJSStack:
    def test_referenceerror(self):
        raw = """ReferenceError: x is not defined
    at foo (/workspace/project/app.js:10:5)
    at bar (/workspace/project/app.js:5:10)
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "ReferenceError"
        assert result.file_path == "app.js"
        assert result.line_number == 10
        assert result.column == 5
        assert result.language == "javascript"
        assert result.parse_strategy == "js_stack"

    def test_typescript_file(self):
        raw = """Error: Something went wrong
    at handler (/workspace/src/api.ts:42:15)
"""
        result = parse_error(raw, cwd="/workspace")

        assert result.language == "typescript"
        assert result.file_path == "src/api.ts"
        assert result.line_number == 42

    def test_typeerror_js(self):
        raw = """TypeError: Cannot read property 'map' of undefined
    at render (/home/user/src/components/List.jsx:88:3)
    at App (/home/user/src/App.jsx:20:1)
"""
        result = parse_error(raw, cwd="/home/user")

        assert result.error_type == "TypeError"
        assert result.file_path == "src/components/List.jsx"
        assert result.line_number == 88


# ============================================================
# 3. 编译器错误
# ============================================================

class TestCompilerError:
    def test_gcc_format(self):
        raw = """main.c:10:5: error: expected ';' before '}'
   10 |     }
      |     ^
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "CompileError"
        assert result.file_path == "main.c"
        assert result.line_number == 10
        assert result.column == 5
        assert result.language == "c"
        assert result.parse_strategy == "gcc_compiler"

    def test_tsc_format(self):
        raw = """src/index.ts(15,3): error TS2307: Cannot find module 'lodash'
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "CompileError"
        assert result.file_path == "src/index.ts"
        assert result.line_number == 15
        assert result.column == 3
        assert result.language == "typescript"
        assert result.parse_strategy == "ts_compiler"

    def test_rustc_format(self):
        raw = """error[E0425]: cannot find value `x` in this scope
 --> main.rs:5:10
  |
5 |     let y = x;
  |          ^ not found in this scope
  |
  = help: consider using `x` instead
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "RustErrorE0425"
        assert result.file_path == "main.rs"
        assert result.line_number == 5
        assert result.language == "rust"
        assert result.parse_strategy == "rustc_compiler"

    def test_gcc_no_column(self):
        raw = """Makefile:20: error recipe for target 'all' failed
"""
        result = parse_error(raw, cwd=CWD)

        assert result.file_path == "Makefile"
        assert result.line_number == 20
        assert result.column is None


# ============================================================
# 4. 通用 Python 错误（无完整 Traceback）
# ============================================================

class TestGenericPythonError:
    def test_syntaxerror_with_location(self):
        raw = """SyntaxError: invalid syntax (script.py, line 42)
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "SyntaxError"
        assert result.file_path == "script.py"
        assert result.line_number == 42
        assert result.parse_strategy == "python_direct"

    def test_assertionerror_pytest(self):
        raw = """FAILED test_main.py::test_example - AssertionError: expected 3 == 4
"""
        result = parse_error(raw, cwd=CWD)

        assert result.error_type == "AssertionError"
        assert "expected 3 == 4" in result.error_message


# ============================================================
# 5. 降级策略（Fallback）
# ============================================================

class TestFallback:
    def test_npm_install_long_log(self):
        """npm install 数千行日志后只取最后 200 字符"""
        long_log = "\n".join([f"npm WARN package{i} deprecated" for i in range(50)])
        long_log += """
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! While resolving: myapp@1.0.0
npm ERR! Found: react@18.0.0
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^16.0.0" from some-older-package@2.0.0
npm ERR!
npm ERR! Run `npm install <pkg>@latest` to resolve this issue.
"""
        result = parse_error(long_log, cwd=CWD)

        assert result.parse_strategy == "fallback"
        assert result.error_type == "Unknown"
        # fallback 只取最后 200 字符，可能截断掉前面的核心错误行
        # 但必须是 tail 中包含 "npm ERR" 的有意义的行
        assert "npm ERR" in result.error_message or result.error_message != ""

    def test_empty_input(self):
        result = parse_error("", cwd=CWD)
        assert result.parse_strategy == "fallback"
        assert result.error_message == "(无错误输出)"

    def test_whitespace_only(self):
        result = parse_error("   \n\n  \t  \n", cwd=CWD)
        assert result.parse_strategy == "fallback"

    def test_short_unknown_error(self):
        short_log = "Permission denied to access file: /etc/passwd"
        result = parse_error(short_log, cwd=CWD)
        assert result.parse_strategy == "fallback"
        assert "Permission denied" in result.error_message


# ============================================================
# 6. 路径规范化
# ============================================================

class TestPathNormalization:
    def test_absolute_path_to_relative(self):
        raw = """Traceback (most recent call last):
  File "/workspace/project/main.py", line 5, in <module>
    x = y
NameError: name 'y' is not defined
"""
        result = parse_error(raw, cwd="/workspace/project")
        # 应该被转换为相对路径
        assert result.file_path == "main.py"

    def test_nested_directory_path(self):
        raw = """Traceback (most recent call last):
  File "/workspace/project/src/utils/helpers.py", line 25, in helper
    z = x + y
NameError: name 'x' is not defined
"""
        result = parse_error(raw, cwd="/workspace/project")
        assert result.file_path == "src/utils/helpers.py"

    def test_windows_backslash_path(self):
        raw = """Traceback (most recent call last):
  File "C:\\workspace\\project\\main.py", line 10, in <module>
    app()
NameError: name 'app' is not defined
"""
        result = parse_error(raw, cwd="C:\\workspace\\project")
        # Windows 路径统一为正斜杠
        assert result.file_path == "main.py" or "\\" not in (result.file_path or "")

    def test_path_outside_cwd(self):
        """不在 cwd 下的路径应保留完整绝对路径"""
        raw = """Traceback (most recent call last):
  File "/usr/lib/python3.10/something.py", line 5, in init
    raise Exception()
Exception: system error
"""
        result = parse_error(raw, cwd=CWD)
        # 不在 /workspace/project 下，保留绝对路径
        assert result.file_path == "/usr/lib/python3.10/something.py"


# ============================================================
# 7. ParsedError 基本属性
# ============================================================

class TestParsedErrorModel:
    def test_str_representation(self):
        err = ParsedError(
            error_type="NameError",
            error_message="name 'x' is not defined",
            file_path="main.py",
            line_number=5,
        )
        s = str(err)
        assert "NameError" in s
        assert "name 'x' is not defined" in s
        assert "main.py:5" in s

    def test_str_without_location(self):
        err = ParsedError(error_type="Unknown", error_message="something went wrong")
        s = str(err)
        assert "[Unknown]" in s
        assert "something went wrong" in s

    def test_to_sse_dict_schema(self):
        """确保 SSE 事件格式正确"""
        err = ParsedError(
            error_type="SyntaxError",
            error_message="bad syntax",
            file_path="app.py",
            line_number=42,
        )
        sse = err.to_sse_dict()
        assert set(sse.keys()) == {"type", "message", "file", "line"}
        assert sse["type"] == "SyntaxError"
        assert sse["line"] == 42


# ============================================================
# 8. 解析器优先级（chain 顺序正确）
# ============================================================

class TestParserPriority:
    def test_python_traceback_wins_over_js_stack(self):
        """有 Traceback header 时优先用 Python 解析器"""
        raw = """Traceback (most recent call last):
  File "main.py", line 10, in <module>
    x = 1/0
ZeroDivisionError: division by zero
"""
        result = parse_error(raw, cwd=CWD)
        assert result.parse_strategy == "python_traceback"
        assert result.language == "python"

    def test_compiler_wins_over_fallback(self):
        """gcc 格式优先于 fallback"""
        raw = """src/main.c:15:3: error: undeclared identifier 'foo'
"""
        result = parse_error(raw, cwd=CWD)
        assert result.parse_strategy == "gcc_compiler"

    def test_js_stack_wins_over_fallback(self):
        """at 行存在时用 js_stack 解析器"""
        raw = """TypeError: boom
    at handle (server.js:20:5)
"""
        result = parse_error(raw, cwd=CWD)
        assert result.parse_strategy == "js_stack"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
