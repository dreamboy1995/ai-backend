"""
S8 第 71-72 天：工具注册中心单元测试

验收标准（来自 Sprint_8.md 第 71-72 天）：
  "编写单元测试：调用 execute_tool(ToolCall(tool_name='read_file', arguments={'file_path': 'test.txt'}))，
   能正确返回文件内容（或报错提示文件不存在）。"

覆盖场景：
  - execute_tool 路由：read_file 成功 / 文件不存在 / 行号区间
  - 路径逃逸拦截：../../../etc/passwd 被拒绝
  - write_file 确认流程：返回 requires_confirmation + Diff 预览
  - run_command 危险命令黑名单：rm -rf / 直接拦截
  - run_command 普通命令：需确认，确认后真正执行
  - confirm_tool 流程：allow 执行 / deny 拒绝 / 凭证无效
  - 未知工具报错
  - 审计日志写入
  - MCPToolExecutor 桥接 S7 抽象类
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.models.tool import ToolCall
from app.services.tool_executor import MCPToolExecutor, ToolExecutionError
from app.services.tool_registry import (
    confirm_tool,
    execute_tool,
    get_registered_tools,
)


# ============================================================
# 测试夹具
# ============================================================

@pytest.fixture
def tmp_workspace(tmp_path):
    """创建临时工作区目录"""
    return str(tmp_path)


def _make_tool_call(tool_name, **arguments):
    return ToolCall(tool_name=tool_name, arguments=arguments)


# ============================================================
# execute_tool 路由测试
# ============================================================

def test_registered_tools_contains_all_five():
    """注册表包含 S8 定义的 5 个标准工具"""
    tools = get_registered_tools()
    assert set(tools) == {"read_file", "write_file", "run_command", "grep_search", "git_commit"}


def test_execute_unknown_tool_returns_error(tmp_workspace):
    """未知工具返回 success=False + 错误提示"""
    tool_call = ToolCall(tool_name="read_file", arguments={"file_path": "x.txt"})
    # 用 patch 模拟一个不存在的工具名
    with patch("app.services.tool_registry._TOOL_REGISTRY", {}):
        result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))
    assert result.success is False
    assert "未知工具" in result.error


def test_execute_read_file_success(tmp_workspace):
    """read_file 成功读取文件内容（验收标准核心用例）"""
    # 准备测试文件
    test_file = Path(tmp_workspace) / "test.txt"
    test_file.write_text("hello\nworld\n", encoding="utf-8")

    tool_call = _make_tool_call("read_file", file_path="test.txt")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    assert result.output == "hello\nworld\n"
    assert result.error is None
    assert result.requires_confirmation is False


def test_execute_read_file_not_found(tmp_workspace):
    """read_file 文件不存在时返回友好错误"""
    tool_call = _make_tool_call("read_file", file_path="nonexistent.txt")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is False
    assert "不存在" in result.error


def test_execute_read_file_with_line_range(tmp_workspace):
    """read_file 支持行号区间过滤"""
    test_file = Path(tmp_workspace) / "lines.txt"
    lines = [f"line {i}\n" for i in range(1, 11)]
    test_file.write_text("".join(lines), encoding="utf-8")

    tool_call = _make_tool_call("read_file", file_path="lines.txt", start_line=3, end_line=5)
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    assert result.output == "line 3\nline 4\nline 5\n"


def test_execute_read_file_path_traversal_blocked(tmp_workspace):
    """路径逃逸拦截：../../../etc/passwd 被拒绝"""
    tool_call = _make_tool_call("read_file", file_path="../../../etc/passwd")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is False
    assert "逃逸" in result.error or "非法" in result.error


def test_execute_read_file_missing_file_path(tmp_workspace):
    """read_file 缺少 file_path 参数返回错误"""
    tool_call = _make_tool_call("read_file")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is False
    assert "file_path" in result.error


# ============================================================
# write_file 确认流程测试
# ============================================================

def test_execute_write_file_requires_confirmation(tmp_workspace):
    """write_file 返回 requires_confirmation=True + Diff 预览"""
    tool_call = _make_tool_call(
        "write_file", file_path="new.py", content="print('hi')\n"
    )
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    assert result.requires_confirmation is True
    assert result.confirmation_id is not None
    assert result.confirmation_prompt is not None
    # 新增文件的 Diff 应包含 +++ 标记
    assert "+++" in result.output or "新增" in result.output or "新内容" in result.output

    # 确认前文件不应被创建
    assert not (Path(tmp_workspace) / "new.py").exists()


def test_execute_write_file_missing_content(tmp_workspace):
    """write_file 缺少 content 参数返回错误"""
    tool_call = _make_tool_call("write_file", file_path="x.py")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is False
    assert "content" in result.error


# ============================================================
# run_command 测试
# ============================================================

def test_execute_run_command_danger_blocked(tmp_workspace):
    """危险命令 rm -rf / 直接拦截，无需确认"""
    tool_call = _make_tool_call("run_command", cmd="rm -rf /")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is False
    assert "拦截" in result.error
    assert result.requires_confirmation is False


def test_execute_run_command_sudo_blocked(tmp_workspace):
    """sudo 命令被黑名单拦截"""
    tool_call = _make_tool_call("run_command", cmd="sudo apt install")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is False
    assert "拦截" in result.error


def test_execute_run_command_normal_requires_confirmation(tmp_workspace):
    """普通命令需用户确认"""
    tool_call = _make_tool_call("run_command", cmd="echo hello")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    assert result.requires_confirmation is True
    assert result.confirmation_id is not None


# ============================================================
# confirm_tool 流程测试
# ============================================================

def test_confirm_tool_allow_writes_file(tmp_workspace):
    """confirm_tool allow 后 write_file 真正写入文件"""
    # 第一步：execute 生成 confirmation_id
    tool_call = _make_tool_call(
        "write_file", file_path="confirmed.py", content="print('confirmed')\n"
    )
    exec_result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))
    assert exec_result.requires_confirmation is True
    confirmation_id = exec_result.confirmation_id

    # 确认前文件不存在
    target = Path(tmp_workspace) / "confirmed.py"
    assert not target.exists()

    # 第二步：confirm allow
    confirm_result = asyncio.run(
        confirm_tool(confirmation_id, "allow", "test-session")
    )

    assert confirm_result.success is True
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "print('confirmed')\n"


def test_confirm_tool_deny_does_not_write(tmp_workspace):
    """confirm_tool deny 不写入文件"""
    tool_call = _make_tool_call(
        "write_file", file_path="denied.py", content="print('denied')\n"
    )
    exec_result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))
    confirmation_id = exec_result.confirmation_id

    confirm_result = asyncio.run(
        confirm_tool(confirmation_id, "deny", "test-session")
    )

    assert confirm_result.success is False
    assert "拒绝" in confirm_result.error
    assert not (Path(tmp_workspace) / "denied.py").exists()


def test_confirm_tool_invalid_id(tmp_workspace):
    """无效的 confirmation_id 返回错误"""
    result = asyncio.run(confirm_tool("nonexistent-id", "allow", "test-session"))
    assert result.success is False
    assert "不存在" in result.error or "过期" in result.error


def test_confirm_tool_one_time_use(tmp_workspace):
    """confirmation_id 一次性：allow 后再次使用返回错误"""
    tool_call = _make_tool_call(
        "write_file", file_path="once.py", content="x=1\n"
    )
    exec_result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))
    confirmation_id = exec_result.confirmation_id

    # 第一次 allow 成功
    first = asyncio.run(confirm_tool(confirmation_id, "allow", "test-session"))
    assert first.success is True

    # 第二次使用同凭证失败（一次性）
    second = asyncio.run(confirm_tool(confirmation_id, "allow", "test-session"))
    assert second.success is False


def test_confirm_tool_allow_run_command(tmp_workspace):
    """confirm_tool allow 后 run_command 真正执行命令"""
    tool_call = _make_tool_call("run_command", cmd="echo hello_world")
    exec_result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))
    confirmation_id = exec_result.confirmation_id

    confirm_result = asyncio.run(
        confirm_tool(confirmation_id, "allow", "test-session")
    )

    assert confirm_result.success is True
    assert "hello_world" in confirm_result.output


# ============================================================
# grep_search 测试
# ============================================================

def test_execute_grep_search_finds_matches(tmp_workspace):
    """grep_search 能找到匹配的代码行"""
    # 准备测试文件
    (Path(tmp_workspace) / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (Path(tmp_workspace) / "b.py").write_text("x = 1\n", encoding="utf-8")

    tool_call = _make_tool_call("grep_search", pattern="def foo")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    assert "a.py" in result.output
    assert "def foo" in result.output


def test_grep_search_context_lines_format(tmp_workspace):
    """grep_search 上下文行输出格式：匹配行用 :，上下文行用 -"""
    content = "\n".join([
        "line 1",
        "line 2",
        "def foo():",
        "    pass",
        "line 5",
        "line 6",
    ]) + "\n"
    (Path(tmp_workspace) / "ctx.py").write_text(content, encoding="utf-8")

    tool_call = _make_tool_call(
        "grep_search", pattern="def foo", context_lines=1
    )
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    lines = result.output.splitlines()
    # 匹配行（第3行）用冒号分隔：path:lineno:content
    match_line = next(l for l in lines if "def foo" in l)
    assert "ctx.py:3:def foo" in match_line
    # 上下文行用减号分隔：path-lineno-content
    context_lines = [l for l in lines if l.startswith("ctx.py-")]
    assert any("ctx.py-2-line 2" in l for l in context_lines)
    assert any("ctx.py-4-    pass" in l for l in context_lines)


def test_grep_search_zero_context_lines(tmp_workspace):
    """context_lines=0 时只返回匹配行，无上下文"""
    content = "\n".join(["a", "b", "TARGET", "c", "d"]) + "\n"
    (Path(tmp_workspace) / "zero.py").write_text(content, encoding="utf-8")

    tool_call = _make_tool_call(
        "grep_search", pattern="TARGET", context_lines=0
    )
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    lines = result.output.splitlines()
    assert len(lines) == 1
    assert "zero.py:3:TARGET" in lines[0]


def test_grep_search_skip_node_modules(tmp_workspace):
    """grep_search 跳过 node_modules 等大型依赖目录"""
    # 在 node_modules 中放置匹配内容
    nm_dir = Path(tmp_workspace) / "node_modules" / "pkg"
    nm_dir.mkdir(parents=True)
    (nm_dir / "index.js").write_text("const SECRET = 1;\n", encoding="utf-8")
    # 在正常目录中放置匹配内容
    (Path(tmp_workspace) / "main.js").write_text("const SECRET = 2;\n", encoding="utf-8")

    tool_call = _make_tool_call("grep_search", pattern="SECRET")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    # node_modules 中的内容不应出现
    assert "node_modules" not in result.output
    # 正常目录中的内容应出现
    assert "main.js" in result.output


def test_grep_search_max_results_limit(tmp_workspace):
    """grep_search 受 max_results 限制"""
    lines = [f"item {i}\n" for i in range(50)]
    (Path(tmp_workspace) / "many.py").write_text("".join(lines), encoding="utf-8")

    tool_call = _make_tool_call(
        "grep_search", pattern="item", max_results=3, context_lines=0
    )
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    # context_lines=0 时，每个匹配一行，最多 3 行
    assert len(result.output.splitlines()) <= 3


def test_grep_search_context_lines_clamped(tmp_workspace):
    """context_lines 超过上限 10 时被截断为 10"""
    content = "\n".join([f"L{i}" for i in range(30)]) + "\n"
    (Path(tmp_workspace) / "clamp.py").write_text(content, encoding="utf-8")

    tool_call = _make_tool_call(
        "grep_search", pattern="L15", context_lines=999
    )
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is True
    # 匹配行 L15（第16行）+ 前后各10行上下文 = 最多21行
    lines = result.output.splitlines()
    assert len(lines) <= 21


def test_grep_search_invalid_regex(tmp_workspace):
    """grep_search 无效正则表达式返回错误"""
    (Path(tmp_workspace) / "x.py").write_text("hello\n", encoding="utf-8")

    tool_call = _make_tool_call("grep_search", pattern="[invalid")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    assert result.success is False
    assert "无效" in result.error or "正则" in result.error


# ============================================================
# git_commit 测试（无变更时返回错误）
# ============================================================

def test_execute_git_commit_no_changes(tmp_workspace):
    """git_commit 在无变更时返回友好错误"""
    tool_call = _make_tool_call("git_commit", message="test commit")
    result = asyncio.run(execute_tool(tool_call, tmp_workspace, "test-session"))

    # 不在 git 仓库中或无变更，应返回错误
    assert result.success is False


# ============================================================
# 审计日志测试
# ============================================================

def test_audit_log_written(tmp_workspace, tmp_path):
    """工具调用后审计日志被写入"""
    import app.services.tool_registry as registry_mod

    # 临时修改审计日志路径
    log_path = tmp_path / "audit.log"
    with patch.object(registry_mod.settings, "TOOL_AUDIT_LOG_PATH", str(log_path)):
        # 重新创建 pending store（单例可能已初始化，这里不影响审计日志）
        tool_call = _make_tool_call("read_file", file_path="nonexistent.txt")
        asyncio.run(execute_tool(tool_call, tmp_workspace, "audit-session"))

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    entry = json.loads(lines[-1])
    assert entry["tool_name"] == "read_file"
    assert entry["session_id"] == "audit-session"
    assert entry["success"] is False


# ============================================================
# MCPToolExecutor 桥接测试
# ============================================================

def test_mcp_executor_read_file(tmp_workspace):
    """MCPToolExecutor 调用 read_file 返回 ToolResult，output 包含文件内容"""
    test_file = Path(tmp_workspace) / "bridge.txt"
    test_file.write_text("bridge content", encoding="utf-8")

    executor = MCPToolExecutor(workspace_root=tmp_workspace, session_id="bridge-test")
    result = asyncio.run(executor.execute("read_file", {"file_path": "bridge.txt"}))

    assert result.success is True
    assert result.output == "bridge content"


def test_mcp_executor_failed_returns_false(tmp_workspace):
    """MCPToolExecutor 工具失败时返回 ToolResult(success=False)，不再抛异常"""
    executor = MCPToolExecutor(workspace_root=tmp_workspace, session_id="bridge-test")

    result = asyncio.run(executor.execute("read_file", {"file_path": "no.txt"}))
    assert result.success is False
    assert result.error is not None


def test_mcp_executor_write_file_confirmation(tmp_workspace):
    """MCPToolExecutor 对 write_file 返回 requires_confirmation=True"""
    executor = MCPToolExecutor(workspace_root=tmp_workspace, session_id="bridge-test")
    result = asyncio.run(executor.execute("write_file", {"file_path": "x.py", "content": "y"}))

    assert result.requires_confirmation is True
    assert result.confirmation_prompt is not None
    assert result.confirmation_id is not None
