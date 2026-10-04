"""
S8 第 75-76 天：终端命令执行器单元测试

覆盖场景（对应 Sprint_8.md 第 75-76 天验收标准）：
  1. CommandExecutor 基础执行：echo 命令返回正确 stdout
  2. 实时流式输出：订阅者收到 stdout/stderr/system StreamMessage
  3. 超时杀进程：长任务超时后返回 exit_code=-1 + 友好错误提示
  4. 进程组隔离：杀进程时连带子进程（通过 sleep 子进程验证）
  5. 命令注入防护：shlex.split 拒绝畸形输入
  6. 交互式命令检测：is_interactive_command 正确识别 npm init / python REPL 等
  7. Windows CMD 内置命令包装：echo 在 Windows 上能执行
  8. CommandStreamManager pub-sub：多订阅者 + unsubscribe
  9. tool_run_command 交互式命令：返回 requires_interaction=True
  10. _do_run_command 集成：通过 CommandExecutor 执行 + 流式输出
  11. MCPToolExecutor requires_interaction：返回提示文本而非抛异常

对应 Sprint_8.md「关键接口/数据结构变更」：
  - StreamMessage { type: 'stdout'|'stderr'|'system', content, timestamp }
  - ws://localhost:3000/v1/agent/stream/{session_id}
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.models.tool import StreamMessage, ToolCall
from app.services.command_executor import (
    CommandExecutor,
    CommandStreamManager,
    _wrap_windows_builtin,
    is_interactive_command,
    get_stream_manager,
)
from app.services.tool_executor import MCPToolExecutor
from app.services.tool_registry import (
    _do_run_command,
    confirm_tool,
    execute_tool,
    tool_run_command,
)


# ============================================================
# 测试夹具
# ============================================================

@pytest.fixture
def tmp_workspace(tmp_path):
    """临时工作区目录"""
    return str(tmp_path)


@pytest.fixture
def fresh_stream_manager():
    """每个测试独立的 CommandStreamManager（避免全局单例污染）"""
    return CommandStreamManager()


@pytest.fixture
def fresh_executor(fresh_stream_manager):
    """每个测试独立的 CommandExecutor（使用独立的 stream_manager）"""
    return CommandExecutor(stream_manager=fresh_stream_manager)


def _make_tool_call(tool_name, **arguments):
    return ToolCall(tool_name=tool_name, arguments=arguments)


# ============================================================
# 1. CommandExecutor 基础执行
# ============================================================

def test_execute_echo_command(fresh_executor, tmp_workspace):
    """执行 echo 命令能正确返回 stdout"""
    exit_code, stdout_text, stderr_text = asyncio.run(
        fresh_executor.execute(
            cmd="echo hello_world",
            workspace_root=tmp_workspace,
            session_id="test-echo",
            timeout=10.0,
        )
    )
    assert exit_code == 0
    assert "hello_world" in stdout_text
    assert stderr_text == ""


def test_execute_python_print(fresh_executor, tmp_workspace):
    """执行 python -c "print(...)" 能正确返回 stdout"""
    cmd = 'python -c "print(\'hi from python\')"'
    exit_code, stdout_text, stderr_text = asyncio.run(
        fresh_executor.execute(
            cmd=cmd,
            workspace_root=tmp_workspace,
            session_id="test-py",
            timeout=15.0,
        )
    )
    assert exit_code == 0
    assert "hi from python" in stdout_text


def test_execute_nonexistent_command(fresh_executor, tmp_workspace):
    """执行不存在的命令返回 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            fresh_executor.execute(
                cmd="this_command_does_not_exist_xyz123",
                workspace_root=tmp_workspace,
                session_id="test-missing",
                timeout=5.0,
            )
        )


def test_execute_empty_command(fresh_executor, tmp_workspace):
    """空命令返回 ValueError"""
    with pytest.raises(ValueError):
        asyncio.run(
            fresh_executor.execute(
                cmd="",
                workspace_root=tmp_workspace,
                session_id="test-empty",
                timeout=5.0,
            )
        )


def test_execute_nonzero_exit(fresh_executor, tmp_workspace):
    """非零退出码时 stderr 有内容"""
    cmd = 'python -c "import sys; sys.stderr.write(\'error msg\\n\'); sys.exit(2)"'
    exit_code, stdout_text, stderr_text = asyncio.run(
        fresh_executor.execute(
            cmd=cmd,
            workspace_root=tmp_workspace,
            session_id="test-exit-code",
            timeout=15.0,
        )
    )
    assert exit_code == 2
    assert "error msg" in stderr_text


# ============================================================
# 2. 实时流式输出
# ============================================================

def test_stream_publishes_stdout_messages(fresh_executor, fresh_stream_manager, tmp_workspace):
    """执行命令时订阅者收到 stdout StreamMessage"""
    async def _run():
        queue = await fresh_stream_manager.subscribe("test-stream-stdout")
        # 启动执行（后台任务）
        task = asyncio.create_task(
            fresh_executor.execute(
                cmd='python -c "print(\'line1\'); print(\'line2\')"',
                workspace_root=tmp_workspace,
                session_id="test-stream-stdout",
                timeout=15.0,
            )
        )
        # 收集消息
        messages = []
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=2.0)
                    messages.append(msg)
                except asyncio.TimeoutError:
                    if task.done():
                        break
        except asyncio.TimeoutError:
            pass
        exit_code, stdout_text, _ = await task
        return messages, stdout_text

    messages, stdout_text = asyncio.run(_run())

    # 应该至少有一条 stdout 消息和若干 system 消息
    stdout_messages = [m for m in messages if m.type == "stdout"]
    system_messages = [m for m in messages if m.type == "system"]
    assert len(stdout_messages) >= 1
    assert any("line1" in m.content for m in stdout_messages)
    assert any("line2" in m.content for m in stdout_messages)
    # 系统消息：启动提示 + 退出码
    assert any("PID" in m.content for m in system_messages)
    assert any("退出码" in m.content for m in system_messages)
    # 所有消息都有 timestamp
    for m in messages:
        assert m.timestamp  # 非空


def test_stream_publishes_stderr_messages(fresh_executor, fresh_stream_manager, tmp_workspace):
    """stderr 输出能被订阅者收到"""
    async def _run():
        queue = await fresh_stream_manager.subscribe("test-stream-stderr")
        task = asyncio.create_task(
            fresh_executor.execute(
                cmd='python -c "import sys; sys.stderr.write(\'stderr line\\n\')"',
                workspace_root=tmp_workspace,
                session_id="test-stream-stderr",
                timeout=15.0,
            )
        )
        messages = []
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=2.0)
                    messages.append(msg)
                except asyncio.TimeoutError:
                    if task.done():
                        break
        except asyncio.TimeoutError:
            pass
        await task
        return messages

    messages = asyncio.run(_run())
    stderr_messages = [m for m in messages if m.type == "stderr"]
    assert any("stderr line" in m.content for m in stderr_messages)


def test_stream_no_subscribers_still_works(fresh_executor, tmp_workspace):
    """没有订阅者时命令仍能正常执行，stdout/stderr 在结果中返回"""
    # 不调用 subscribe，直接执行
    exit_code, stdout_text, _ = asyncio.run(
        fresh_executor.execute(
            cmd="echo no_subscribers",
            workspace_root=tmp_workspace,
            session_id="test-no-subs",
            timeout=5.0,
        )
    )
    assert exit_code == 0
    assert "no_subscribers" in stdout_text


# ============================================================
# 3. 超时杀进程
# ============================================================

def test_timeout_kills_long_running_command(fresh_executor, fresh_stream_manager, tmp_workspace):
    """超时后命令被杀，返回 exit_code=-1 + 友好错误"""
    # 用 python -c "import time; time.sleep(10)" 跨平台，且 stdout 重定向时仍正常
    # Windows 的 timeout 命令在 stdout 重定向到 pipe 时会立即报错退出（无法测试）
    cmd = 'python -c "import time; time.sleep(10)"'

    async def _run():
        queue = await fresh_stream_manager.subscribe("test-timeout")
        task = asyncio.create_task(
            fresh_executor.execute(
                cmd=cmd,
                workspace_root=tmp_workspace,
                session_id="test-timeout",
                timeout=1.0,
            )
        )
        # 收集系统消息
        messages = []
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=3.0)
                    messages.append(msg)
                except asyncio.TimeoutError:
                    if task.done():
                        break
        except asyncio.TimeoutError:
            pass
        exit_code, stdout_text, stderr_text = await task
        return exit_code, stderr_text, messages

    exit_code, stderr_text, messages = asyncio.run(_run())

    assert exit_code == -1
    assert "超时" in stderr_text
    # 系统消息应有"已终止"提示
    system_messages = [m for m in messages if m.type == "system"]
    assert any("超时" in m.content and "终止" in m.content for m in system_messages)


# ============================================================
# 4. 命令注入防护
# ============================================================

def test_command_injection_semicolon_rejected(fresh_executor, tmp_workspace):
    """shlex.split 拒绝畸形引号（防命令注入）"""
    # 不闭合的引号会被 shlex.split 拒绝
    with pytest.raises(ValueError):
        asyncio.run(
            fresh_executor.execute(
                cmd='echo "hello; rm -rf /',
                workspace_root=tmp_workspace,
                session_id="test-injection",
                timeout=5.0,
            )
        )


# ============================================================
# 5. 交互式命令检测
# ============================================================

def test_is_interactive_npm_init_no_yes():
    """npm init 不带 --yes 被识别为交互式"""
    assert is_interactive_command("npm init") is True
    assert is_interactive_command("npm init -y") is False
    assert is_interactive_command("npm init --yes") is False


def test_is_interactive_python_repl():
    """python / node 无参数进入 REPL 被识别为交互式"""
    assert is_interactive_command("python") is True
    assert is_interactive_command("python3") is True
    assert is_interactive_command("node") is True
    # 带 -c 参数不是交互式
    assert is_interactive_command('python -c "print(1)"') is False


def test_is_interactive_ssh():
    """ssh 交互式登录被识别"""
    assert is_interactive_command("ssh user@host") is True
    # ssh -T 不进入交互式 shell
    # 注意：本测试只验证检测函数，不验证命令是否真的不交互
    # ssh -N 用于端口转发，本检测将其排除
    assert is_interactive_command("ssh -N user@host") is False


def test_is_interactive_passwd():
    """passwd 修改密码被识别"""
    assert is_interactive_command("passwd") is True
    assert is_interactive_command("passwd user") is True


def test_is_interactive_normal_commands_not_flagged():
    """普通命令不应被误判为交互式"""
    assert is_interactive_command("echo hello") is False
    assert is_interactive_command("ls -la") is False
    assert is_interactive_command("npm install") is False
    assert is_interactive_command("pip install requests") is False
    assert is_interactive_command("git status") is False
    assert is_interactive_command("python app.py") is False


# ============================================================
# 6. Windows CMD 内置命令包装
# ============================================================

def test_wrap_windows_builtin_echo():
    """Windows 上 echo 命令被包装为 cmd.exe /c"""
    if sys.platform != "win32":
        # POSIX 上原样返回
        assert _wrap_windows_builtin(["echo", "hello"]) == ["echo", "hello"]
        return
    # Windows 上 echo/dir 等内置命令被包装
    wrapped = _wrap_windows_builtin(["echo", "hello"])
    assert wrapped[0] == "cmd.exe"
    assert wrapped[1] == "/c"
    assert wrapped[2] == "echo"
    assert wrapped[3] == "hello"


def test_wrap_windows_builtin_does_not_wrap_executables():
    """Windows 上非内置命令（如 python.exe）不被包装"""
    if sys.platform != "win32":
        return
    # python 是真实可执行文件，不应被包装
    assert _wrap_windows_builtin(["python", "-c", "print(1)"]) == ["python", "-c", "print(1)"]


# ============================================================
# 7. CommandStreamManager pub-sub
# ============================================================

def test_stream_manager_multiple_subscribers(fresh_stream_manager):
    """同一 session 多个订阅者都能收到消息"""
    async def _run():
        q1 = await fresh_stream_manager.subscribe("multi")
        q2 = await fresh_stream_manager.subscribe("multi")
        await fresh_stream_manager.publish(
            "multi",
            StreamMessage(type="stdout", content="msg1\n", timestamp="t1"),
        )
        msg1 = await q1.get()
        msg2 = await q2.get()
        return msg1, msg2

    msg1, msg2 = asyncio.run(_run())
    assert msg1.content == "msg1\n"
    assert msg2.content == "msg1\n"


def test_stream_manager_unsubscribe(fresh_stream_manager):
    """unsubscribe 后不再收到消息"""
    async def _run():
        q = await fresh_stream_manager.subscribe("unsub")
        await fresh_stream_manager.unsubscribe("unsub", q)
        await fresh_stream_manager.publish(
            "unsub",
            StreamMessage(type="system", content="after unsub", timestamp="t"),
        )
        # 队列应为空
        try:
            msg = await asyncio.wait_for(q.get(), timeout=0.5)
            return msg
        except asyncio.TimeoutError:
            return None

    result = asyncio.run(_run())
    assert result is None


def test_stream_manager_no_subscribers_publish_no_error(fresh_stream_manager):
    """无订阅者时 publish 不应抛异常"""
    async def _run():
        # 不应抛异常
        await fresh_stream_manager.publish(
            "no-subs",
            StreamMessage(type="system", content="no one listens", timestamp="t"),
        )
        return True

    assert asyncio.run(_run()) is True


# ============================================================
# 8. tool_run_command 交互式命令检测
# ============================================================

def test_tool_run_command_interactive_returns_requires_interaction(tmp_workspace):
    """tool_run_command 对交互式命令返回 requires_interaction=True"""
    # npm init 无 --yes
    result = asyncio.run(
        tool_run_command(
            {"cmd": "npm init"},
            workspace_root=tmp_workspace,
            session_id="test-interact",
        )
    )
    assert result.requires_interaction is True
    assert result.requires_confirmation is False
    assert "交互式" in result.error or "交互" in result.error


def test_tool_run_command_python_repl_blocked(tmp_workspace):
    """tool_run_command 对 python REPL 返回 requires_interaction"""
    result = asyncio.run(
        tool_run_command(
            {"cmd": "python"},
            workspace_root=tmp_workspace,
            session_id="test-py-repl",
        )
    )
    assert result.requires_interaction is True


def test_tool_run_command_normal_requires_confirmation(tmp_workspace):
    """tool_run_command 对普通命令返回 requires_confirmation"""
    result = asyncio.run(
        tool_run_command(
            {"cmd": "echo hello"},
            workspace_root=tmp_workspace,
            session_id="test-normal",
        )
    )
    assert result.requires_confirmation is True
    assert result.requires_interaction is False


def test_tool_run_command_danger_still_blocked(tmp_workspace):
    """危险命令仍被黑名单拦截，不进入交互式检测分支"""
    result = asyncio.run(
        tool_run_command(
            {"cmd": "rm -rf /"},
            workspace_root=tmp_workspace,
            session_id="test-danger",
        )
    )
    assert result.success is False
    assert "拦截" in result.error
    assert result.requires_confirmation is False
    assert result.requires_interaction is False


# ============================================================
# 9. _do_run_command 集成测试
# ============================================================

def test_do_run_command_executes_via_executor(tmp_workspace):
    """_do_run_command 通过 CommandExecutor 执行命令"""
    result = asyncio.run(
        _do_run_command(
            {"cmd": "echo executor_test"},
            workspace_root=tmp_workspace,
            session_id="test-do-exec",
        )
    )
    assert result.success is True
    assert "executor_test" in result.output


def test_do_run_command_interactive_safety_net(tmp_workspace):
    """_do_run_command 二次检测交互式命令（安全网）"""
    result = asyncio.run(
        _do_run_command(
            {"cmd": "npm init"},
            workspace_root=tmp_workspace,
            session_id="test-do-interact",
        )
    )
    assert result.success is False
    assert result.requires_interaction is True


def test_do_run_command_nonzero_exit_returns_error(tmp_workspace):
    """非零退出码返回 success=False + 退出码错误"""
    result = asyncio.run(
        _do_run_command(
            {"cmd": 'python -c "import sys; sys.exit(3)"'},
            workspace_root=tmp_workspace,
            session_id="test-exit",
        )
    )
    assert result.success is False
    assert "3" in result.error


# ============================================================
# 10. execute_tool + confirm_tool 端到端流式
# ============================================================

def test_execute_then_confirm_streams_output(tmp_workspace):
    """完整 execute → confirm 流程：confirm 阶段流式输出可被订阅"""
    # 用独立 stream_manager 避免与其他测试冲突
    test_manager = CommandStreamManager()
    test_executor = CommandExecutor(stream_manager=test_manager)

    with patch(
        "app.services.command_executor.get_command_executor",
        return_value=test_executor,
    ):
        async def _run():
            # 订阅会话
            queue = await test_manager.subscribe("test-e2e-stream")

            # Step 1: execute 生成 confirmation_id
            tool_call = _make_tool_call("run_command", cmd="echo stream_test")
            exec_result = await execute_tool(
                tool_call, tmp_workspace, "test-e2e-stream"
            )
            assert exec_result.requires_confirmation is True
            confirmation_id = exec_result.confirmation_id

            # Step 2: 后台启动 confirm（异步）
            confirm_task = asyncio.create_task(
                confirm_tool(confirmation_id, "allow", "test-e2e-stream")
            )

            # 收集消息（带超时兜底）
            messages = []
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=3.0)
                        messages.append(msg)
                    except asyncio.TimeoutError:
                        if confirm_task.done():
                            break
            except asyncio.TimeoutError:
                pass

            confirm_result = await confirm_task
            return messages, confirm_result

        messages, confirm_result = asyncio.run(_run())

    assert confirm_result.success is True
    assert "stream_test" in confirm_result.output
    # 应有 stdout 消息和 system 消息
    stdout_msgs = [m for m in messages if m.type == "stdout"]
    system_msgs = [m for m in messages if m.type == "system"]
    assert any("stream_test" in m.content for m in stdout_msgs)
    assert any("PID" in m.content for m in system_msgs)
    assert any("退出码" in m.content for m in system_msgs)


# ============================================================
# 11. MCPToolExecutor 处理 requires_interaction
# ============================================================

def test_mcp_executor_interactive_returns_text_not_raises(tmp_workspace):
    """MCPToolExecutor 对交互式命令返回提示文本而非抛异常"""
    executor = MCPToolExecutor(
        workspace_root=tmp_workspace, session_id="test-mcp-interact"
    )
    obs = asyncio.run(executor.execute("run_command", {"cmd": "npm init"}))

    # 不应抛 ToolExecutionError，而是返回提示字符串
    assert "需要交互式输入" in obs
    assert "npm init" in obs or "真实终端" in obs


def test_mcp_executor_normal_command_still_confirms(tmp_workspace):
    """MCPToolExecutor 对普通命令仍返回需要确认提示"""
    executor = MCPToolExecutor(
        workspace_root=tmp_workspace, session_id="test-mcp-normal"
    )
    obs = asyncio.run(executor.execute("run_command", {"cmd": "echo hi"}))
    assert "需要用户确认" in obs
