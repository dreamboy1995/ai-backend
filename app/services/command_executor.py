"""
S8 第 75-76 天：终端命令执行器（最危险也最强大的工具）

对应 Sprint_8.md 第 75-76 天后端任务：
  1. 编写 command_executor.py，使用 asyncio.create_subprocess_exec 执行命令。
  2. 实时流式输出：将 stdout 和 stderr 通过 WebSocket 推送给插件，
     让用户看到实时执行日志（就像在真正的终端里一样）。
  3. 危险命令黑名单：已在 tool_registry.py 实现，这里不重复。
  4. 普通命令需用户确认：已在 tool_registry.py 实现。
  5. 设置超时（默认 60 秒，长任务如 npm install 可延长至 300 秒）。

风险预警应对（S8 关键技术预研与风险预警）：
  - **命令执行必须在独立的进程组中启动**：
      使用 start_new_session=True（POSIX）/ CREATE_NEW_PROCESS_GROUP（Windows），
      方便 Agent 随时 SIGTERM 杀死超时或卡住的命令，而不会影响后端主进程。
  - **命令注入攻击**：
      使用 subprocess 列表参数模式（shlex.split），从根源杜绝字符串拼接注入。
      Windows 平台对 echo/dir 等 CMD 内置命令做安全包装（cmd.exe /c）。
  - **终端命令的交互式输入**：
      检测 npm init（无 --yes）、python REPL、ssh 等交互式命令，
      通过 is_interactive_command() 返回 True，由 tool_registry 标记
      requires_interaction=True，让用户在真实终端手动完成。

架构设计：
  ┌─────────────────────────────────────────────────────────────┐
  │  CommandExecutor.execute(cmd, workspace, session, timeout)  │
  │    1. shlex.split → 列表参数（防注入）                       │
  │    2. _wrap_windows_builtin → 处理 Windows CMD 内置命令       │
  │    3. create_subprocess_exec(独立进程组)                     │
  │    4. 并发读 stdout + stderr，按行 publish 到订阅者          │
  │    5. asyncio.wait_for(proc.wait(), timeout)                 │
  │    6. 超时 → _kill_process_group(SIGTERM→SIGKILL)            │
  │    7. 返回 (exit_code, stdout_text, stderr_text)             │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │  CommandStreamManager（每会话独立 pub-sub）                 │
  │    - subscribe(session_id) → asyncio.Queue                  │
  │    - publish(session_id, StreamMessage) → 推送所有订阅者     │
  │    - WebSocket /v1/agent/stream/{session_id} 持有 Queue      │
  └─────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.models.tool import StreamMessage

logger = logging.getLogger(__name__)


# ============================================================
# 交互式命令检测（S8 风险预警：终端命令的交互式输入）
# ============================================================
# 这些命令会卡住等待用户输入，必须特殊处理：
#   - npm init 无 --yes：会询问 package name/version 等一系列问题
#   - python / node 无参数：进入交互式 REPL
#   - mysql -p（不带 < 重定向）：会等密码输入
#   - ssh：密码输入 / 交互式 shell
#   - passwd：修改密码交互
#   - sudo -S：从 stdin 读密码（Agent 没有 stdin 输入会卡住）
#
# 检测命中后，tool_run_command 返回 requires_interaction=True，
# 由插件提示用户在真实终端手动执行后告知 Agent 继续。

_INTERACTIVE_PATTERNS: List[str] = [
    # npm init 不带 -y / --yes
    r"^\s*npm\s+init\b(?:(?!\s+(-y|--yes)\b).)*$",
    # python / python3 / node 进入 REPL（命令行末尾无参数或仅 -i）
    r"^\s*python\d*\s*(-i\b)?\s*$",
    r"^\s*node\s*(-i\b)?\s*$",
    # mysql 交互式密码（-p 后无直接密码）
    r"^\s*mysql\b.*-p(?:\s|$)",
    # ssh 交互式登录（不带 -T / -N / -O）
    r"^\s*ssh\b(?:(?!\s+(-T|-N|-O|host_key_check)\b).)*$",
    # passwd 修改密码
    r"^\s*passwd\b",
    # sudo -S 读 stdin 密码
    r"^\s*sudo\s+.*-S\b",
]

_INTERACTIVE_REGEXES = [re.compile(p) for p in _INTERACTIVE_PATTERNS]


def is_interactive_command(cmd: str) -> bool:
    """
    检测命令是否可能需要交互式输入。

    Agent 执行 npm init 或 python 交互式命令会卡住，
    命中此处模式后由 tool_run_command 标记 requires_interaction=True，
    由用户在真实终端手动完成。

    Returns:
        True 表示命令需要交互式输入，不应自动执行。
    """
    for regex in _INTERACTIVE_REGEXES:
        if regex.match(cmd):
            return True
    return False


# ============================================================
# Windows CMD 内置命令包装（防 create_subprocess_exec 找不到命令）
# ============================================================
# Windows 上 echo / dir / cd / type 等是 CMD 内置命令，没有独立可执行文件，
# 直接 create_subprocess_exec(['echo', 'hello']) 会报 FileNotFoundError。
# 这些命令必须通过 cmd.exe /c 包装。仍使用列表参数模式，杜绝字符串拼接注入。

_WIN_CMD_BUILTINS = {
    "echo", "dir", "cd", "type", "cls", "copy", "del", "move", "ren",
    "rename", "mkdir", "md", "rmdir", "rd", "set", "path", "ver", "vol",
    "assoc", "ftype", "color", "date", "time", "title", "prompt",
    "pushd", "popd", "shift", "start", "call", "exit", "rem",
    "if", "for", "goto", "verify", "setlocal", "endlocal", "chcp",
    "break", "prompt",
}


def _wrap_windows_builtin(cmd_list: List[str]) -> List[str]:
    """
    在 Windows 上，若命令首段是 CMD 内置命令，包装为 cmd.exe /c 调用。

    列表参数模式仍然保留（不进行字符串拼接），杜绝命令注入。
    POSIX 平台原样返回 cmd_list（shell 内置命令在该平台有对应可执行文件，
    或由 shell 处理；S8 文档明确要求使用列表参数模式）。
    """
    if sys.platform != "win32" or not cmd_list:
        return cmd_list
    if cmd_list[0].lower() in _WIN_CMD_BUILTINS:
        return ["cmd.exe", "/c", *cmd_list]
    return cmd_list


# ============================================================
# 进程组启动参数
# ============================================================

def _process_group_kwargs() -> Dict[str, int]:
    """
    构造进程启动参数，使命令在独立进程组中运行。

    - POSIX (Linux/Mac): start_new_session=True（setsid），
      创建新会话和进程组，killpg 可以杀整个进程组。
    - Windows: creationflags=CREATE_NEW_PROCESS_GROUP，
      创建新进程组（CTRL_BREAK_EVENT 可中断整组）。

    这样 Agent 可以通过 SIGTERM 杀死整个进程组，包括子进程，
    而不会影响后端主进程（S8 风险预警：「命令执行必须在独立的进程组中启动」）。
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# ============================================================
# 时间戳工具
# ============================================================

def _now_iso() -> str:
    """ISO 8601 UTC 时间戳，对应 StreamMessage.timestamp 字段"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ============================================================
# 命令流式输出管理器（每会话 pub-sub）
# ============================================================

class CommandStreamManager:
    """
    命令流式输出管理器：每个 session_id 独立的订阅列表。

    发布-订阅模式：
      - CommandExecutor 执行命令时 publish StreamMessage 到该会话的所有订阅者
      - WebSocket /v1/agent/stream/{session_id} 持有订阅 Queue，转发给前端

    一个 session 可能有多个并发订阅者（用户在多个浏览器标签页打开同一会话）。

    队列容量上限 256 条消息：
      - 命令输出按行推送，256 行已足够覆盖典型场景
      - 队列满时丢弃最旧消息（订阅者消费过慢的兜底）
      - WebSocket 断开时 unsubscribe，避免内存泄漏
    """

    def __init__(self, queue_maxsize: int = 256):
        self._queue_maxsize = queue_maxsize
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """订阅指定会话的命令输出流，返回一个 asyncio.Queue。"""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        async with self._lock:
            if session_id not in self._subscribers:
                self._subscribers[session_id] = []
            self._subscribers[session_id].append(queue)
        logger.info(
            f"[StreamManager] 新订阅: session={session_id}, "
            f"总订阅数={len(self._subscribers.get(session_id, []))}"
        )
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """取消订阅。WebSocket 断开时调用，避免内存泄漏。"""
        async with self._lock:
            queues = self._subscribers.get(session_id)
            if queues is None:
                return
            try:
                queues.remove(queue)
            except ValueError:
                pass
            if not queues:
                del self._subscribers[session_id]
        logger.info(
            f"[StreamManager] 取消订阅: session={session_id}, "
            f"剩余订阅数={len(self._subscribers.get(session_id, []))}"
        )

    async def publish(self, session_id: str, message: StreamMessage) -> None:
        """
        发布消息到该会话的所有订阅者。

        无订阅者时静默丢弃（HTTP /v1/tool/confirm 在没有 WebSocket 连接时
        仍能正常执行命令，只是没有流式输出，stdout/stderr 仍会在结果中返回）。
        """
        async with self._lock:
            queues = list(self._subscribers.get(session_id, []))
        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # 订阅者消费过慢，丢弃最旧消息腾出空间
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except Exception:
                    pass


# 全局单例
_stream_manager: Optional[CommandStreamManager] = None


def get_stream_manager() -> CommandStreamManager:
    """获取 CommandStreamManager 单例。"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = CommandStreamManager()
    return _stream_manager


# ============================================================
# CommandExecutor：流式输出 + 进程组隔离 + SIGTERM 杀死
# ============================================================

class CommandExecutor:
    """
    终端命令执行器（S8 第 75-76 天核心模块）。

    特性：
      1. **流式输出**：实时按行推送 stdout/stderr 到订阅者
      2. **进程组隔离**：start_new_session / CREATE_NEW_PROCESS_GROUP
      3. **超时杀进程**：先 SIGTERM 给清理机会，1s 后 SIGKILL 强杀
      4. **命令注入防护**：shlex.split 列表参数模式
      5. **跨平台**：Windows CMD 内置命令自动包装 cmd.exe /c

    使用方式：
        executor = CommandExecutor()
        exit_code, stdout_text, stderr_text = await executor.execute(
            cmd="echo hello",
            workspace_root="/path/to/workspace",
            session_id="abc-123",
            timeout=60.0,
        )
    """

    def __init__(self, stream_manager: Optional[CommandStreamManager] = None):
        self._stream_manager = stream_manager or get_stream_manager()

    async def execute(
        self,
        cmd: str,
        workspace_root: str,
        session_id: str,
        timeout: float = 60.0,
    ) -> Tuple[int, str, str]:
        """
        执行命令，实时流式输出 stdout/stderr。

        Args:
            cmd:            命令字符串（将由 shlex.split 拆分为列表参数）
            workspace_root: 工作区根目录（cwd）
            session_id:     会话 ID（用于路由流式消息到订阅者）
            timeout:        超时秒数（超时后会杀掉整个进程组）

        Returns:
            (exit_code, stdout_text, stderr_text)
            超时时 exit_code=-1，stderr_text 包含超时提示。

        Raises:
            ValueError:      shlex.split 解析失败
            FileNotFoundError: 命令可执行文件不存在
            OSError:         启动进程失败
        """
        # 命令注入防护：列表参数模式
        try:
            cmd_list = shlex.split(cmd)
        except ValueError as e:
            raise ValueError(f"命令解析失败: {e}")

        if not cmd_list:
            raise ValueError("命令为空")

        # Windows CMD 内置命令包装
        cmd_list = _wrap_windows_builtin(cmd_list)

        # 进程组启动参数
        pg_kwargs = _process_group_kwargs()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_root or None,
                **pg_kwargs,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"命令不存在: {cmd_list[0]}")
        except OSError as e:
            raise OSError(f"启动进程失败: {e}")

        # 推送启动系统消息（让前端看到即将执行的命令、PID、超时）
        await self._publish_system(
            session_id,
            f"$ {cmd}\n[PID={proc.pid}, 超时={timeout}s]",
        )

        stdout_buf: List[str] = []
        stderr_buf: List[str] = []

        async def read_stream(
            stream: asyncio.StreamReader,
            buf: List[str],
            msg_type: str,
        ) -> None:
            """
            并发读取进程输出流，按行 publish 到订阅者。

            按行读取而非按字节：终端日志天然以行为单位，便于插件端渲染。
            """
            while True:
                try:
                    line = await stream.readline()
                except Exception as e:
                    # 读取异常不应导致整个执行崩溃
                    logger.warning(
                        f"[CommandExecutor] 读取 {msg_type} 异常: {e}"
                    )
                    return
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                buf.append(text)
                await self._stream_manager.publish(
                    session_id,
                    StreamMessage(
                        type=msg_type,
                        content=text,
                        timestamp=_now_iso(),
                    ),
                )

        # 并发读取 stdout + stderr（避免管道缓冲区满导致进程阻塞）
        stdout_task = asyncio.create_task(
            read_stream(proc.stdout, stdout_buf, "stdout")
        )
        stderr_task = asyncio.create_task(
            read_stream(proc.stderr, stderr_buf, "stderr")
        )

        # 等待进程结束（带超时）
        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            # 超时：杀死整个进程组（SIGTERM → SIGKILL 兜底）
            await self._kill_process_group(proc)
            await self._publish_system(
                session_id,
                f"[命令超时（{timeout}s），进程组已终止]",
            )
            # 等待读取任务自然结束（进程被杀后流会 EOF）
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        if not timed_out:
            # 等待读取任务完成（drain 剩余输出）
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        exit_code = proc.returncode if proc.returncode is not None else -1
        stdout_text = "".join(stdout_buf)
        stderr_text = "".join(stderr_buf)

        if not timed_out:
            await self._publish_system(
                session_id,
                f"[进程退出码={exit_code}]",
            )

        if timed_out:
            stderr_text = (
                f"命令执行超时（{timeout}s），进程组已被终止\n" + stderr_text
            )
            return -1, stdout_text, stderr_text

        return exit_code, stdout_text, stderr_text

    async def _kill_process_group(self, proc: asyncio.subprocess.Process) -> None:
        """
        杀死整个进程组（包括子进程），不只是主进程。

        - POSIX: killpg(SIGTERM, -pgid) 给子进程清理机会，
                 1s 后仍存活则 killpg(SIGKILL, -pgid) 强杀。
        - Windows: proc.kill()（CREATE_NEW_PROCESS_GROUP 已隔离）。

        对应 S8 风险预警："命令执行必须在独立的进程组中启动，
                          方便 Agent 随时 SIGTERM 杀死超时或卡住的命令"。
        """
        if sys.platform == "win32":
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return

        # POSIX：杀整个进程组
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        try:
            # 先 SIGTERM 给清理机会
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError:
            # 权限不足，回退到杀主进程
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return

        # 等待 1s，若仍存活则 SIGKILL 强杀
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

    async def _publish_system(self, session_id: str, content: str) -> None:
        """发布系统事件消息（启动提示、退出码、超时等元信息）。"""
        await self._stream_manager.publish(
            session_id,
            StreamMessage(
                type="system",
                content=content,
                timestamp=_now_iso(),
            ),
        )


# ============================================================
# 全局单例
# ============================================================

_executor: Optional[CommandExecutor] = None


def get_command_executor() -> CommandExecutor:
    """获取 CommandExecutor 单例。"""
    global _executor
    if _executor is None:
        _executor = CommandExecutor()
    return _executor
