"""
S8 第 71-72 天：工具注册中心（MCP 协议适配层核心）

对应 Sprint_8.md 第 71-72 天任务：
  "编写 tool_registry.py，将每个工具实现为一个异步函数，并注册到字典 {tool_name: handler_function}。
   编写统一的 execute_tool(tool_call: ToolCall) -> ToolResult 入口函数，负责路由和异常捕获。"

同时兼顾：
  - 关键接口变更：POST /v1/tool/execute、POST /v1/tool/confirm
  - 风险预警：命令注入（subprocess 列表参数）、大文件截断、危险命令黑名单、审计日志

架构设计：
  ┌─────────────────────────────────────────────────────────┐
  │  execute_tool(tool_call, workspace_root, session_id)    │
  │    1. 路由到注册的 handler                               │
  │    2. 若 handler 返回 requires_confirmation=True        │
  │       → 生成 confirmation_id，存入 PendingConfirmationStore│
  │       → 不真正执行，返回 ToolResult 供插件展示确认框     │
  │    3. 异常捕获 → 返回 success=False 的 ToolResult        │
  │    4. 所有调用写入审计日志（JSON Lines）                 │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  confirm_tool(confirmation_id, action, session_id)      │
  │    1. 从 PendingConfirmationStore 取出待确认操作         │
  │    2. action='allow' → 真正执行该工具                    │
  │    3. action='deny'  → 返回拒绝结果                      │
  │    4. 无论 allow/deny，确认后清除记录（一次性凭证）      │
  └─────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from app.config import settings
from app.models.tool import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# ============================================================
# S8 第 77-78 天：git_commit 自动生成 Commit Message（LLM 驱动）
# ============================================================
# 当用户未提供 commit message 时，基于 git diff（优先 staged，fallback 工作区）
# 调用 LLM 按 Conventional Commits 规范生成 message。
#
# 设计要点：
#   - 在 tool_git_commit（execute 阶段）尝试生成，让用户在确认框预览 AI 建议
#   - LLM 调用失败（超时 / 限流 / 网络）时静默降级为 settings.GIT_COMMIT_FALLBACK_MESSAGE
#   - diff 过长时截断（settings.GIT_COMMIT_DIFF_MAX_CHARS），避免 Prompt 膨胀
#   - 异步安全：LLM 调用不阻塞工具注册中心的其他 handler

_COMMIT_MESSAGE_SYSTEM_PROMPT = """你是一个专业的 Git Commit Message 生成器。
请根据提供的 git diff 输出，生成一条符合 Conventional Commits 规范的 commit message。

规范：
- 格式：<type>(<scope>): <subject>
- type 必须为以下之一：feat / fix / refactor / docs / style / test / chore / ci / build / perf
- scope 可选，用括号括起相关模块名
- subject 用简短的中文或英文描述变更内容，不超过 72 字符
- 只输出 commit message 本身，不要任何解释、前后缀或 Markdown 格式

示例：
- feat(auth): 增加 OAuth2 登录支持
- fix(api): 修复用户列表分页参数未传递的 bug
- refactor(utils): 提取字符串工具函数到独立模块
- docs: 更新 README 安装说明"""


def _truncate_diff_for_llm(diff_text: str) -> str:
    """
    截断 git diff 以适配 LLM Prompt 大小限制。
    超过 settings.GIT_COMMIT_DIFF_MAX_CHARS 时保留前 2/3 + 后 1/3。
    """
    max_chars = settings.GIT_COMMIT_DIFF_MAX_CHARS
    if not diff_text or len(diff_text) <= max_chars:
        return diff_text

    # 前 2/3 + 后 1/3，中间插入截断标记
    head_size = int(max_chars * 2 / 3)
    tail_size = max_chars - head_size - 60  # 留空间给截断标记
    if tail_size < 0:
        tail_size = 0

    return (
        diff_text[:head_size]
        + f"\n... [diff truncated: {len(diff_text) - max_chars} chars omitted] ...\n"
        + diff_text[-tail_size:]
    )


async def _generate_commit_message_via_llm(diff_text: str) -> Optional[str]:
    """
    调用 LLM 基于 git diff 生成 commit message。

    Returns:
        生成的 commit message 字符串；LLM 不可用或输出无效时返回 None。
    """
    if not settings.GIT_COMMIT_AUTO_MESSAGE_ENABLED or not diff_text.strip():
        return None

    try:
        from app.services.llm import (
            AdapterError,
            AdapterFactory,
            chat_completion_text,
        )

        truncated_diff = _truncate_diff_for_llm(diff_text)
        messages = [
            {"role": "system", "content": _COMMIT_MESSAGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"以下是 git diff：\n\n```diff\n{truncated_diff}\n```"},
        ]

        raw = await chat_completion_text(
            messages=messages,
            model=settings.GIT_COMMIT_AUTO_MESSAGE_MODEL,
            temperature=settings.GIT_COMMIT_AUTO_MESSAGE_TEMPERATURE,
            timeout=settings.GIT_COMMIT_AUTO_MESSAGE_TIMEOUT_SECONDS,
            max_tokens=settings.GIT_COMMIT_AUTO_MESSAGE_MAX_TOKENS,
        )

        # 清洗输出：去除可能的 ``` 包裹和多余空白
        message = raw.strip()
        if message.startswith("```"):
            # 去掉开头的 ``` 或 ```text / ```bash 等标记
            first_newline = message.find("\n")
            if first_newline != -1:
                message = message[first_newline + 1:]
        if message.endswith("```"):
            message = message[:-3]
        message = message.strip()

        # 取第一行（commit message 通常只需要第一行 subject）
        first_line = message.split("\n", 1)[0].strip()

        # 简单校验：Conventional Commits 格式正则
        cc_pattern = r"^(feat|fix|refactor|docs|style|test|chore|ci|build|perf)(\([a-zA-Z0-9_\-/]+\))?:\s+.+"
        if first_line and re.match(cc_pattern, first_line):
            logger.info(f"[ToolRegistry] LLM 生成 commit message: {first_line}")
            return first_line
        elif first_line and len(first_line) <= 120:
            # 格式不严格但看起来像一句话，作为 fallback 接受
            logger.warning(
                f"[ToolRegistry] LLM 输出不符合 Conventional Commits 格式，"
                f"作为降级方案接受: {first_line}"
            )
            return first_line

        logger.warning(
            f"[ToolRegistry] LLM 生成的 commit message 无效: {raw[:100]}"
        )
        return None

    except AdapterError as e:
        logger.warning(f"[ToolRegistry] LLM 调用失败，降级为默认 message: {e}")
        return None
    except Exception as e:
        logger.warning(f"[ToolRegistry] 自动生成 commit message 异常: {e}", exc_info=True)
        return None


async def _get_git_diff_for_commit(workspace_root: str) -> str:
    """
    获取用于生成 commit message 的 git diff。

    优先级：
      1. git diff --staged（已暂存的变更，最精确）
      2. git diff（未暂存但已修改的文件）
      3. git diff HEAD（兜底，包含所有差异）

    返回 diff 文本；无变更或 git 不可用时返回空串。
    """
    # 1. staged diff
    staged = await _run_git(["diff", "--staged"], workspace_root)
    if staged.success and staged.output.strip():
        return staged.output

    # 2. 工作区 diff（untracked 文件不在 diff 里，但 status 已经确认有变更了）
    unstaged = await _run_git(["diff"], workspace_root)
    if unstaged.success and unstaged.output.strip():
        return unstaged.output

    # 3. 兜底：diff HEAD
    head = await _run_git(["diff", "HEAD"], workspace_root)
    if head.success and head.output.strip():
        return head.output

    return ""


# ============================================================
# 工具 handler 类型签名
# ============================================================
# 每个工具 handler 接收 (arguments, workspace_root, session_id)，
# 返回 ToolResult。workspace_root 用于解析相对路径，session_id 用于审计。
ToolHandler = Callable[[Dict[str, Any], str, str], Awaitable[ToolResult]]


# ============================================================
# 待确认操作存储（内存 + TTL）
# ============================================================
# 当工具返回 requires_confirmation=True 时，生成 confirmation_id 并存入此处。
# 用户通过 /v1/tool/confirm 回传 confirmation_id 后，后端取出原始 ToolCall 执行。
# 一次性凭证：确认后即删除；超时自动失效。

class _PendingConfirmation:
    """一条待确认的工具调用记录"""

    def __init__(
        self,
        tool_call: ToolCall,
        workspace_root: str,
        session_id: str,
        created_at: float,
    ):
        self.tool_call = tool_call
        self.workspace_root = workspace_root
        self.session_id = session_id
        self.created_at = created_at


class PendingConfirmationStore:
    """
    待确认操作存储（线程安全，内存实现）。

    - confirmation_id 为 uuid4，不可预测，防止伪造确认请求。
    - TTL 由 settings.TOOL_CONFIRMATION_TTL_SECONDS 控制，超时自动失效。
    - 确认后立即删除（一次性凭证）。
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._store: Dict[str, _PendingConfirmation] = {}
        self._lock = asyncio.Lock()

    async def put(
        self, tool_call: ToolCall, workspace_root: str, session_id: str
    ) -> str:
        """存入一条待确认记录，返回 confirmation_id"""
        confirmation_id = str(uuid.uuid4())
        async with self._lock:
            self._store[confirmation_id] = _PendingConfirmation(
                tool_call=tool_call,
                workspace_root=workspace_root,
                session_id=session_id,
                created_at=time.time(),
            )
        logger.info(
            f"[ToolRegistry] 生成确认凭证: id={confirmation_id}, "
            f"tool={tool_call.tool_name}, session={session_id}"
        )
        return confirmation_id

    async def get(self, confirmation_id: str) -> Optional[_PendingConfirmation]:
        """
        获取待确认记录。若已过期则删除并返回 None。
        """
        async with self._lock:
            record = self._store.get(confirmation_id)
            if record is None:
                return None
            # TTL 检查
            if time.time() - record.created_at > self._ttl:
                del self._store[confirmation_id]
                logger.info(
                    f"[ToolRegistry] 确认凭证已过期: id={confirmation_id}"
                )
                return None
            return record

    async def pop(self, confirmation_id: str) -> Optional[_PendingConfirmation]:
        """
        取出并删除待确认记录（确认后调用，一次性凭证）。
        """
        async with self._lock:
            record = self._store.pop(confirmation_id, None)
            if record is not None:
                logger.info(
                    f"[ToolRegistry] 消耗确认凭证: id={confirmation_id}, "
                    f"tool={record.tool_call.tool_name}"
                )
            return record


# 全局单例
_pending_store: Optional[PendingConfirmationStore] = None


def get_pending_confirmation_store() -> PendingConfirmationStore:
    global _pending_store
    if _pending_store is None:
        _pending_store = PendingConfirmationStore(
            ttl_seconds=settings.TOOL_CONFIRMATION_TTL_SECONDS
        )
    return _pending_store


# ============================================================
# 审计日志（JSON Lines）
# ============================================================
# 对应 S8 第 77-78 天："将所有工具的执行日志写入审计日志（JSON Lines）"。
# 提前在第 71-72 天落地，确保从第一天起就有完整的工具调用记录。

_audit_lock = asyncio.Lock()


def _ensure_audit_dir() -> None:
    """确保审计日志目录存在"""
    log_path = Path(settings.TOOL_AUDIT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)


async def write_audit_log(
    session_id: str,
    tool_call: ToolCall,
    result: ToolResult,
    duration_ms: float = 0.0,
) -> None:
    """
    异步写入审计日志（JSON Lines）。

    记录格式（每行一个 JSON）：
    {
      "timestamp": "2026-10-04T10:00:00.123Z",
      "session_id": "...",
      "tool_name": "read_file",
      "arguments": {"file_path": "main.py"},
      "success": true,
      "requires_confirmation": false,
      "duration_ms": 12.34,
      "output_chars": 1024,
      "output_preview": "...前 500 字符...",    // 可选，根据 TOOL_AUDIT_LOG_OUTPUT_MAX_CHARS
      "error": null,
      "has_confirmation_id": false
    }

    设计要点（S8 第 77-78 天："用于后续调试和 P4 的数据飞轮"）：
      - arguments 完整记录（便于复现）
      - output 仅记录字符数 + 可选截断预览（避免日志膨胀，大输出如 run_command 的
        npm install 日志可达数十 MB，全量写日志会拖垮磁盘）
      - duration_ms 记录执行耗时（毫秒级，便于定位慢工具）
      - 用 time.strftime + utc 时间戳，保证跨时区一致性
      - 审计日志写入失败不阻断工具执行（try/except 内部吞掉）

    Args:
        session_id:      会话 ID。
        tool_call:       原始工具调用。
        result:          工具执行结果。
        duration_ms:     工具执行耗时（毫秒）。0 表示未统计（兼容旧调用）。
    """
    try:
        _ensure_audit_dir()

        # output 预览：可选截断（S8 风险预警：大输出撑爆日志）
        output_chars = len(result.output) if result.output else 0
        max_preview = settings.TOOL_AUDIT_LOG_OUTPUT_MAX_CHARS
        output_preview = None
        if result.output and max_preview > 0 and output_chars > 0:
            if output_chars <= max_preview:
                output_preview = result.output
            else:
                output_preview = (
                    result.output[:max_preview]
                    + f"\n... [truncated, total {output_chars} chars] ..."
                )

        entry = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S.") + f"{int((time.time() % 1) * 1000):03d}Z",
            "session_id": session_id,
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "success": result.success,
            "requires_confirmation": result.requires_confirmation,
            "duration_ms": round(duration_ms, 2),
            "output_chars": output_chars,
            "error": result.error,
            "has_confirmation_id": result.confirmation_id is not None,
        }
        if output_preview is not None:
            entry["output_preview"] = output_preview

        async with _audit_lock:
            with open(settings.TOOL_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # 审计日志失败不应阻断工具执行
        logger.warning(f"[ToolRegistry] 审计日志写入失败: {e}")


# ============================================================
# 路径安全工具
# ============================================================

def _resolve_safe_path(file_path: str, workspace_root: str) -> Optional[Path]:
    """
    将相对路径解析为工作区内的绝对路径，并校验不逃逸出工作区。

    对应 S6 diff_generator.py 的 _is_path_within 逻辑，统一路径安全策略。
    防止 ../../../etc/passwd 之类的路径逃逸。

    Returns:
        解析后的绝对 Path；若逃逸则返回 None。
    """
    if not workspace_root:
        return None
    rel = file_path.replace("\\", "/")
    full = Path(workspace_root) / rel
    try:
        root_resolved = Path(workspace_root).resolve()
        full_resolved = full.resolve()
        common = os.path.commonpath([str(root_resolved), str(full_resolved)])
        if Path(common) != root_resolved:
            logger.warning(
                f"[ToolRegistry] 路径逃逸拦截: file_path={file_path}, "
                f"workspace_root={workspace_root}"
            )
            return None
        return full_resolved
    except (ValueError, OSError):
        return None


# ============================================================
# 工具实现
# ============================================================

async def tool_read_file(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """
    read_file：读取文件内容。

    arguments:
      - file_path:  文件相对路径（相对于 workspace_root），必填。
                    兼容 LLM 可能返回的别名 'path'（ReAct prompt 里写的是 path）。
      - start_line: 起始行号（1-based，可选）。
      - end_line:   结束行号（1-based，闭区间，可选）。

    安全限制（S8 第 73-74 天）：
      - 文件大小 > TOOL_READ_FILE_MAX_BYTES 时自动截断，
        返回前 HEAD_LINES 行 + 后 TAIL_LINES 行，并附带警告。
    """
    # 参数名兼容：LLM 可能返回 'path'（prompt 里写的），handler 期望 'file_path'
    file_path = arguments.get("file_path") or arguments.get("path")
    if not file_path or not isinstance(file_path, str):
        return ToolResult(success=False, error="缺少必填参数 file_path")

    full_path = _resolve_safe_path(file_path, workspace_root)
    if full_path is None:
        return ToolResult(success=False, error=f"文件路径非法或逃逸出工作区: {file_path}")

    if not full_path.exists():
        return ToolResult(success=False, error=f"文件不存在: {file_path}")
    if not full_path.is_file():
        return ToolResult(success=False, error=f"路径不是文件: {file_path}")

    # 大文件截断保护
    file_size = full_path.stat().st_size
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            if file_size > settings.TOOL_READ_FILE_MAX_BYTES:
                # 大文件：只读前 HEAD_LINES + 后 TAIL_LINES 行
                head_lines: list[str] = []
                for i, line in enumerate(f):
                    if i >= settings.TOOL_READ_FILE_HEAD_LINES:
                        break
                    head_lines.append(line)
                # 读取尾部
                # 简化实现：用 seek 到末尾附近
                try:
                    f.seek(0, os.SEEK_END)
                    end_pos = f.tell()
                    # 估算尾部 TAIL_LINES 行的字节数（按平均 80 字符/行）
                    tail_bytes = settings.TOOL_READ_FILE_TAIL_LINES * 120
                    seek_pos = max(0, end_pos - tail_bytes)
                    f.seek(seek_pos)
                    tail_content = f.read()
                    tail_lines = tail_content.splitlines(keepends=True)
                    # 只保留最后 TAIL_LINES 行
                    tail_lines = tail_lines[-settings.TOOL_READ_FILE_TAIL_LINES:]
                except OSError:
                    tail_lines = []

                warning = (
                    f"\n... [文件过大（{file_size} bytes > "
                    f"{settings.TOOL_READ_FILE_MAX_BYTES}），已截断，"
                    f"仅显示前 {settings.TOOL_READ_FILE_HEAD_LINES} 行和"
                    f"后 {settings.TOOL_READ_FILE_TAIL_LINES} 行] ...\n"
                )
                content = "".join(head_lines) + warning + "".join(tail_lines)
            else:
                content = f.read()
    except OSError as e:
        return ToolResult(success=False, error=f"读取文件失败: {e}")

    # 行号区间过滤
    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")
    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        total = len(lines)
        s = int(start_line) if start_line is not None else 1
        e = int(end_line) if end_line is not None else total
        s = max(1, min(s, total))
        e = max(s, min(e, total))
        content = "".join(lines[s - 1 : e])

    return ToolResult(success=True, output=content)


async def tool_write_file(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """
    write_file：写入文件（需用户确认）。

    arguments:
      - file_path: 文件相对路径，必填。
      - content:   要写入的内容，必填。
      - mode:      'overwrite'（默认）或 'append'。

    安全机制（S8 第 73-74 天）：
      - 不直接写入，而是返回 requires_confirmation=True，
        output 字段包含 Diff 预览（复用 diff_generator）。
      - 用户在插件端点击确认后，由 confirm_tool 调用 _do_write_file 真正落盘。
    """
    file_path = arguments.get("file_path") or arguments.get("path")
    content = arguments.get("content")
    mode = arguments.get("mode", "overwrite")

    if not file_path or not isinstance(file_path, str):
        return ToolResult(success=False, error="缺少必填参数 file_path")
    if content is None or not isinstance(content, str):
        return ToolResult(success=False, error="缺少必填参数 content")
    if mode not in ("overwrite", "append"):
        return ToolResult(success=False, error=f"非法的 mode: {mode}，仅支持 overwrite / append")

    full_path = _resolve_safe_path(file_path, workspace_root)
    if full_path is None:
        return ToolResult(success=False, error=f"文件路径非法或逃逸出工作区: {file_path}")

    # 读取原文件内容，生成 Diff 预览
    try:
        from app.services.diff_generator import (
            generate_unified_diff,
            determine_context_lines,
        )

        old_content = ""
        if full_path.exists() and full_path.is_file():
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                old_content = f.read()

        if mode == "append":
            new_content = old_content + content
        else:
            new_content = content

        context_lines = determine_context_lines(
            old_content if len(old_content) >= len(new_content) else new_content
        )
        diff_text = generate_unified_diff(
            old_content, new_content, file_path, context_lines=context_lines
        )
    except Exception as e:
        logger.warning(f"[ToolRegistry] write_file Diff 生成失败，降级为无预览: {e}")
        diff_text = ""

    # 构造确认提示
    action_desc = "追加" if mode == "append" else "覆盖写入"
    prompt = f"即将{action_desc}文件 {file_path}（{len(content)} 字符），是否继续？"

    return ToolResult(
        success=True,
        output=diff_text or f"[Diff 预览不可用]\n原内容长度: {len(old_content)}\n新内容长度: {len(content)}",
        requires_confirmation=True,
        confirmation_prompt=prompt,
    )


async def _do_write_file(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """
    write_file 的真正执行逻辑（用户确认后由 confirm_tool 调用）。

    风险预警应对（大文件写入 OOM）：
      - 分块流式写入，而非一次性 f.write(content)。
      - 虽然 content 已经在内存中（来自模型输出），但分块写入可减少
        单次 IO 压力，并在中途失败时保留已写入部分。
    """
    file_path = arguments.get("file_path") or arguments.get("path")
    content = arguments["content"]
    mode = arguments.get("mode", "overwrite")

    full_path = _resolve_safe_path(file_path, workspace_root)
    if full_path is None:
        return ToolResult(success=False, error=f"文件路径非法: {file_path}")

    # 确保父目录存在
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return ToolResult(success=False, error=f"创建目录失败: {e}")

    try:
        open_mode = "a" if mode == "append" else "w"
        with open(full_path, open_mode, encoding="utf-8") as f:
            # 分块写入（每块 64KB），应对大文件 OOM 风险
            chunk_size = 64 * 1024
            for i in range(0, len(content), chunk_size):
                f.write(content[i : i + chunk_size])
    except OSError as e:
        return ToolResult(success=False, error=f"写入文件失败: {e}")

    return ToolResult(success=True, output=f"文件 {file_path} 写入成功（{len(content)} 字符）")


async def tool_run_command(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """
    run_command：执行终端命令（需用户确认，危险命令直接拦截）。

    arguments:
      - cmd:     要执行的命令字符串，必填。
      - timeout: 超时秒数（可选，默认 60，最大 300）。

    安全机制（S8 第 75-76 天）：
      1. 危险命令黑名单：匹配 TOOL_DANGER_COMMAND_PATTERNS 的命令直接拒绝，
         无需用户确认（requires_confirmation=False）。
      2. **交互式命令检测（S8 风险预警：终端命令的交互式输入）**：
         命中 is_interactive_command() 的命令（npm init 无 --yes / python REPL
         / ssh / mysql -p 等）返回 requires_interaction=True，由插件提示用户
         在真实终端手动执行后告知 Agent 继续。Agent 自动执行会卡死。
      3. 普通命令需用户确认：返回 requires_confirmation=True，
         用户确认后由 confirm_tool 调用 _do_run_command 执行。
      4. 真正执行时委托给 CommandExecutor（独立进程组 + 流式输出）。
    """
    from app.services.command_executor import is_interactive_command

    cmd = arguments.get("cmd")
    if not cmd or not isinstance(cmd, str):
        return ToolResult(success=False, error="缺少必填参数 cmd")

    # 危险命令黑名单检查
    for pattern in settings.TOOL_DANGER_COMMAND_PATTERNS:
        if re.search(pattern, cmd):
            logger.warning(
                f"[ToolRegistry] 危险命令已拦截: session={session_id}, cmd={cmd!r}"
            )
            return ToolResult(
                success=False,
                error="危险命令已被拦截，禁止执行",
            )

    # 交互式命令检测（S8 风险预警：终端命令的交互式输入）
    # 命中后不弹确认框，直接返回 requires_interaction=True
    if is_interactive_command(cmd):
        logger.info(
            f"[ToolRegistry] 检测到交互式命令，需用户手动执行: "
            f"session={session_id}, cmd={cmd!r}"
        )
        return ToolResult(
            success=False,
            output=f"[需要交互式输入] {cmd}",
            error="命令需要交互式输入（如密码、选项确认等），Agent 自动执行会卡死。"
                  "请在真实终端手动执行后告知 Agent 继续，或在命令中追加非交互参数"
                  "（如 npm init --yes、python -c \"code\"）。",
            requires_interaction=True,
        )

    # 普通命令需用户确认
    prompt = f"即将执行命令：\n{cmd}\n\n是否允许执行？"
    return ToolResult(
        success=True,
        output=f"[待确认命令]\n{cmd}",
        requires_confirmation=True,
        confirmation_prompt=prompt,
    )


async def _do_run_command(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """
    run_command 的真正执行逻辑（用户确认后由 confirm_tool 调用）。

    S8 第 75-76 天：委托给 CommandExecutor 实现：
      - 流式输出：stdout/stderr 通过 CommandStreamManager 实时推送
        给 WebSocket 订阅者（/v1/agent/stream/{session_id}）。
      - 进程组隔离：start_new_session / CREATE_NEW_PROCESS_GROUP，
        方便 SIGTERM 杀死整个进程组。
      - 命令注入防护：shlex.split 列表参数模式。

    风险预警应对：
      - 交互式命令二次检测（安全网，防止绕过 tool_run_command 直接调用 confirm）。
      - 超时杀进程：先 SIGTERM 后 SIGKILL（在 CommandExecutor._kill_process_group）。
    """
    from app.services.command_executor import (
        CommandExecutor,
        get_command_executor,
        is_interactive_command,
    )

    cmd = arguments["cmd"]
    timeout = float(
        arguments.get("timeout", settings.TOOL_COMMAND_DEFAULT_TIMEOUT)
    )
    timeout = min(timeout, settings.TOOL_COMMAND_MAX_TIMEOUT)

    # 交互式命令二次检测（安全网）
    if is_interactive_command(cmd):
        logger.warning(
            f"[ToolRegistry] 确认阶段检测到交互式命令，已阻止执行: "
            f"session={session_id}, cmd={cmd!r}"
        )
        return ToolResult(
            success=False,
            error="命令需要交互式输入，请在真实终端手动执行",
            requires_interaction=True,
        )

    executor = get_command_executor()
    try:
        exit_code, stdout_text, stderr_text = await executor.execute(
            cmd=cmd,
            workspace_root=workspace_root,
            session_id=session_id,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        return ToolResult(success=False, error=str(e))
    except ValueError as e:
        return ToolResult(success=False, error=str(e))
    except OSError as e:
        return ToolResult(success=False, error=str(e))

    output = stdout_text
    if stderr_text:
        output += ("\n" if output else "") + stderr_text

    if exit_code != 0:
        return ToolResult(
            success=False,
            output=output,
            error=f"命令退出码 {exit_code}",
        )

    return ToolResult(success=True, output=output)


async def tool_grep_search(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """
    grep_search：正则搜索代码。

    arguments:
      - pattern:       正则模式，必填。
      - path:          搜索目录（相对 workspace_root，默认 "."）。
      - max_results:   最大结果数（默认 100）。
      - context_lines: 每个匹配行前后展示的上下文行数（默认 2，最大 10）。

    实现策略（S8 第 73-74 天）：
      - 优先使用 ripgrep（rg），若未安装则降级为 Python re + os.walk。
      - 返回匹配的「文件路径 + 行号 + 上下文行」，格式遵循 ripgrep 约定：
        匹配行用 `path:lineno:content`，上下文行用 `path:lineno-content`，
        不同匹配组之间用 `--` 分隔。
      - 设置超时 TOOL_GREP_TIMEOUT_SECONDS，防止在 node_modules 中卡死。
      - 自动跳过 node_modules / .git / __pycache__ 等大型依赖目录。
    """
    pattern = arguments.get("pattern")
    if not pattern or not isinstance(pattern, str):
        return ToolResult(success=False, error="缺少必填参数 pattern")

    search_path = arguments.get("path", ".")
    max_results = int(arguments.get("max_results", 100))
    context_lines = int(arguments.get("context_lines", 2))
    # 限制上下文行数上限，避免输出膨胀
    context_lines = max(0, min(context_lines, 10))

    full_search_path = _resolve_safe_path(search_path, workspace_root)
    if full_search_path is None:
        return ToolResult(success=False, error=f"搜索路径非法: {search_path}")
    if not full_search_path.exists():
        return ToolResult(success=False, error=f"搜索路径不存在: {search_path}")

    # 优先尝试 ripgrep
    rg_available = shutil_which("rg") is not None
    if rg_available:
        # 计算相对于 workspace_root 的搜索路径，使 ripgrep 输出相对路径
        # （与 Python 降级模式的 os.path.relpath 输出保持一致）
        rel_search = os.path.relpath(str(full_search_path), workspace_root) if workspace_root else str(full_search_path)
        result = await _grep_with_ripgrep(
            pattern, rel_search, max_results, context_lines, workspace_root
        )
        if result is not None:
            return result
        # ripgrep 失败时降级到 Python 实现
        logger.info("[ToolRegistry] ripgrep 执行失败，降级到 Python 正则搜索")

    return await _grep_with_python(
        pattern, str(full_search_path), max_results, context_lines, workspace_root
    )


def shutil_which(cmd: str) -> Optional[str]:
    """shutil.which 的封装（便于测试 mock）"""
    import shutil
    return shutil.which(cmd)


async def _grep_with_ripgrep(
    pattern: str, search_path: str, max_results: int, context_lines: int,
    workspace_root: str,
) -> Optional[ToolResult]:
    """
    使用 ripgrep 搜索。返回 ToolResult 或 None（失败时降级）。

    - 使用 `-C {context_lines}` 输出匹配行前后的上下文行。
    - 使用 `--glob '!**/node_modules/**'` 等跳过大型依赖目录，
      防止在 node_modules 中卡死（S8 风险预警）。
    - 使用列表参数模式（create_subprocess_exec），杜绝命令注入。
    """
    # 跳过的大型目录 glob 模式（与 Python 降级实现的 skip_dirs 保持一致）
    skip_globs = [
        "!**/node_modules/**",
        "!**/.git/**",
        "!**/__pycache__/**",
        "!**/.ai_index/**",
        "!**/.ai_cache/**",
    ]

    args = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color", "never",
        "--max-count", str(max_results),
    ]
    if context_lines > 0:
        args += ["-C", str(context_lines)]
    for g in skip_globs:
        args += ["--glob", g]
    args += [pattern, search_path]

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_root or None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=settings.TOOL_GREP_TIMEOUT_SECONDS
        )
        if proc.returncode in (0, 1):  # 0=有匹配, 1=无匹配
            raw_output = stdout.decode("utf-8", errors="replace")
            # 路径归一化：ripgrep 在 Windows 上输出反斜杠，且搜索 "." 时前缀 ".\"
            # 统一为正斜杠并去除 "./" 前缀，与 Python 降级模式输出保持一致
            output = _normalize_ripgrep_paths(raw_output)
            return ToolResult(success=True, output=output)
        return None
    except (asyncio.TimeoutError, OSError):
        return None


def _normalize_ripgrep_paths(output: str) -> str:
    """
    归一化 ripgrep 输出中的路径分隔符与前缀。

    ripgrep 在 Windows 上使用反斜杠，且当搜索路径为 "." 时会输出 ".\\" 前缀。
    统一转换为正斜杠并去除 "./" 前缀，使插件端解析逻辑与 Python 降级模式一致。
    """
    if not output:
        return output
    normalized_lines = []
    for line in output.splitlines():
        # 替换反斜杠为正斜杠
        line = line.replace("\\", "/")
        # 去除行首的 "./" 前缀（ripgrep 搜索 "." 目录时产生）
        if line.startswith("./"):
            line = line[2:]
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


async def _grep_with_python(
    pattern: str, search_path: str, max_results: int, context_lines: int,
    workspace_root: str,
) -> ToolResult:
    """
    Python 降级实现：re + os.walk，带超时保护。

    输出格式遵循 ripgrep 约定（便于插件端统一解析）：
      - 匹配行：  `path:lineno:content`
      - 上下文行：`path:lineno-content`
      - 不同匹配组之间用 `--` 分隔
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return ToolResult(success=False, error=f"正则表达式无效: {e}")

    results: list[str] = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".ai_index", ".ai_cache"}

    match_count = 0

    def _process_file(fpath: str) -> bool:
        """
        处理单个文件，将匹配结果（含上下文行）追加到 results。
        返回 True 表示已达到 max_results，调用方应停止遍历。
        """
        nonlocal match_count
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            return False

        total = len(lines)
        # 找出所有匹配行号
        match_line_nos: set[int] = set()
        for idx, line in enumerate(lines):
            if regex.search(line):
                match_line_nos.add(idx)  # 0-based
                match_count += 1
                if match_count >= max_results:
                    break

        if not match_line_nos:
            return False

        rel = os.path.relpath(fpath, workspace_root).replace("\\", "/")

        # 合并重叠的上下文区间
        # 每个匹配行 i 的上下文区间为 [i - ctx, i + ctx]
        ranges: list[tuple[int, int]] = []
        for ln in sorted(match_line_nos):
            start = max(0, ln - context_lines)
            end = min(total - 1, ln + context_lines)
            if ranges and start <= ranges[-1][1] + 1:
                # 与上一区间重叠或相邻，合并
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((start, end))

        # 按区间输出
        # 格式遵循 ripgrep 约定：
        #   匹配行：  path:lineno:content
        #   上下文行：path-lineno-content
        for r_idx, (start, end) in enumerate(ranges):
            if r_idx > 0:
                results.append("--")
            for idx in range(start, end + 1):
                if idx in match_line_nos:
                    results.append(f"{rel}:{idx + 1}:{lines[idx].rstrip()}")
                else:
                    results.append(f"{rel}-{idx + 1}-{lines[idx].rstrip()}")

        return match_count >= max_results

    async def _search():
        for root, dirs, files in os.walk(search_path):
            # 跳过大型依赖目录
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                fpath = os.path.join(root, fname)
                if _process_file(fpath):
                    return

    try:
        await asyncio.wait_for(_search(), timeout=settings.TOOL_GREP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            error=f"搜索超时（{settings.TOOL_GREP_TIMEOUT_SECONDS}s），结果可能不完整",
            output="\n".join(results),
        )

    return ToolResult(success=True, output="\n".join(results))


async def tool_git_commit(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """
    git_commit：Git 提交（需用户确认）。

    arguments:
      - message: 提交信息（可选）。未提供时自动调用 LLM 基于暂存区 Diff 生成。

    安全机制（S8 第 77-78 天）：
      - 提交前先执行 git status，若工作区无变更则直接返回失败，不执行空提交。
      - 未提供 message 时，调用 LLM 自动生成（基于 git diff，遵循 Conventional Commits）。
        LLM 不可用时降级为 settings.GIT_COMMIT_FALLBACK_MESSAGE。
      - 返回 requires_confirmation=True，output 包含变更文件列表 + AI 建议的 message，
        用户确认后由 confirm_tool 调用 _do_git_commit 执行。

    风险预警应对：
      - LLM 调用有独立超时（GIT_COMMIT_AUTO_MESSAGE_TIMEOUT_SECONDS），不阻塞主流程。
      - diff 过长时截断（_truncate_diff_for_llm），避免 Prompt 膨胀。
      - 所有异常降级：LLM 失败不影响 git_commit 主流程，用 fallback message 继续。
    """
    # 先检查 git 状态
    status_result = await _run_git(["status", "--short"], workspace_root)
    if not status_result.success:
        return ToolResult(
            success=False,
            error=f"获取 git status 失败: {status_result.error}",
        )

    status_output = status_result.output.strip()
    if not status_output:
        return ToolResult(success=False, error="工作区无变更，无需提交")

    message = arguments.get("message", "").strip()

    # 未提供 message → 尝试自动生成（S8 第 77-78 天核心功能）
    auto_generated = False
    if not message:
        message, auto_generated = await _try_auto_generate_message(
            workspace_root, session_id
        )

    # 更新 arguments 确保 message 字段完整（供后续 confirm_tool 使用）
    arguments["message"] = message

    # 构造确认提示
    ai_hint = "（AI 自动生成）" if auto_generated else ""
    prompt = (
        f"即将执行 git commit，提交信息{ai_hint}：\n{message}\n\n"
        f"涉及变更文件：\n{status_output}\n\n是否继续？"
    )

    output = f"[变更文件]\n{status_output}\n\n[Commit Message{ai_hint}]\n{message}"

    return ToolResult(
        success=True,
        output=output,
        requires_confirmation=True,
        confirmation_prompt=prompt,
    )


async def _try_auto_generate_message(
    workspace_root: str, session_id: str
) -> Tuple[str, bool]:
    """
    尝试自动生成 commit message。

    Returns:
        (message, auto_generated_flag): message 为最终要用的 commit message，
        auto_generated_flag=True 表示是 LLM 生成的，False 表示降级为 fallback。
    """
    diff_text = await _get_git_diff_for_commit(workspace_root)

    llm_generated = await _generate_commit_message_via_llm(diff_text)
    if llm_generated:
        return llm_generated, True

    # LLM 不可用 / 失败 → 使用 fallback
    fallback = settings.GIT_COMMIT_FALLBACK_MESSAGE
    logger.info(
        f"[ToolRegistry] 使用 fallback commit message: {fallback}, session={session_id}"
    )
    return fallback, False


async def _do_git_commit(
    arguments: Dict[str, Any], workspace_root: str, session_id: str
) -> ToolResult:
    """git_commit 的真正执行逻辑（用户确认后调用）。"""
    message = arguments.get("message", "").strip()
    if not message:
        message = settings.GIT_COMMIT_FALLBACK_MESSAGE

    # git add -A
    add_result = await _run_git(["add", "-A"], workspace_root)
    if not add_result.success:
        return ToolResult(success=False, error=f"git add 失败: {add_result.error}")

    # git commit -m
    commit_result = await _run_git(
        ["commit", "-m", message], workspace_root
    )
    if not commit_result.success:
        return ToolResult(success=False, error=f"git commit 失败: {commit_result.error or commit_result.output}")

    # 获取 commit hash
    hash_result = await _run_git(["rev-parse", "HEAD"], workspace_root)
    commit_hash = hash_result.output.strip() if hash_result.success else "unknown"

    return ToolResult(
        success=True,
        output=f"提交成功: {commit_hash}\n{commit_result.output}",
    )


async def _run_git(args: list[str], workspace_root: str) -> ToolResult:
    """执行 git 子命令的内部辅助函数。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_root or None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=settings.TOOL_COMMAND_DEFAULT_TIMEOUT
        )
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            return ToolResult(
                success=False,
                output=stdout_text,
                error=stderr_text or f"git {' '.join(args)} 退出码 {proc.returncode}",
            )
        return ToolResult(success=True, output=stdout_text)
    except asyncio.TimeoutError:
        return ToolResult(success=False, error=f"git {' '.join(args)} 超时")
    except FileNotFoundError:
        return ToolResult(success=False, error="git 未安装")
    except OSError as e:
        return ToolResult(success=False, error=f"git 执行失败: {e}")


# ============================================================
# 工具注册表
# ============================================================
# 所有工具注册到此字典。execute_tool 根据 tool_name 路由到对应 handler。
# 注意：write_file / run_command / git_commit 的 handler 返回 requires_confirmation=True，
# 真正执行由 _do_* 函数在 confirm_tool 中完成。

_TOOL_REGISTRY: Dict[str, ToolHandler] = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_command": tool_run_command,
    "grep_search": tool_grep_search,
    "git_commit": tool_git_commit,
}

# 确认后真正执行的 handler 映射（key 与 _TOOL_REGISTRY 一致）
_CONFIRM_HANDLERS: Dict[str, ToolHandler] = {
    "write_file": _do_write_file,
    "run_command": _do_run_command,
    "git_commit": _do_git_commit,
}


def get_registered_tools() -> list[str]:
    """返回已注册的工具名称列表（供调试/文档展示）"""
    return list(_TOOL_REGISTRY.keys())


# ============================================================
# 统一入口：execute_tool
# ============================================================

async def execute_tool(
    tool_call: ToolCall,
    workspace_root: str = "",
    session_id: str = "",
) -> ToolResult:
    """
    工具执行统一入口（S8 第 71-72 天核心）。

    流程：
      1. 从 _TOOL_REGISTRY 查找 handler，未找到返回错误。
      2. 调用 handler，捕获所有异常。
      3. 若 handler 返回 requires_confirmation=True：
         - 生成 confirmation_id，存入 PendingConfirmationStore。
         - 将 confirmation_id 填入 ToolResult 返回。
      4. 无论成功失败，写入审计日志（含执行耗时 duration_ms）。

    Args:
        tool_call:       工具调用请求。
        workspace_root:  工作区根目录，用于解析相对路径。
        session_id:      会话 ID，用于审计日志。

    Returns:
        ToolResult。
    """
    handler = _TOOL_REGISTRY.get(tool_call.tool_name)
    if handler is None:
        result = ToolResult(
            success=False,
            error=f"未知工具: {tool_call.tool_name}，可用工具: {get_registered_tools()}",
        )
        await write_audit_log(session_id, tool_call, result, duration_ms=0.0)
        return result

    start = time.perf_counter()
    try:
        result = await handler(tool_call.arguments, workspace_root, session_id)
    except Exception as e:
        # 兜底异常捕获，防止工具内部异常导致整个服务崩溃
        logger.error(
            f"[ToolRegistry] 工具执行异常: tool={tool_call.tool_name}, "
            f"session={session_id}, err={e}",
            exc_info=True,
        )
        result = ToolResult(
            success=False,
            error=f"工具执行异常: {type(e).__name__}: {e}",
        )
    duration_ms = (time.perf_counter() - start) * 1000

    # 需要用户确认：生成 confirmation_id
    if result.requires_confirmation and not result.confirmation_id:
        confirmation_id = await get_pending_confirmation_store().put(
            tool_call, workspace_root, session_id
        )
        result.confirmation_id = confirmation_id

    await write_audit_log(session_id, tool_call, result, duration_ms=duration_ms)
    return result


# ============================================================
# 确认执行：confirm_tool
# ============================================================

async def confirm_tool(
    confirmation_id: str,
    action: str,
    session_id: str = "",
) -> ToolResult:
    """
    处理用户对工具的确认操作（S8 关键接口）。

    流程：
      1. 从 PendingConfirmationStore 取出待确认记录（取出即删除，一次性凭证）。
      2. 记录不存在或已过期 → 返回错误。
      3. action='deny' → 返回拒绝结果。
      4. action='allow' → 调用 _CONFIRM_HANDLERS 中对应的 _do_* 函数真正执行。

    Args:
        confirmation_id: execute_tool 返回的确认凭证。
        action:          'allow' 或 'deny'。
        session_id:      会话 ID，用于审计日志。

    Returns:
        ToolResult（allow 时为真实执行结果；deny 时为拒绝结果）。
    """
    store = get_pending_confirmation_store()
    record = await store.pop(confirmation_id)
    if record is None:
        return ToolResult(
            success=False,
            error="确认凭证不存在或已过期",
        )

    # 校验 session_id 一致性（防止跨会话确认）
    if session_id and record.session_id != session_id:
        return ToolResult(
            success=False,
            error="确认凭证与会话不匹配",
        )

    if action == "deny":
        logger.info(
            f"[ToolRegistry] 用户拒绝执行: tool={record.tool_call.tool_name}, "
            f"session={record.session_id}"
        )
        result = ToolResult(
            success=False,
            error="用户拒绝执行该操作",
        )
        await write_audit_log(record.session_id, record.tool_call, result, duration_ms=0.0)
        return result

    # allow：调用真正的执行 handler
    confirm_handler = _CONFIRM_HANDLERS.get(record.tool_call.tool_name)
    if confirm_handler is None:
        result = ToolResult(
            success=False,
            error=f"工具 {record.tool_call.tool_name} 无需确认或不支持确认执行",
        )
        await write_audit_log(record.session_id, record.tool_call, result, duration_ms=0.0)
        return result

    start = time.perf_counter()
    try:
        result = await confirm_handler(
            record.tool_call.arguments,
            record.workspace_root,
            record.session_id,
        )
    except Exception as e:
        logger.error(
            f"[ToolRegistry] 确认后执行异常: tool={record.tool_call.tool_name}, "
            f"err={e}",
            exc_info=True,
        )
        result = ToolResult(
            success=False,
            error=f"执行异常: {type(e).__name__}: {e}",
        )
    duration_ms = (time.perf_counter() - start) * 1000

    await write_audit_log(record.session_id, record.tool_call, result, duration_ms=duration_ms)
    return result
