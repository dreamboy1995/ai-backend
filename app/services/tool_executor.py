"""
S7/S8 工具执行器抽象基类（依赖注入设计）

对应 S7 风险预警：
  "模拟工具与真实工具的衔接：S7 的 Mock 工具返回假数据，但 S8 要换成真实的 MCP 工具。
   需要在代码层面做好依赖注入（Dependency Injection），定义一个 ToolExecutor 抽象基类，
   S7 注入 MockToolExecutor，S8 只需替换为 MCPToolExecutor，上层循环代码无需修改。"

本次改造（S8 确认链路落地）：
  execute() 返回类型从 str 改为 ToolResult，让上层 react_loop 能拿到
  requires_confirmation / confirmation_id 等标志，从而实现「写入前暂停、
  前端弹确认浮层、确认后恢复执行」的完整链路。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.models.tool import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """工具执行失败"""
    pass


class ToolExecutor(ABC):
    """
    工具执行器抽象基类。

    所有具体执行器（Mock / MCP / ...）必须实现 execute 方法。
    ReAct 循环（react_loop.py，S7 第 67-68 天）通过依赖注入接收
    ToolExecutor 实例，不关心底层是 Mock 还是真实 MCP。
    """

    @abstractmethod
    async def execute(
        self,
        tool: str,
        params: Dict[str, Any],
        *,
        workspace_root: str = "",
        session_id: str = "",
    ) -> ToolResult:
        """
        执行一个工具调用。

        Args:
            tool:           工具名称（如 "write_file" / "run_command" / "read_file"）
            params:         工具参数字典
            workspace_root:  工作区根路径（供 MCP 工具解析相对路径）
            session_id:      会话 ID（供审计日志）

        Returns:
            ToolResult，包含 success / output / error / requires_confirmation 等。

        Raises:
            ToolExecutionError: 工具执行失败且不应由调用方自动处理。
        """
        ...


class MockToolExecutor(ToolExecutor):
    """
    S7 模拟工具执行器（开发联调用）。

    不真正读写文件或执行命令，直接返回 success=True 的 ToolResult。
    默认行为：不触发确认流程（用于快速联调循环逻辑）。
    若需要验证确认链路，可设置 confirm_tools 指定哪些工具模拟确认。
    """

    def __init__(self, confirm_tools=None):
        # 默认不卡确认——S7 Mock 是快速联调循环逻辑用的，
        # 只有显式配置 confirm_tools 才模拟 requires_confirmation=True
        self._confirm_tools = confirm_tools or set()

    async def execute(
        self,
        tool: str,
        params: Dict[str, Any],
        *,
        workspace_root: str = "",
        session_id: str = "",
    ) -> ToolResult:
        logger.info(
            f"[MockToolExecutor] 模拟执行工具: tool={tool}, params={params}"
        )

        # 如果该工具在 confirm_tools 集合中，模拟确认流程
        if tool in self._confirm_tools:
            prompt = self._build_prompt(tool, params)
            output = self._build_mock_preview(tool, params)
            import uuid
            confirmation_id = str(uuid.uuid4())
            return ToolResult(
                success=True,
                output=output,
                requires_confirmation=True,
                confirmation_prompt=prompt,
                confirmation_id=confirmation_id,
            )

        # 默认：直接返回成功，不触发确认
        output_map = {
            "write_file": f"[Mock] 文件 {params.get('path', params.get('file_path', ''))} 写入成功（模拟）",
            "run_command": f"[Mock] 命令 '{params.get('cmd', '')}' 执行成功，退出码 0（模拟）",
            "read_file": f"[Mock] 读取文件 {params.get('file_path', params.get('path', ''))} 成功（模拟）",
            "grep_search": f"[Mock] 搜索 '{params.get('pattern', '')}' 找到 3 个结果（模拟）",
            "git_commit": "[Mock] git commit 成功（模拟）",
            "ask_user": f"[PENDING_QUESTION] {params.get('question', '')}",
        }
        msg = output_map.get(tool, f"[Mock] 工具 {tool} 执行成功（模拟）")
        logger.info(f"[MockToolExecutor] 返回: {msg}")
        return ToolResult(success=True, output=msg)

    @staticmethod
    def _build_prompt(tool: str, params: Dict[str, Any]) -> str:
        if tool == "write_file":
            path = params.get("path") or params.get("file_path") or "<unknown>"
            content_len = len(params.get("content", ""))
            return f"即将覆盖写入文件 {path}（{content_len} 字符），是否继续？"
        elif tool == "run_command":
            return f"即将执行命令：\n{params.get('cmd', '')}\n\n是否允许执行？"
        elif tool == "git_commit":
            return f"即将执行 git commit，信息：{params.get('message', '(未提供)')}\n\n是否继续？"
        return f"即将执行 {tool}，是否继续？"

    @staticmethod
    def _build_mock_preview(tool: str, params: Dict[str, Any]) -> str:
        if tool == "write_file":
            path = params.get("path") or params.get("file_path") or "<unknown>"
            content = params.get("content", "")
            return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(content.splitlines())} @@\n+ {content[:200]}..."
        elif tool == "run_command":
            return f"[待确认命令]\n{params.get('cmd', '')}"
        elif tool == "git_commit":
            return "[Mock] git status --short 输出（模拟）\n M src/index.js"
        return ""


class MCPToolExecutor(ToolExecutor):
    """
    S8 真实工具执行器（MCP 协议适配层）。

    将 S7 的 ToolExecutor 抽象接口桥接到 S8 的 tool_registry.execute_tool。
    react_loop 调用 execute(tool, params) 时，本类：
      1. 将 (tool, params) 转换为 ToolCall。
      2. 调用 tool_registry.execute_tool（内部生成 confirmation_id 等）。
      3. 直接返回 ToolResult（可能带有 requires_confirmation=True）。
    """

    def __init__(self, workspace_root: str = "", session_id: str = ""):
        self.workspace_root = workspace_root
        self.session_id = session_id

    async def execute(
        self,
        tool: str,
        params: Dict[str, Any],
        *,
        workspace_root: str = "",
        session_id: str = "",
    ) -> ToolResult:
        from app.services.tool_registry import execute_tool as registry_execute

        # 优先用调用方传入的 workspace_root / session_id，
        # 否则 fallback 到实例初始化时的值（兼容旧调用）
        ws = workspace_root or self.workspace_root
        sid = session_id or self.session_id

        logger.info(
            f"[MCPToolExecutor] 执行工具: tool={tool}, params={params}"
        )

        tool_call = ToolCall(tool_name=tool, arguments=params)
        return await registry_execute(
            tool_call=tool_call,
            workspace_root=ws,
            session_id=sid,
        )


# 全局默认执行器（S7 为 Mock，应用启动时可切换为 MCP）
_default_executor: ToolExecutor = MockToolExecutor()


def get_default_tool_executor() -> ToolExecutor:
    """获取默认工具执行器（MockToolExecutor，可在 lifespan 中切换为 MCP）"""
    return _default_executor


def set_default_tool_executor(executor: ToolExecutor) -> None:
    """
    设置默认工具执行器。

    应用启动时（lifespan.py）调用此方法将默认执行器切换为 MCPToolExecutor，
    使 react_loop 使用真实工具（含确认链路）。
    """
    global _default_executor
    _default_executor = executor
    logger.info(f"[ToolExecutor] 默认执行器已切换为: {type(executor).__name__}")
