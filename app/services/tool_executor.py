"""
S7 第 61-62 天：工具执行器抽象基类（依赖注入设计）

对应 S7 风险预警：
  "模拟工具与真实工具的衔接：S7 的 Mock 工具返回假数据，但 S8 要换成真实的 MCP 工具。
   需要在代码层面做好依赖注入（Dependency Injection），定义一个 ToolExecutor 抽象基类，
   S7 注入 MockToolExecutor，S8 只需替换为 MCPToolExecutor，上层循环代码无需修改。"

设计：
  - ToolExecutor: 抽象基类，定义 execute(tool, params) 接口。
  - MockToolExecutor: S7 使用，返回硬编码的成功消息，不真正读写文件。
  - MCPToolExecutor: S8 实现，调用真实 MCP 工具。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
    async def execute(self, tool: str, params: Dict[str, Any]) -> str:
        """
        执行一个工具调用。

        Args:
            tool:   工具名称（如 "write_file" / "run_command" / "search_code" / "ask_user"）
            params: 工具参数字典

        Returns:
            工具执行结果的文本描述（将作为 observation 写入步骤）

        Raises:
            ToolExecutionError: 工具执行失败，调用方应将步骤标记为 failed
        """
        ...


class MockToolExecutor(ToolExecutor):
    """
    S7 模拟工具执行器。

    不真正读写文件或执行命令，而是根据工具名返回硬编码的成功消息，
    用于验证 ReAct 循环逻辑的通畅性。
    """

    async def execute(self, tool: str, params: Dict[str, Any]) -> str:
        logger.info(
            f"[MockToolExecutor] 模拟执行工具: tool={tool}, params={params}"
        )
        # 模拟不同工具的返回消息
        mock_messages = {
            "write_file": f"文件 {params.get('path', '<unknown>')} 创建成功（模拟）",
            "run_command": f"命令 '{params.get('cmd', '')}' 执行成功，退出码 0（模拟）",
            "search_code": f"搜索 '{params.get('query', '')}' 找到 3 个结果（模拟）",
            "ask_user": f"[PENDING_QUESTION] {params.get('question', '')}",
        }
        msg = mock_messages.get(tool, f"工具 {tool} 执行成功（模拟）")
        logger.info(f"[MockToolExecutor] 返回 observation: {msg}")
        return msg


class MCPToolExecutor(ToolExecutor):
    """
    S8 真实工具执行器（MCP 协议适配层）。

    将 S7 的 ToolExecutor 抽象接口桥接到 S8 的 tool_registry.execute_tool。
    react_loop 调用 execute(tool, params) 时，本类：
      1. 将 (tool, params) 转换为 ToolCall。
      2. 调用 tool_registry.execute_tool。
      3. 将 ToolResult 转换为字符串 observation 返回给 react_loop。

    注意：write_file / run_command / git_commit 等需要用户确认的工具，
    在 S8 第 71-72 天阶段由插件端通过 /v1/tool/confirm 处理。
    MCPToolExecutor 在此阶段对 requires_confirmation=True 的结果
    返回提示文本，告知用户需要在插件端确认。
    """

    def __init__(self, workspace_root: str = "", session_id: str = ""):
        self.workspace_root = workspace_root
        self.session_id = session_id

    async def execute(self, tool: str, params: Dict[str, Any]) -> str:
        from app.models.tool import ToolCall
        from app.services.tool_registry import execute_tool as registry_execute

        logger.info(
            f"[MCPToolExecutor] 执行工具: tool={tool}, params={params}"
        )

        tool_call = ToolCall(tool_name=tool, arguments=params)
        result = await registry_execute(
            tool_call=tool_call,
            workspace_root=self.workspace_root,
            session_id=self.session_id,
        )

        if not result.success:
            # 失败：抛出 ToolExecutionError，让 react_loop 标记步骤 failed
            raise ToolExecutionError(result.error or "工具执行失败")

        if result.requires_confirmation:
            # 需要用户确认：返回提示文本（S8 第 71-72 天阶段）
            # 完整的确认流程由插件端 /v1/tool/confirm 处理
            obs = f"[需要用户确认] {result.confirmation_prompt or ''}\n{result.output}"
            if result.confirmation_id:
                obs += f"\n确认凭证: {result.confirmation_id}"
            return obs

        # 成功执行
        return result.output or f"工具 {tool} 执行成功"


# 全局默认执行器（S7 为 Mock，S8 替换为 MCP）
_default_executor: ToolExecutor = MockToolExecutor()


def get_default_tool_executor() -> ToolExecutor:
    """获取默认工具执行器（S7 为 MockToolExecutor）"""
    return _default_executor


def set_default_tool_executor(executor: ToolExecutor) -> None:
    """
    设置默认工具执行器。

    S8 启动时可调用此方法将默认执行器切换为 MCPToolExecutor，
    使 react_loop 使用真实工具。
    """
    global _default_executor
    _default_executor = executor
    logger.info(f"[ToolExecutor] 默认执行器已切换为: {type(executor).__name__}")
