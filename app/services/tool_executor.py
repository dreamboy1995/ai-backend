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
from typing import Any, Dict

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


# 全局默认执行器（S7 为 Mock，S8 替换为 MCP）
_default_executor: ToolExecutor = MockToolExecutor()


def get_default_tool_executor() -> ToolExecutor:
    """获取默认工具执行器（S7 为 MockToolExecutor）"""
    return _default_executor
