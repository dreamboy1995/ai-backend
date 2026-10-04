"""
S8 第 75-76 天：终端日志流式接口（WebSocket）

对应 Sprint_8.md「关键接口/数据结构变更（S8 新增）」：

  // 使用 WebSocket 路径
  ws://localhost:3000/v1/agent/stream/{session_id}
  // 消息格式
  interface StreamMessage {
    type: 'stdout' | 'stderr' | 'system';
    content: string;  // 可能包含 ANSI 颜色码
    timestamp: string;
  }

设计要点：
  - 前端在调用 POST /v1/tool/confirm 之前先建立 WebSocket 连接，
    以接收 CommandExecutor 执行命令时实时推送的 stdout/stderr/system 消息。
  - 一个 session 可有多个并发订阅者（多个浏览器标签页打开同一会话）。
  - 连接断开时自动 unsubscribe，避免内存泄漏。
  - CommandExecutor 在没有订阅者时仍能正常执行命令，
    stdout/stderr 仍会在 ToolResult.output 中返回（降级为非流式）。
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.command_executor import get_stream_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/agent/stream/{session_id}")
async def command_stream(ws: WebSocket, session_id: str):
    """
    命令执行实时日志流（WebSocket）。

    连接建立后订阅 session_id 对应的命令输出流，
    将 StreamMessage 以 JSON 推送给前端。

    消息示例：
      {"type":"system","content":"$ echo hello\\n[PID=1234, 超时=60s]",
       "timestamp":"2026-10-04T10:00:00.000Z"}
      {"type":"stdout","content":"hello\\n","timestamp":"2026-10-04T10:00:00.123Z"}
      {"type":"system","content":"[进程退出码=0]","timestamp":"2026-10-04T10:00:00.456Z"}

    错误处理：
      - 订阅队列满时由 CommandStreamManager.publish 自动丢弃最旧消息。
      - WebSocket 异常断开时 unsubscribe，不影响其他订阅者。
    """
    await ws.accept()
    manager = get_stream_manager()
    queue = await manager.subscribe(session_id)

    logger.info(f"[Stream] WebSocket 已连接: session={session_id}")

    try:
        while True:
            try:
                message = await queue.get()
                await ws.send_text(message.model_dump_json())
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        # 客户端正常断开
        pass
    except Exception as e:
        logger.error(
            f"[Stream] WebSocket 异常: session={session_id}, err={e}",
            exc_info=True,
        )
    finally:
        await manager.unsubscribe(session_id, queue)
        logger.info(f"[Stream] WebSocket 已断开: session={session_id}")
