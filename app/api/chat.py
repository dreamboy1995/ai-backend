# app/api/chat.py
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, List
from app.models.schemas import ChatRequest, ChatChunk
from app.services.llm import (
    ZAIAdapter,
    ZAIAdapterError,
    ZAIRateLimitError,
    ZAITokenTooLongError,
    ZAITimeoutError,
    ZAINetworkError,
    ZAIServiceError,
)
from app.services.session import get_session_service, count_tokens
from app.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# 每个用户的适配器实例缓存
_zai_adapters: Dict[str, ZAIAdapter] = {}


def get_zai_adapter_for_user(api_key) -> ZAIAdapter:
    """为指定用户获取或创建ZAI适配器实例"""
    if api_key not in _zai_adapters:
        _zai_adapters[api_key] = ZAIAdapter(api_key=api_key)
    return _zai_adapters[api_key]


@router.post("/chat/completions")
async def chat_completions(
        request: ChatRequest,
        current_user: dict = Depends(get_current_user)
):
    """
    聊天完成接口 - 代理到ZAI API

    支持会话管理（第 11-12 天）：
    - 若携带 session_id，后端会维护多轮对话历史，并使用滑动窗口裁剪后发送给模型。
    - 若未携带 session_id，则按请求中的 messages 直接透传（无状态模式）。
    """
    # 从JWT token中提取用户提交的API key
    api_key = current_user.get("sub")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="未找到API Key，请重新登录"
        )

    zai_adapter = get_zai_adapter_for_user(api_key)

    # 验证请求参数
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages are required"
        )

    # ------------------------------------------------------------------
    # 会话管理：将请求消息入库，并获取裁剪后的历史
    # ------------------------------------------------------------------
    session_id = request.session_id
    if session_id:
        session_service = get_session_service()
        # 若会话不存在则创建
        if not session_service.exists(session_id):
            session_service.create(session_id)
        # 将本次请求的消息追加到会话（通常是用户的最新提问）
        for msg in request.messages:
            session_service.append(session_id, msg.model_dump())
        # 获取裁剪后的历史作为发送给模型的消息
        llm_messages: List[dict] = session_service.get_history(session_id)
        if not llm_messages:
            # 极端情况：裁剪后为空（理论上不会发生），回退到请求消息
            llm_messages = [m.model_dump() for m in request.messages]
        logger.info(
            f"[Chat] 使用会话历史: session_id={session_id}, "
            f"发送给模型的消息条数={len(llm_messages)}, "
            f"Token≈{count_tokens(llm_messages)}"
        )
    else:
        # 无状态模式：直接使用请求中的消息
        llm_messages = [m.model_dump() for m in request.messages]

    try:
        # 构建适配器需要的参数
        adapter_params = {
            "messages": llm_messages,
            "model": request.model or "glm-4.5-air",
            "temperature": request.temperature or 0.7,
            "stream": request.stream or True
        }

        # 如果是流式请求，返回StreamingResponse
        if request.stream:
            async def generate_stream():
                assistant_content_parts: List[str] = []
                try:
                    async for chunk in zai_adapter.chat_completion(**adapter_params):
                        # 收集助手回复内容，用于后续写入会话历史
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                assistant_content_parts.append(content)

                        # 转换为ChatChunk格式
                        chat_chunk = ChatChunk(
                            id=chunk.get("id", ""),
                            object=chunk.get("object", ""),
                            created=chunk.get("created", 0),
                            model=chunk.get("model", ""),
                            choices=chunk.get("choices", []),
                            usage=chunk.get("usage")
                        )
                        yield f"data: {chat_chunk.model_dump_json()}\n\n"
                    # 流正常结束，发送 [DONE] 标记（OpenAI 标准）
                    yield "data: [DONE]\n\n"

                    # 流正常结束后，将助手完整回复写入会话历史
                    if session_id and assistant_content_parts:
                        full_content = "".join(assistant_content_parts)
                        session_service.append(
                            session_id,
                            {"role": "assistant", "content": full_content}
                        )
                        logger.info(
                            f"[Chat] 助手回复已写入会话: session_id={session_id}, "
                            f"内容长度={len(full_content)}"
                        )
                except ZAIRateLimitError as e:
                    logger.warning(f"Rate limit error: {e.message}")
                    yield f'data: {{"error": {{"code": {e.code}, "msg": "{e.message}"}}}}\n\n'
                except ZAITokenTooLongError as e:
                    logger.warning(f"Token too long error: {e.message}")
                    yield f'data: {{"error": {{"code": {e.code}, "msg": "{e.message}"}}}}\n\n'
                except ZAITimeoutError as e:
                    logger.warning(f"Timeout error: {e.message}")
                    yield f'data: {{"error": {{"code": {e.code}, "msg": "{e.message}"}}}}\n\n'
                except ZAINetworkError as e:
                    logger.warning(f"Network error: {e.message}")
                    yield f'data: {{"error": {{"code": {e.code}, "msg": "{e.message}"}}}}\n\n'
                except ZAIServiceError as e:
                    logger.warning(f"Service error: {e.message}")
                    yield f'data: {{"error": {{"code": {e.code}, "msg": "{e.message}"}}}}\n\n'
                except ZAIAdapterError as e:
                    logger.warning(f"Adapter error: {e.message}")
                    yield f'data: {{"error": {{"code": {e.code}, "msg": "{e.message}"}}}}\n\n'
                except Exception as e:
                    logger.error(f"Stream generation error: {e}", exc_info=True)
                    yield f'data: {{"error": {{"code": 10201, "msg": "服务器内部错误"}}}}\n\n'

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            # 非流式请求（暂时不支持）
            raise HTTPException(
                status_code=400,
                detail="Non-streaming requests are not supported yet"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理适配器"""
    global _zai_adapters
    for adapter in _zai_adapters.values():
        await adapter.close()
    _zai_adapters.clear()
