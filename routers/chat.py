# routers/chat.py
import asyncio
import json
import uuid
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional

from schemas import ChatRequest, ChatChunk, Choice, Delta, Usage
from adapters.zai_adapter import ZAIAdapter

router = APIRouter()

# 全局适配器实例
_zai_adapter: Optional[ZAIAdapter] = None

async def get_zai_adapter() -> ZAIAdapter:
    """获取ZAI适配器实例"""
    global _zai_adapter
    if _zai_adapter is None:
        api_key = os.getenv("ZAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="ZAI API key not configured"
            )
        _zai_adapter = ZAIAdapter(api_key=api_key)
    return _zai_adapter

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    zai_adapter: ZAIAdapter = Depends(get_zai_adapter)
):
    """
    聊天完成接口 - 代理到ZAI API
    """
    try:
        # 验证请求参数
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="Messages are required"
            )
        
        # 构建适配器需要的参数 - 将ChatMessage对象转换为字典
        adapter_params = {
            "messages": [message.model_dump() for message in request.messages],
            "model": request.model or "glm-4v",
            "temperature": request.temperature or 0.7,
            "stream": request.stream or True
        }
        
        # 如果是流式请求，返回StreamingResponse
        if request.stream:
            async def generate_stream():
                try:
                    async for chunk in zai_adapter.chat_completion(**adapter_params):
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
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Stream generation error: {e}")
                    error_chunk = ChatChunk(
                        id="",
                        object="chat.completion.chunk",
                        created=0,
                        model="",
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                            "error": {"message": str(e)}
                        }],
                        usage=None
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
            
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
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理适配器"""
    global _zai_adapter
    if _zai_adapter:
        await _zai_adapter.close()
        _zai_adapter = None
