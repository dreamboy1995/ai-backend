# routers/chat.py
import asyncio
import json
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from schemas import ChatRequest, ChatChunk, Choice, Delta, Usage

router = APIRouter()


async def mock_sse_generator(req: ChatRequest):
    """异步生成器：逐字产出 SSE 数据块"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    # 假装把用户最后一条消息当作要回显的内容
    reply_text = "你好，我是 Mock 模型，正在流式回复你。"

    # 1) 首帧：通常先推一个 role（OpenAI 的行为）
    first = ChatChunk(
        id=chat_id,
        model=req.model,
        choices=[Choice(index=0, delta=Delta(role="assistant", content=""))],
    )
    yield f"data: {first.model_dump_json()}\n\n"

    # 2) 逐字推送 content
    for ch in reply_text:
        chunk = ChatChunk(
            id=chat_id,
            model=req.model,
            choices=[Choice(index=0, delta=Delta(content=ch))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0.2)   # 每 200ms 吐一个字

    # 3) 结束帧：delta 为空 + finish_reason=stop
    end = ChatChunk(
        id=chat_id,
        model=req.model,
        choices=[Choice(index=0, delta=Delta(), finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=len(reply_text),
                    total_tokens=10 + len(reply_text)),
    )
    yield f"data: {end.model_dump_json()}\n\n"

    # 4) SSE 结束标志
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if req.stream:
        return StreamingResponse(
            mock_sse_generator(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，避免卡顿
            },
        )

    # 非流式：一次性返回
    content = "你好，我是 Mock 模型。"
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(content),
                  "total_tokens": 10 + len(content)},
    })