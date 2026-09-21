# app/api/chat.py
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional
from app.models.schemas import ChatRequest, ChatChunk, ContextItem
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
from app.services.context_builder import ContextBuilder
from app.middlewares.request_id import get_request_id, REQUEST_ID_HEADER
from app.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# 每个用户的适配器实例缓存
_zai_adapters: Dict[str, ZAIAdapter] = {}

# ContextBuilder 单例（无状态配置，全局共享）
_context_builder: Optional[ContextBuilder] = None


def get_context_builder() -> ContextBuilder:
    """获取 ContextBuilder 单例。"""
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder()
    return _context_builder


def get_zai_adapter_for_user(api_key) -> ZAIAdapter:
    """为指定用户获取或创建ZAI适配器实例"""
    if api_key not in _zai_adapters:
        _zai_adapters[api_key] = ZAIAdapter(api_key=api_key)
    return _zai_adapters[api_key]


def _extract_system_content(messages: List[dict]) -> tuple:
    """
    从消息列表中提取系统提示词内容，并返回（system_content, non_system_messages）。
    若存在多条 system 消息，合并为一条（用换行分隔）。
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    system_content = "\n\n".join(system_parts) if system_parts else None
    return system_content, non_system


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

    支持上下文注入（第 15-16 天）：
    - 接收可选的 contexts 数组（@文件 / @选中代码 / 隐式上下文）。
    - ContextBuilder 将 contexts 格式化为结构化 XML 标签插入 System Prompt：
      <context_files><file path="main.py" lang="python">...</file></context_files>
    - 系统提示词 Token 纳入裁剪预算，保证总 Token 不超限。
    - 超长上下文按优先级丢弃：file/selection（用户主动 @）保留，implicit（自动附带）可截断。
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

    # session_id 需提前赋值，供上下文处理日志和会话管理共同使用
    session_id = request.session_id

    # ------------------------------------------------------------------
    # 上下文拼装（S2 第 15-16 天）：将 contexts 格式化为 XML 标签插入 System Prompt
    # 必须在获取会话历史之前构建，以便将系统提示词 Token 纳入裁剪预算
    # ------------------------------------------------------------------
    contexts: List[ContextItem] = request.contexts or []
    context_builder = get_context_builder()

    # 从请求消息中提取已有 system 内容，用于更准确地预估系统提示词 Token 占用
    request_messages_dicts = [m.model_dump() for m in request.messages]
    request_system_content, _ = _extract_system_content(request_messages_dicts)

    # 基于请求中的 system 内容 + contexts 构建系统提示词，获取 Token 占用
    # 用于在会话裁剪时预留预算（若会话历史中还有 system 消息，20% 余量可覆盖差异）
    system_prompt_preview = context_builder.build_system_prompt(
        contexts, existing_system_content=request_system_content
    )
    reserved_tokens = context_builder.last_system_tokens

    if contexts:
        type_counter: Dict[str, int] = {}
        for c in contexts:
            type_counter[c.type] = type_counter.get(c.type, 0) + 1
        logger.info(
            f"[Chat] 接收到上下文: session_id={session_id or '无'}, "
            f"条数={len(contexts)}, 类型分布={type_counter}, "
            f"系统提示词预留Token≈{reserved_tokens}"
        )

    # ------------------------------------------------------------------
    # 会话管理：将请求消息入库，并获取裁剪后的历史
    # 裁剪时已为系统提示词预留 reserved_tokens，保证总 Token 不超限
    # ------------------------------------------------------------------
    if session_id:
        session_service = get_session_service()
        # 若会话不存在则创建
        if not session_service.exists(session_id):
            session_service.create(session_id)
        # 将本次请求的消息追加到会话（通常是用户的最新提问）
        for msg in request.messages:
            session_service.append(session_id, msg.model_dump())
        # 获取裁剪后的历史（已扣除系统提示词预留 Token）
        llm_messages: List[dict] = session_service.get_history(
            session_id, reserved_tokens=reserved_tokens
        )
        if not llm_messages:
            # 极端情况：裁剪后为空（理论上不会发生），回退到请求消息
            llm_messages = request_messages_dicts
        logger.info(
            f"[Chat] 使用会话历史: session_id={session_id}, "
            f"发送给模型的消息条数={len(llm_messages)}, "
            f"历史Token≈{count_tokens(llm_messages)}, "
            f"系统提示词预留Token={reserved_tokens}"
        )
    else:
        # 无状态模式：直接使用请求中的消息
        llm_messages = request_messages_dicts

    # ------------------------------------------------------------------
    # 注入系统提示词：将上下文 XML 与已有 System 消息合并，置于消息列表首位
    # ------------------------------------------------------------------
    existing_system_content, non_system_messages = _extract_system_content(llm_messages)
    # 基于实际的已有系统内容（含会话历史中的 system）重新构建最终系统提示词
    final_system_prompt = context_builder.build_system_prompt(
        contexts, existing_system_content=existing_system_content
    )
    # 组装最终消息列表：[system] + 非 system 历史
    llm_messages = [{"role": "system", "content": final_system_prompt}] + non_system_messages

    logger.info(
        f"[Chat] 最终消息列表: 总条数={len(llm_messages)}, "
        f"system在首位={llm_messages[0]['role'] == 'system' if llm_messages else False}, "
        f"总Token≈{count_tokens(llm_messages)}"
    )
    # 打印系统提示词的前 500 字符，便于验证上下文是否正确注入（不打印全文避免日志过大）
    logger.info(
        f"[Chat] 系统提示词预览(前500字符): {final_system_prompt[:500]}"
    )

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
                    "Access-Control-Allow-Origin": "*",
                    # S2 第 19-20 天：SSE 响应头显式携带 X-Request-ID，
                    # 便于前端/插件在流式响应中关联后端日志（中间件亦会统一回写）
                    REQUEST_ID_HEADER: get_request_id(),
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
