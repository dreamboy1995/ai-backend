"""
请求链路追踪中间件（S2 第 19-20 天：异常处理、联调与 S2 Demo 准备）

核心能力：
- 每个请求带上 X-Request-ID，前后端打通，方便排查问题。
- 优先复用客户端传入的 X-Request-ID；未传则后端生成 uuid4。
- 将 request_id 存入 contextvars，使业务代码与日志均能获取（async 安全）。
- 响应头回写 X-Request-ID，便于前端/插件关联日志。

设计要点：
- 使用 contextvars.ContextVar 而非线程局部变量，兼容 asyncio 的并发模型。
- 配合 logging Filter，让所有日志自动带上 request_id 字段。
- 客户端传入的 ID 做长度限制与去空白，防止恶意超长 header 污染日志。
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import Request

logger = logging.getLogger(__name__)

# 请求头 / 响应头名称
REQUEST_ID_HEADER = "X-Request-ID"

# 当前请求的 request_id（async 安全）。
# 默认值 "-" 表示不在请求上下文中（如启动期日志）。
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

# 客户端传入的 X-Request-ID 最大长度，超出则截断，防止日志污染
_MAX_CLIENT_REQUEST_ID_LEN = 128

# 已安装的日志过滤器单例（供 logging_config 引用）
_filter_instance: Optional["_RequestIdFilter"] = None


def get_request_id() -> str:
    """获取当前请求的 request_id（供业务代码/日志使用）。不在请求上下文时返回 "-"。"""
    return request_id_ctx.get()


def get_request_id_filter() -> "_RequestIdFilter":
    """获取日志过滤器单例（供 logging_config 安装到 handler）。"""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = _RequestIdFilter()
    return _filter_instance


class _RequestIdFilter(logging.Filter):
    """
    日志过滤器：将当前 contextvars 中的 request_id 注入到每条日志记录，
    使日志格式中的 %(request_id)s 字段可用。

    不在请求上下文时（如启动期、后台清理任务），request_id 为 "-"。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def _generate_request_id() -> str:
    return uuid.uuid4().hex


def _sanitize_client_id(raw: Optional[str]) -> Optional[str]:
    """清洗客户端传入的 X-Request-ID：去空白、限长。不合法返回 None。"""
    if not raw:
        return None
    val = raw.strip()
    if not val:
        return None
    if len(val) > _MAX_CLIENT_REQUEST_ID_LEN:
        val = val[:_MAX_CLIENT_REQUEST_ID_LEN]
    return val


async def request_id_middleware(request: Request, call_next):
    """
    X-Request-ID 链路追踪中间件。

    流程：
    1. 优先复用客户端传入的 X-Request-ID；未传则生成 uuid4。
    2. 存入 contextvars，下游中间件（auth 等）、路由、SSE 流式生成器
       均可在同一请求上下文中读取。
    3. 响应头回写 X-Request-ID，便于前端/插件关联日志。

    说明：BaseHTTPMiddleware 会在子任务中（复制上下文）运行下游应用，
    因此即使本中间件在 call_next 返回后 reset 自己的 contextvar，
    下游（含 StreamingResponse 的 body 生成器）仍能在其复制的上下文中读到 rid。
    """
    client_id = _sanitize_client_id(request.headers.get(REQUEST_ID_HEADER))
    rid = client_id or _generate_request_id()

    token = request_id_ctx.set(rid)
    try:
        response = await call_next(request)
        # 回写响应头，覆盖任何下游设置的值（保持与当前请求一致）
        response.headers[REQUEST_ID_HEADER] = rid
        return response
    finally:
        request_id_ctx.reset(token)
