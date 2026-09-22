import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.auth import verify_token

logger = logging.getLogger(__name__)

# 精确路径匹配，避免前缀匹配被绕过
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}


async def auth_middleware(request: Request, call_next):
    """JWT 认证中间件"""
    path = request.url.path

    # 公开路径或认证接口无需验证
    if path in PUBLIC_PATHS or path.startswith("/auth/"):
        return await call_next(request)

    # 检查 Authorization 头
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "code": 401,
                "message": "缺少认证令牌",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]

    try:
        payload = verify_token(token)
        # S3 第 21-22 天：将解码后的 JWT payload 存入 request.state，
        # 供限频中间件（rate_limiter）读取 user_id（sub 字段）。
        # 限频中间件在本中间件之后执行（注册顺序更内层）。
        request.state.user_payload = payload
    except Exception as e:
        # 中间件内异常需自行处理，全局异常处理器对中间件不生效
        logger.warning(f"Token verification failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "code": 401,
                "message": str(e.detail) if hasattr(e, "detail") else "无效的认证令牌",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)
