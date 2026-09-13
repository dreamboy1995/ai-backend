"""
Settings 会从 .env 读取 PORT、ZAI_API_KEY、JWT_SECRET。

@app.exception_handler(...) 就是 FastAPI 里的"全局异常过滤器"。

RequestValidationError 处理参数校验错误。

HTTPException 处理你主动抛出的业务异常。

Exception 处理未捕获异常，避免把堆栈直接暴露给客户端。

docs_url="/docs" 就是 Swagger 文档路由。
"""

import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from routers.chat import router as chat_router
from routers.auth import router as auth_router
from exception_handlers import (
    request_validation_exception_handler,
    http_exception_handler,
    unhandled_exception_handler
)
from auth import verify_token


# =========================
# 1. 加载 .env 文件
# =========================
# 在程序启动时加载 .env 文件
load_dotenv()


# =========================
# 2. 读取 .env 配置
# =========================
class Settings(BaseSettings):
    PORT: int = 3000
    ZAI_API_KEY: str = "xxx"
    JWT_SECRET: str = "xxx"
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# =========================
# 3. 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai-backend")


# =========================
# 4. 应用生命周期管理
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("Starting AI Backend application")

    # 检查必要的环境变量
    required_env_vars = ["ZAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")

    yield

    # 关闭时执行
    logger.info("Shutting down AI Backend application")


# =========================
# 5. 创建 FastAPI 应用
# =========================
app = FastAPI(
    title="AI Backend",
    description="FastAPI + ZAI 后端服务",
    version="0.1.0",
    docs_url="/docs",          # Swagger UI 地址
    redoc_url="/redoc",        # ReDoc 地址
    openapi_url="/openapi.json", # OpenAPI JSON 地址
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router, prefix="/v1", tags=["聊天"])
app.include_router(auth_router, tags=["认证"])


# =========================
# 6. JWT中间件
# =========================
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # 允许健康检查接口、文档接口和认证接口无需认证
    if (request.url.path == "/health" or
            request.url.path.startswith("/docs") or
            request.url.path.startswith("/auth/")):
        response = await call_next(request)
        return response

    # 检查Authorization头
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]  # 去掉"Bearer "前缀

    try:
        # 验证Token
        verify_token(token)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    response = await call_next(request)
    return response


# =========================
# 7. 全局异常过滤器 / 异常处理器
# =========================

# 5.1 请求参数校验失败
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)

# 5.2 主动抛出的 HTTPException
app.add_exception_handler(HTTPException, http_exception_handler)

# 5.3 未处理异常，统一返回 500
app.add_exception_handler(Exception, unhandled_exception_handler)


# =========================
# 8. 示例路由
# =========================
class Item(BaseModel):
    name: str
    age: int


@app.get("/", tags=["默认"])
async def root():
    return {"message": "AI Backend is running"}


@app.get("/health", tags=["健康检查"])
async def health():
    return {
        "status": "ok",
        "port": settings.PORT,
    }


@app.get("/config-check", tags=["配置"])
async def config_check():
    # 注意：不要返回真实密钥，这里只检查是否已配置
    return {
        "port": settings.PORT,
        "zai_api_key_configured": settings.ZAI_API_KEY not in ("", "xxx"),
        "jwt_secret_configured": settings.JWT_SECRET not in ("", "xxx"),
        "environment": settings.ENVIRONMENT,
    }


@app.get("/test-error", tags=["测试异常"])
async def test_error():
    raise HTTPException(status_code=400, detail="这是一个测试业务异常")


@app.post("/test-validation", tags=["测试异常"])
async def test_validation(item: Item):
    return {"received": item}


@app.get("/test-500", tags=["测试异常"])
async def test_500():
    raise RuntimeError("这是一个测试未处理异常")


# =========================
# 9. 本地启动入口
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,
    )
