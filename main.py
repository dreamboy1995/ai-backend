"""
Settings 会从 .env 读取 PORT、DEEPSEEK_API_KEY、JWT_SECRET。

@app.exception_handler(...) 就是 FastAPI 里的“全局异常过滤器”。

RequestValidationError 处理参数校验错误。

HTTPException 处理你主动抛出的业务异常。

Exception 处理未捕获异常，避免把堆栈直接暴露给客户端。

docs_url="/docs" 就是 Swagger 文档路由。
"""

import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from routers.chat import router as chat_router


# =========================
# 1. 读取 .env 配置
# =========================
class Settings(BaseSettings):
    PORT: int = 3000
    ZAI_API_KEY: str = "xxx"
    JWT_SECRET: str = "xxx"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# =========================
# 2. 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai-backend")


# =========================
# 3. 创建 FastAPI 应用
# =========================
app = FastAPI(
    title="AI Backend",
    description="FastAPI + ZAI 后端服务",
    version="0.1.0",
    docs_url="/docs",          # Swagger UI 地址
    redoc_url="/redoc",        # ReDoc 地址
    openapi_url="/openapi.json" # OpenAPI JSON 地址
)
app.include_router(chat_router)


# =========================
# 4. 全局异常过滤器 / 异常处理器
# =========================

# 4.1 请求参数校验失败
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": 422,
            "message": "请求参数校验失败",
            "detail": jsonable_encoder(exc.errors()),
            "path": request.url.path,
        },
    )


# 4.2 主动抛出的 HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path,
        },
    )


# 4.3 未处理异常，统一返回 500
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理异常: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 500,
            "message": "服务器内部错误",
            "path": request.url.path,
        },
    )


# =========================
# 5. 示例路由
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
# 6. 本地启动入口
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,
    )