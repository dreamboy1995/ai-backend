from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import logging

logger = logging.getLogger(__name__)

def request_validation_exception_handler(request: Request, exc: HTTPException):
    """请求参数校验异常处理器"""
    logger.warning(f"Request validation error: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": 422,
            "message": "请求参数校验失败",
            "detail": jsonable_encoder(exc.detail),
            "path": request.url.path,
        },
    )

def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path,
        },
    )

def unhandled_exception_handler(request: Request, exc: Exception):
    """未处理异常处理器"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 500,
            "message": "服务器内部错误",
            "path": request.url.path,
        },
    )
