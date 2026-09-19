from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import settings

router = APIRouter(tags=["调试"])


class Item(BaseModel):
    name: str
    age: int


@router.get("/config-check")
async def config_check():
    """配置检查接口"""
    return {
        "port": settings.PORT,
        "zai_api_key_configured": bool(settings.ZAI_API_KEY),
        "jwt_secret_configured": bool(settings.JWT_SECRET),
        "environment": settings.ENVIRONMENT,
    }


@router.get("/test-error")
async def test_error():
    """测试业务异常"""
    raise HTTPException(status_code=400, detail="这是一个测试业务异常")


@router.post("/test-validation")
async def test_validation(item: Item):
    """测试参数校验"""
    return {"received": item}


@router.get("/test-500")
async def test_500():
    """测试未处理异常"""
    raise RuntimeError("这是一个测试未处理异常")
