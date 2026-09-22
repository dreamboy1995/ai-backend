"""
模型列表接口（S3 第 23-24 天：设置面板 & 模型切换）

提供 /v1/models 接口，返回可用模型列表（含展示名称和上下文长度）。
插件端设置面板从此接口获取模型下拉框选项。
"""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth import get_current_user
from app.services.llm import AdapterFactory

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelItem(BaseModel):
    """单个模型信息"""
    id: str
    label: str
    context_window: int


@router.get("/models", response_model=list[ModelItem])
async def list_models(current_user: dict = Depends(get_current_user)):
    """
    获取可用模型列表（S3 第 23-24 天）。

    返回格式：[{id, label, context_window}, ...]
    示例：[{id:'deepseek-v3', label:'DeepSeek V3', context_window:64000}]

    插件端设置面板从此接口获取模型选择下拉框选项。
    所有已注册模型均会返回，无论对应厂商 API Key 是否已配置。
    """
    models = AdapterFactory.list_models()
    result = [ModelItem(**m.to_dict()) for m in models]

    logger.info(
        f"[Models] 返回模型列表: 数量={len(result)}, "
        f"模型IDs={[m.id for m in result]}"
    )
    return result
