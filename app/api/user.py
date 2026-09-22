"""
用户相关接口（S3 第 21-22 天：限频与配额系统）

提供 /v1/user/usage 接口，返回用户当日 Token 消耗量与配额信息。
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from app.auth import get_current_user
from app.services.quota import get_quota_service

logger = logging.getLogger(__name__)

router = APIRouter()


class UsageResponse(BaseModel):
    """用量查询响应（S3 契约：snake_case 字段名）"""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    user_id: str
    used_tokens_today: int
    quota_limit_per_day: int
    percentage: float
    reset_at: str


@router.get("/user/usage", response_model=UsageResponse, response_model_by_alias=False)
async def get_usage(current_user: dict = Depends(get_current_user)):
    """
    查询当前用户今日 Token 用量（S3 第 21-22 天）。

    返回字段：
    - user_id: 用户标识（JWT sub 字段的哈希值）
    - used_tokens_today: 今日已消耗 Token 数
    - quota_limit_per_day: 每日配额上限
    - percentage: 已用比例（0.0 ~ 1.0）
    - reset_at: 配额重置时间（ISO 8601，UTC 午夜）
    """
    api_key = current_user.get("sub")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="未找到用户标识，请重新登录"
        )

    quota_service = get_quota_service()
    usage_data = quota_service.get_usage(api_key)

    logger.info(
        f"[UserUsage] 查询用量: user_id={usage_data['user_id'][:8]}..., "
        f"used={usage_data['used_tokens_today']}, "
        f"quota={usage_data['quota_limit_per_day']}, "
        f"percentage={usage_data['percentage']}"
    )

    return UsageResponse(
        user_id=usage_data["user_id"],
        used_tokens_today=usage_data["used_tokens_today"],
        quota_limit_per_day=usage_data["quota_limit_per_day"],
        percentage=usage_data["percentage"],
        reset_at=usage_data["reset_at"],
    )
