from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta
from ..auth import create_access_token, verify_token, validate_api_key, add_to_blacklist
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
security = HTTPBearer()

class TokenRequest(BaseModel):
    api_key: str
    model: Optional[str] = "glm-4.5-air"

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    获取当前用户
    """
    try:
        payload = verify_token(credentials.credentials)
        return payload
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/auth/api-key", response_model=TokenResponse)
async def get_api_key_token(request: TokenRequest):
    """
    使用API Key获取访问令牌
    """
    # 验证API Key
    if not validate_api_key(request.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API Key"
        )
    
    # 创建访问令牌
    access_token_expires = timedelta(minutes=60 * 24)  # 24小时
    access_token = create_access_token(
        data={"sub": request.api_key, "model": request.model},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=access_token_expires.seconds
    )

@router.post("/auth/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    用户登出，将Token加入黑名单
    """
    try:
        payload = verify_token(credentials.credentials)
        # 将Token加入黑名单，设置过期时间
        add_to_blacklist(
            credentials.credentials,
            timedelta(seconds=payload.get("exp", 0) - payload.get("iat", 0))
        )
        return {"message": "登出成功"}
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据"
        )
