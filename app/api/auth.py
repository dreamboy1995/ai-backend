from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta
from app.auth import create_access_token, verify_token, validate_api_key, add_to_blacklist
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

@router.get("/auth/validate")
async def validate_token(request: Request):
    """
    验证当前Token是否有效
    """
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"valid": False, "detail": "缺少认证令牌"}

    token = auth_header[7:]
    try:
        payload = verify_token(token)
        return {"valid": True, "payload": payload}
    except HTTPException as e:
        return {"valid": False, "detail": e.detail}
    except Exception as e:
        return {"valid": False, "detail": str(e)}

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
