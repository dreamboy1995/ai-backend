from jose import jwt, JWTError, ExpiredSignatureError
from datetime import datetime, timedelta
from typing import Optional, Dict
import uuid
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# 内存存储Token黑名单
token_blacklist: Dict[str, datetime] = {}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建JWT访问令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=60)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    to_encode["jti"] = str(uuid.uuid4())  # 添加JWT ID用于黑名单

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """
    验证JWT令牌
    """
    try:
        # 检查Token是否在黑名单中
        if token in token_blacklist:
            if datetime.utcnow() < token_blacklist[token]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token已失效"
                )
            else:
                # 黑名单中的Token已过期，可以移除
                del token_blacklist[token]

        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token已过期"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的Token"
        )

def add_to_blacklist(token: str, expires_delta: timedelta):
    """
    将Token添加到黑名单
    """
    token_blacklist[token] = datetime.utcnow() + expires_delta

def validate_api_key(api_key: str) -> bool:
    """
    验证API Key的有效性
    这里可以添加更复杂的验证逻辑，比如调用ZAI API进行验证
    """
    if not api_key:
        return False
    
    # 简单验证：检查API Key是否不为空且长度合理
    if len(api_key.strip()) < 10:
        return False
    
    # 这里可以添加实际的API Key验证逻辑
    # 例如：调用ZAI API的验证接口
    return True

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """获取当前用户（FastAPI 依赖注入）"""
    try:
        payload = verify_token(credentials.credentials)
        return payload
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
