# schemas.py
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field
import time


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "glm-4v"
    temperature: float = 0.7
    stream: bool = True
    max_tokens: Optional[int] = None


class Delta(BaseModel):
    """choices[].delta 中的增量内容"""
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    delta: Optional[Delta] = None
    finish_reason: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatChunk(BaseModel):
    """
    标准 SSE 流式 DTO，一条 chunk 对应一次 data: 推送
    例：{"id":"chatcmpl-xxx","object":"chat.completion.chunk",
         "created":1699999999,"model":"glm-4v",
         "choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
