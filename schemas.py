# schemas.py
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
import time


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "mock-model"
    stream: bool = False


class Delta(BaseModel):
    """choices[].delta 中的增量内容"""
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChunk(BaseModel):
    """
    标准 SSE 流式 DTO，一条 chunk 对应一次 data: 推送
    例：{"id":"chatcmpl-xxx","object":"chat.completion.chunk",
         "created":1699999999,"model":"mock-model",
         "choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None