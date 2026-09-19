from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator
import time


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# S2 第 13-14 天新增：上下文条目，用于 @文件 / @选中代码 / 隐式上下文
# 关键技术点：content_snippet 应为插件端截断后的内容（头尾各 200 行 + 光标附近 50 行），
# 后端再做防御性长度校验，避免恶意/异常的大文件直接爆 Token 预算。
class ContextItem(BaseModel):
    """
    上下文条目（S2 第 13-14 天）。

    - type='file'        : 用户主动 @ 的整个文件
    - type='selection'   : 用户在编辑器中选中的代码片段
    - type='implicit'    : 插件自动附带的当前激活文件（不显示 @ 标签）
    """
    type: Literal["file", "selection", "implicit"]
    file_path: str = Field(..., min_length=1, description="文件路径，必填且非空")
    content_snippet: str = Field(
        ...,
        description="插件端截断后的文件/代码片段内容。ContextBuilder 会将其格式化为 XML 标签插入 System Prompt"
    )
    language: Optional[str] = Field(
        default=None,
        description="文件语言（如 python/javascript），用于代码块语言标签"
    )

    @field_validator("content_snippet")
    @classmethod
    def _validate_snippet_length(cls, v: str) -> str:
        # 防御性截断：单条上下文 snippet 上限 50000 字符（约 12500 token）。
        # 正常插件端截断后远小于此值；超过则说明客户端未做截断，直接拒绝以保护 Token 预算。
        max_chars = 50_000
        if len(v) > max_chars:
            raise ValueError(
                f"content_snippet 过长（{len(v)} 字符 > {max_chars}），"
                f"请在插件端先做截断（头尾各 200 行 + 光标附近 50 行）后再发送"
            )
        return v


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "glm-4.5-air"
    temperature: float = 0.7
    stream: bool = True
    max_tokens: Optional[int] = None
    # S2 新增：会话 ID，用于多轮对话历史记忆（第 11-12 天）
    session_id: Optional[str] = None
    # S2 第 13-14 天新增：上下文数组，承载 @文件 / @选中代码 / 隐式上下文。
    # ContextBuilder 会消费此字段拼装 System Prompt（第 15-16 天已实现）。
    contexts: Optional[List[ContextItem]] = None


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
         "created":1699999999,"model":"glm-4.5-air",
         "choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
