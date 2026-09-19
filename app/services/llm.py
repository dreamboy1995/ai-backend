import httpx
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
import uuid
import logging
from app.error_codes import ErrorCode, get_error_message

logger = logging.getLogger(__name__)


class ZAIAdapterError(Exception):
    """ZAI适配器基础异常类"""
    def __init__(self, code: int, message: str = None):
        self.code = code
        self.message = message or get_error_message(code)
        super().__init__(self.message)


class ZAIRateLimitError(ZAIAdapterError):
    """模型限流异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_RATE_LIMIT, message)


class ZAITokenTooLongError(ZAIAdapterError):
    """Token超长异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_TOKEN_TOO_LONG, message)


class ZAITimeoutError(ZAIAdapterError):
    """超时异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_TIMEOUT, message)


class ZAINetworkError(ZAIAdapterError):
    """网络连接异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_NETWORK_ERROR, message)


class ZAIServiceError(ZAIAdapterError):
    """模型服务异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_SERVICE_ERROR, message)

class ZAIAdapter:
    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )

    async def close(self):
        await self.client.aclose()

    async def chat_completion(
        self,
        messages: list,
        model: str = "glm-4.5-air",
        temperature: float = 0.7,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        调用ZAI API并返回流式响应
        将ZAI格式转换为OpenAI兼容格式
        """
        request_id = str(uuid.uuid4())
        
        # ZAI API请求参数
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/paas/v4/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream"}
            ) as response:
                # 检查HTTP状态码，抛出对应异常
                if response.status_code == 429:
                    raise ZAIRateLimitError("模型调用频率超限")
                elif response.status_code == 400:
                    # 读取响应体以判断具体错误原因（模型无效 / token超长 / 其他参数错误）
                    error_text = await response.aread()
                    error_msg = "请求参数错误"
                    try:
                        error_data = json.loads(error_text)
                        error_msg = error_data.get("error", {}).get("message", error_text.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        error_msg = error_text.decode(errors="replace") if isinstance(error_text, bytes) else str(error_text)

                    lower_msg = error_msg.lower()
                    if "token" in lower_msg or "length" in lower_msg or "too long" in lower_msg or "maximum" in lower_msg:
                        raise ZAITokenTooLongError(error_msg)
                    elif "model" in lower_msg and ("not" in lower_msg or "exist" in lower_msg or "invalid" in lower_msg or "unavailable" in lower_msg):
                        raise ZAIServiceError(f"模型不存在或无效: {error_msg}")
                    else:
                        raise ZAIServiceError(f"请求参数错误: {error_msg}")
                elif response.status_code >= 500:
                    raise ZAIServiceError(f"模型服务异常，状态码: {response.status_code}")
                elif response.status_code >= 400:
                    raise ZAIServiceError(f"请求失败，状态码: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            # 流结束
                            yield {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }
                                ]
                            }
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # 检查ZAI返回的错误信息
                            if "error" in data:
                                error_msg = data["error"].get("message", "未知错误")
                                error_code = data["error"].get("code", "")
                                
                                # 根据错误码判断错误类型
                                if "rate" in error_code.lower() or "limit" in error_code.lower():
                                    raise ZAIRateLimitError(error_msg)
                                elif "token" in error_code.lower() or "length" in error_code.lower():
                                    raise ZAITokenTooLongError(error_msg)
                                else:
                                    raise ZAIServiceError(error_msg)
                            
                            # 转换ZAI格式为OpenAI格式
                            openai_chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": model,
                                "choices": []
                            }
                            
                            if "choices" in data and len(data["choices"]) > 0:
                                zai_choice = data["choices"][0]
                                openai_choice = {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None
                                }
                                
                                # 处理内容
                                if "delta" in zai_choice and "content" in zai_choice["delta"]:
                                    openai_choice["delta"]["content"] = zai_choice["delta"]["content"]
                                
                                # 处理结束标志
                                if "finish_reason" in zai_choice:
                                    openai_choice["finish_reason"] = zai_choice["finish_reason"]
                                
                                openai_chunk["choices"].append(openai_choice)
                            
                            yield openai_chunk
                            
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE data: {data_str}")
                            continue
                            
        except httpx.TimeoutException as e:
            logger.error(f"ZAI API timeout: {e}")
            raise ZAITimeoutError("模型响应超时")
        except httpx.ConnectError as e:
            logger.error(f"ZAI API connection error: {e}")
            raise ZAINetworkError("网络连接失败")
        except httpx.HTTPError as e:
            logger.error(f"ZAI API request failed: {e}")
            raise ZAIServiceError(f"HTTP请求失败: {str(e)}")
        except (ZAIRateLimitError, ZAITokenTooLongError, ZAITimeoutError, ZAINetworkError, ZAIServiceError):
            # 重新抛出自定义异常
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ZAI adapter: {e}", exc_info=True)
            raise ZAIServiceError(f"未知错误: {str(e)}")
