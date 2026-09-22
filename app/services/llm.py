import httpx
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List
import uuid
import logging
from app.error_codes import ErrorCode, get_error_message
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义（S3 第 23-24 天：统一多厂商异常体系）
# ============================================================

class AdapterError(Exception):
    """适配器基础异常"""
    def __init__(self, code: int, message: str = None):
        self.code = code
        self.message = message or get_error_message(code)
        super().__init__(self.message)


class AdapterRateLimitError(AdapterError):
    """模型限流异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_RATE_LIMIT, message)


class AdapterTokenTooLongError(AdapterError):
    """Token 超长异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_TOKEN_TOO_LONG, message)


class AdapterTimeoutError(AdapterError):
    """超时异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_TIMEOUT, message)


class AdapterNetworkError(AdapterError):
    """网络连接异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_NETWORK_ERROR, message)


class AdapterServiceError(AdapterError):
    """模型服务异常"""
    def __init__(self, message: str = None):
        super().__init__(ErrorCode.MODEL_SERVICE_ERROR, message)


# 向后兼容：S1-S2 代码中使用 ZAI* 前缀的异常名
ZAIAdapterError = AdapterError
ZAIRateLimitError = AdapterRateLimitError
ZAITokenTooLongError = AdapterTokenTooLongError
ZAITimeoutError = AdapterTimeoutError
ZAINetworkError = AdapterNetworkError
ZAIServiceError = AdapterServiceError


# ============================================================
# 模型注册表（S3 第 23-24 天）
# ============================================================

class ModelInfo:
    """模型元信息"""
    def __init__(self, id: str, label: str, context_window: int,
                 vendor: str, default_max_tokens: int = 4096):
        self.id = id
        self.label = label
        self.context_window = context_window
        self.vendor = vendor
        self.default_max_tokens = default_max_tokens

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "context_window": self.context_window,
        }


# 模型注册表：model_id -> ModelInfo
# S3 契约：/v1/models 返回 [{id, label, context_window}]
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "glm-4.5-air": ModelInfo(
        id="glm-4.5-air", label="GLM-4.5 Air",
        context_window=128000, vendor="zai",
    ),
    "deepseek-v3": ModelInfo(
        id="deepseek-v3", label="DeepSeek V3",
        context_window=64000, vendor="deepseek",
    ),
    "gpt-4o": ModelInfo(
        id="gpt-4o", label="GPT-4o",
        context_window=128000, vendor="openai",
    ),
    "claude-3.5-sonnet": ModelInfo(
        id="claude-3.5-sonnet", label="Claude 3.5 Sonnet",
        context_window=200000, vendor="anthropic",
    ),
}

DEFAULT_MODEL = "glm-4.5-air"

# 厂商 -> 适配器类 + 对应的 settings 字段名
_VENDOR_CONFIG: Dict[str, dict] = {}  # 在适配器类定义后填充


# ============================================================
# 适配器基类
# ============================================================

class BaseAdapter:
    """
    适配器基类，定义统一接口。
    所有厂商适配器继承此类，实现 chat_completion 流式生成方法。
    输出统一为 OpenAI 兼容的 chunk 格式。
    """

    def __init__(self, api_key: str, base_url: str, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"Content-Type": "application/json"}
        )

    async def close(self):
        await self.client.aclose()

    async def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        调用厂商 API 并返回流式响应（OpenAI 兼容格式）。
        子类必须实现此方法。
        """
        raise NotImplementedError


# ============================================================
# OpenAI 兼容适配器基类（ZAI / DeepSeek / OpenAI 共用）
# ============================================================

class OpenAICompatibleAdapter(BaseAdapter):
    """
    OpenAI 兼容适配器基类。
    适用于 API 格式与 OpenAI /v1/chat/completions 一致的厂商（ZAI、DeepSeek、OpenAI）。
    子类只需覆盖 _get_endpoint() 和 __init__ 中的 base_url。
    """

    def __init__(self, api_key: str, base_url: str, timeout: float = 60.0):
        super().__init__(api_key, base_url, timeout)
        # OpenAI 兼容厂商统一使用 Bearer Token 认证
        self.client.headers["Authorization"] = f"Bearer {api_key}"

    def _get_endpoint(self) -> str:
        """返回 chat completions 端点 URL，子类可覆盖"""
        return f"{self.base_url}/v1/chat/completions"

    async def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        调用 OpenAI 兼容 API 并返回流式响应。
        将厂商返回格式转换为 OpenAI 兼容 chunk 格式。
        """
        request_id = str(uuid.uuid4())

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        try:
            async with self.client.stream(
                "POST",
                self._get_endpoint(),
                json=payload,
                headers={"Accept": "text/event-stream"}
            ) as response:
                # 检查 HTTP 状态码，抛出对应异常
                if response.status_code == 429:
                    raise AdapterRateLimitError("模型调用频率超限")
                elif response.status_code == 400:
                    error_text = await response.aread()
                    error_msg = "请求参数错误"
                    try:
                        error_data = json.loads(error_text)
                        error_msg = error_data.get("error", {}).get("message", error_text.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        error_msg = error_text.decode(errors="replace") if isinstance(error_text, bytes) else str(error_text)

                    lower_msg = error_msg.lower()
                    if "token" in lower_msg or "length" in lower_msg or "too long" in lower_msg or "maximum" in lower_msg:
                        raise AdapterTokenTooLongError(error_msg)
                    elif "model" in lower_msg and ("not" in lower_msg or "exist" in lower_msg or "invalid" in lower_msg or "unavailable" in lower_msg):
                        raise AdapterServiceError(f"模型不存在或无效: {error_msg}")
                    else:
                        raise AdapterServiceError(f"请求参数错误: {error_msg}")
                elif response.status_code == 401:
                    raise AdapterServiceError("API Key 无效或未授权")
                elif response.status_code >= 500:
                    raise AdapterServiceError(f"模型服务异常，状态码: {response.status_code}")
                elif response.status_code >= 400:
                    raise AdapterServiceError(f"请求失败，状态码: {response.status_code}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
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

                            # 检查返回的错误信息
                            if "error" in data:
                                error_msg = data["error"].get("message", "未知错误")
                                error_code = data["error"].get("code", "")

                                if "rate" in str(error_code).lower() or "limit" in str(error_code).lower():
                                    raise AdapterRateLimitError(error_msg)
                                elif "token" in str(error_code).lower() or "length" in str(error_code).lower():
                                    raise AdapterTokenTooLongError(error_msg)
                                else:
                                    raise AdapterServiceError(error_msg)

                            # 转换为 OpenAI 兼容格式
                            openai_chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": model,
                                "choices": []
                            }

                            if "choices" in data and len(data["choices"]) > 0:
                                vendor_choice = data["choices"][0]
                                openai_choice = {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None
                                }

                                if "delta" in vendor_choice and "content" in vendor_choice["delta"]:
                                    openai_choice["delta"]["content"] = vendor_choice["delta"]["content"]

                                if "finish_reason" in vendor_choice:
                                    openai_choice["finish_reason"] = vendor_choice["finish_reason"]

                                openai_chunk["choices"].append(openai_choice)

                            yield openai_chunk

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE data: {data_str}")
                            continue

        except httpx.TimeoutException as e:
            logger.error(f"API timeout: {e}")
            raise AdapterTimeoutError("模型响应超时")
        except httpx.ConnectError as e:
            logger.error(f"API connection error: {e}")
            raise AdapterNetworkError("网络连接失败")
        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            raise AdapterServiceError(f"HTTP请求失败: {str(e)}")
        except (AdapterRateLimitError, AdapterTokenTooLongError, AdapterTimeoutError, AdapterNetworkError, AdapterServiceError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in adapter: {e}", exc_info=True)
            raise AdapterServiceError(f"未知错误: {str(e)}")


# ============================================================
# 具体厂商适配器
# ============================================================

class ZAIAdapter(OpenAICompatibleAdapter):
    """ZAI（智谱 GLM）适配器"""
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://open.bigmodel.cn")

    def _get_endpoint(self) -> str:
        return f"{self.base_url}/api/paas/v4/chat/completions"


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """DeepSeek 适配器（OpenAI 兼容）"""
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.deepseek.com")


class OpenAIAdapter(OpenAICompatibleAdapter):
    """OpenAI 适配器"""
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.openai.com")


class AnthropicAdapter(BaseAdapter):
    """
    Anthropic（Claude）适配器。
    Anthropic API 格式与 OpenAI 不兼容：
    - 端点为 /v1/messages（非 /v1/chat/completions）
    - 认证使用 x-api-key 头（非 Bearer Token）
    - system 为顶层参数（不在 messages 数组中）
    - max_tokens 为必填参数
    - 流式事件格式不同（content_block_delta / message_delta / message_stop）
    """

    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.anthropic.com")
        self.client.headers["x-api-key"] = api_key
        self.client.headers["anthropic-version"] = "2023-06-01"

    async def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        调用 Anthropic API 并返回 OpenAI 兼容的流式 chunk。
        """
        request_id = str(uuid.uuid4())

        # Anthropic 要求 system 消息从 messages 中提取为顶层参数
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        non_system_messages = [m for m in messages if m.get("role") != "system"]
        system_content = "\n\n".join(system_parts) if system_parts else None

        # Anthropic 要求 max_tokens 必填
        max_tokens = kwargs.pop("max_tokens", None) or 4096

        payload: Dict[str, Any] = {
            "model": model,
            "messages": non_system_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system_content:
            payload["system"] = system_content

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=payload,
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status_code == 429:
                    raise AdapterRateLimitError("模型调用频率超限")
                elif response.status_code == 400:
                    error_text = await response.aread()
                    error_msg = "请求参数错误"
                    try:
                        error_data = json.loads(error_text)
                        error_msg = error_data.get("error", {}).get("message", error_text.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        error_msg = error_text.decode(errors="replace") if isinstance(error_text, bytes) else str(error_text)

                    lower_msg = error_msg.lower()
                    if "token" in lower_msg or "length" in lower_msg or "too long" in lower_msg or "maximum" in lower_msg:
                        raise AdapterTokenTooLongError(error_msg)
                    else:
                        raise AdapterServiceError(f"请求参数错误: {error_msg}")
                elif response.status_code == 401:
                    raise AdapterServiceError("API Key 无效或未授权")
                elif response.status_code >= 500:
                    raise AdapterServiceError(f"模型服务异常，状态码: {response.status_code}")
                elif response.status_code >= 400:
                    raise AdapterServiceError(f"请求失败，状态码: {response.status_code}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse Anthropic SSE data: {data_str}")
                            continue

                        event_type = data.get("type", "")

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None
                                    }]
                                }

                        elif event_type == "message_delta":
                            delta = data.get("delta", {})
                            stop_reason = delta.get("stop_reason")
                            if stop_reason:
                                # 映射 Anthropic stop_reason -> OpenAI finish_reason
                                finish_reason = "stop" if stop_reason == "end_turn" else stop_reason
                                yield {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason
                                    }]
                                }

                        elif event_type == "message_stop":
                            # 流结束，发送 OpenAI 标准的 stop chunk
                            yield {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            break

                        elif event_type == "error":
                            error_info = data.get("error", {})
                            error_msg = error_info.get("message", "未知错误")
                            error_type = error_info.get("type", "")
                            if "rate" in str(error_type).lower() or "overloaded" in str(error_type).lower():
                                raise AdapterRateLimitError(error_msg)
                            else:
                                raise AdapterServiceError(error_msg)

        except httpx.TimeoutException as e:
            logger.error(f"Anthropic API timeout: {e}")
            raise AdapterTimeoutError("模型响应超时")
        except httpx.ConnectError as e:
            logger.error(f"Anthropic API connection error: {e}")
            raise AdapterNetworkError("网络连接失败")
        except httpx.HTTPError as e:
            logger.error(f"Anthropic API request failed: {e}")
            raise AdapterServiceError(f"HTTP请求失败: {str(e)}")
        except (AdapterRateLimitError, AdapterTokenTooLongError, AdapterTimeoutError, AdapterNetworkError, AdapterServiceError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Anthropic adapter: {e}", exc_info=True)
            raise AdapterServiceError(f"未知错误: {str(e)}")


# ============================================================
# 厂商配置（适配器类定义后填充）
# ============================================================

_VENDOR_CONFIG = {
    "zai": {"adapter_class": ZAIAdapter, "key_setting": "ZAI_API_KEY"},
    "deepseek": {"adapter_class": DeepSeekAdapter, "key_setting": "DEEPSEEK_API_KEY"},
    "openai": {"adapter_class": OpenAIAdapter, "key_setting": "OPENAI_API_KEY"},
    "anthropic": {"adapter_class": AnthropicAdapter, "key_setting": "ANTHROPIC_API_KEY"},
}


# ============================================================
# 适配器工厂
# ============================================================

class AdapterFactory:
    """
    适配器工厂（S3 第 23-24 天）。
    根据模型 ID 从 MODEL_REGISTRY 查找厂商，创建/复用对应适配器实例。
    适配器按厂商缓存（同厂商共享连接池），使用服务端配置的 API Key。
    """

    _instances: Dict[str, BaseAdapter] = {}  # key: vendor

    @classmethod
    def get_adapter(cls, model: str) -> BaseAdapter:
        """根据模型 ID 获取适配器实例"""
        if model not in MODEL_REGISTRY:
            raise AdapterServiceError(f"不支持的模型: {model}")

        info = MODEL_REGISTRY[model]
        vendor = info.vendor

        if vendor not in cls._instances:
            api_key = cls._get_vendor_api_key(vendor)
            if not api_key:
                raise AdapterServiceError(
                    f"厂商 '{vendor}' 的 API Key 未配置，无法使用模型 '{model}'"
                )
            adapter_class = _VENDOR_CONFIG[vendor]["adapter_class"]
            cls._instances[vendor] = adapter_class(api_key=api_key)
            logger.info(f"[AdapterFactory] 创建 {vendor} 适配器实例")

        return cls._instances[vendor]

    @classmethod
    def _get_vendor_api_key(cls, vendor: str) -> str:
        """从 settings 获取指定厂商的 API Key"""
        key_setting = _VENDOR_CONFIG[vendor]["key_setting"]
        return getattr(settings, key_setting, "")

    @classmethod
    def get_model_info(cls, model: str) -> Optional[ModelInfo]:
        """获取模型元信息"""
        return MODEL_REGISTRY.get(model)

    @classmethod
    def list_models(cls) -> List[ModelInfo]:
        """返回所有已注册模型列表"""
        return list(MODEL_REGISTRY.values())

    @classmethod
    async def close_all(cls):
        """关闭所有适配器连接（应用关闭时调用）"""
        for adapter in cls._instances.values():
            await adapter.close()
        cls._instances.clear()
        logger.info("[AdapterFactory] 所有适配器连接已关闭")
