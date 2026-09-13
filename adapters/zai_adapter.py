import httpx
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
import uuid
import logging

logger = logging.getLogger(__name__)

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
        model: str = "glm-4v",
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
                response.raise_for_status()
                
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
                            
        except httpx.HTTPError as e:
            logger.error(f"ZAI API request failed: {e}")
            # 返回错误信息
            yield {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error",
                        "error": {"message": str(e)}
                    }
                ]
            }
