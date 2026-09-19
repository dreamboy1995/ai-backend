"""
测试错误处理脚本
用于验证第 9-10 天的异常处理、日志与联调交付任务
"""
import asyncio
import httpx
import json
from dotenv import load_dotenv
import os

load_dotenv()

# 测试配置
BASE_URL = "http://localhost:3000"
TEST_API_KEY = os.getenv("ZAI_API_KEY", "test_key")


async def test_normal_request():
    """测试正常请求"""
    print("\n=== 测试正常请求 ===")
    async with httpx.AsyncClient() as client:
        # 先获取 token
        auth_response = await client.post(
            f"{BASE_URL}/auth/token",
            json={"apiKey": TEST_API_KEY}
        )
        print(f"Auth status: {auth_response.status_code}")
        if auth_response.status_code != 200:
            print(f"Auth failed: {auth_response.text}")
            return
        
        token = auth_response.json()["accessToken"]
        print(f"Got token: {token[:20]}...")
        
        # 发送聊天请求
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True
            },
            timeout=30.0
        ) as response:
            print(f"Chat status: {response.status_code}")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        print("\n收到 [DONE] 信号")
                        break
                    try:
                        data = json.loads(data_str)
                        if "error" in data:
                            print(f"\n收到错误块: {data}")
                        elif "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            delta = choice.get("delta")
                            if delta and isinstance(delta, dict):
                                content = delta.get("content")
                                if content:
                                    print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"\n无法解析: {data_str}")
            print()


async def test_invalid_api_key():
    """测试无效 API Key"""
    print("\n=== 测试无效 API Key ===")
    async with httpx.AsyncClient() as client:
        auth_response = await client.post(
            f"{BASE_URL}/auth/token",
            json={"apiKey": "invalid_key_12345"}
        )
        print(f"Auth status: {auth_response.status_code}")
        print(f"Response: {auth_response.text}")


async def test_no_auth():
    """测试无认证请求"""
    print("\n=== 测试无认证请求 ===")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")


async def test_timeout_simulation():
    """测试超时场景（通过发送超长内容模拟）"""
    print("\n=== 测试超长内容（模拟 Token 超长） ===")
    async with httpx.AsyncClient() as client:
        # 先获取 token
        auth_response = await client.post(
            f"{BASE_URL}/auth/token",
            json={"apiKey": TEST_API_KEY}
        )
        if auth_response.status_code != 200:
            print(f"Auth failed: {auth_response.text}")
            return
        
        token = auth_response.json()["accessToken"]
        
        # 发送超长内容
        long_content = "这是一段非常长的文本，" * 10000
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{"role": "user", "content": long_content}],
                "stream": True
            },
            timeout=30.0
        ) as response:
            print(f"Chat status: {response.status_code}")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if "error" in data:
                            print(f"收到错误块: {data}")
                            break
                        elif "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"无法解析: {data_str}")
            print()


async def main():
    """主测试函数"""
    print("=" * 60)
    print("开始测试错误处理")
    print("=" * 60)
    
    # 测试 1: 无认证请求
    await test_no_auth()
    
    # 测试 2: 无效 API Key
    await test_invalid_api_key()
    
    # 测试 3: 正常请求
    await test_normal_request()
    
    # 测试 4: 超长内容
    await test_timeout_simulation()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
