# AI Backend API 接口文档

## 概述

AI Backend 是一个基于 FastAPI 的后端服务，用于代理 ZAI（智谱AI）的大模型 API。目前实现了流式聊天完成接口，支持 OpenAI 兼容的 SSE 格式。

## 基础信息

- **基础 URL**: `http://localhost:3000`
- **API 版本**: `/v1`
- **认证方式**: 暂未实现（后续将添加 JWT 认证）
- **响应格式**: JSON / SSE（流式）

## 环境配置

### 必需的环境变量

```bash
ZAI_API_KEY=your_zai_api_key_here
PORT=3000
JWT_SECRET=your_jwt_secret_here
```

### 配置文件示例 (.env)

```env
PORT=3000
AIDER_MODEL=zai/glm-4.5-air
ZAI_API_KEY=12139402cf79409582813f5ada668907.5nmPAhhtXT5NMiO7
JWT_SECRET=xxx
```

## 接口列表

### 1. 健康检查

**GET** `/health`

检查服务是否正常运行。

#### 响应

```json
{
  "status": "ok",
  "port": 3000
}
```

#### 状态码

- `200`: 服务正常

---

### 2. 配置检查

**GET** `/config-check`

检查环境配置是否正确。

#### 响应

```json
{
  "port": 3000,
  "zai_api_key_configured": true,
  "jwt_secret_configured": false,
  "environment": "development"
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| port | number | 服务端口 |
| zai_api_key_configured | boolean | ZAI API Key 是否已配置 |
| jwt_secret_configured | boolean | JWT Secret 是否已配置 |
| environment | string | 运行环境 |

#### 状态码

- `200`: 配置检查完成

---

### 3. 聊天完成接口（流式）

**POST** `/v1/chat/completions`

代理 ZAI API 的聊天完成接口，支持流式响应。

#### 请求头

```
Content-Type: application/json
```

#### 请求体

```json
{
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下你自己"
    }
  ],
  "model": "glm-4v",
  "temperature": 0.7,
  "stream": true,
  "max_tokens": null
}
```

#### 字段说明

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| messages | Array[ChatMessage] | 是 | - | 消息列表 |
| model | string | 否 | "glm-4v" | 模型名称 |
| temperature | number | 否 | 0.7 | 温度参数，控制随机性 |
| stream | boolean | 否 | true | 是否流式输出 |
| max_tokens | number | 否 | null | 最大令牌数 |

#### ChatMessage 结构

```json
{
  "role": "user" | "assistant" | "system",
  "content": "消息内容"
}
```

#### 响应

流式响应（SSE），格式如下：

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4v","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4v","choices":[{"index":0,"delta":{"content":"，"},"finish_reason":null}]}
...
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4v","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

#### 响应数据结构

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1699999999,
  "model": "glm-4v",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "你好"
      },
      "finish_reason": null
    }
  ],
  "usage": null
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| id | string | 请求唯一标识 |
| object | string | 对象类型，固定为 "chat.completion.chunk" |
| created | number | 创建时间戳 |
| model | string | 模型名称 |
| choices | Array[Choice] | 选择列表 |
| usage | Usage | 使用统计（流式响应中通常为 null） |

#### Choice 结构

```json
{
  "index": 0,
  "delta": {
    "content": "你好"
  },
  "finish_reason": null
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| index | number | 选择索引 |
| delta | Delta | 增量内容 |
| finish_reason | string | 结束原因 |

#### Delta 结构

```json
{
  "content": "你好",
  "role": null
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| content | string | 增量内容 |
| role | string | 角色信息 |

#### 错误响应

当发生错误时，会返回错误块：

```
data: {"id":"","object":"chat.completion.chunk","created":0,"model":"","choices":[{"index":0,"delta":{},"finish_reason":"error","error":{"message":"错误信息"}}],"usage":null}
```

#### 状态码

- `200`: 请求成功（流式响应）
- `400`: 请求参数错误
- `500`: 服务器内部错误

---

### 4. 测试接口

#### 4.1 测试业务异常

**GET** `/test-error`

用于测试业务异常处理。

#### 响应

```json
{
  "code": 400,
  "message": "这是一个测试业务异常",
  "path": "/test-error"
}
```

#### 状态码

- `400`: 业务异常

---

#### 4.2 测试参数校验

**POST** `/test-validation`

用于测试参数校验。

#### 请求体

```json
{
  "name": "张三",
  "age": 25
}
```

#### 响应

```json
{
  "received": {
    "name": "张三",
    "age": 25
  }
}
```

#### 状态码

- `200`: 请求成功

---

#### 4.3 测试未处理异常

**GET** `/test-500`

用于测试未处理异常。

#### 响应

```json
{
  "code": 500,
  "message": "服务器内部错误",
  "path": "/test-500"
}
```

#### 状态码

- `500`: 服务器内部错误

---

## 错误码说明

| 错误码 | 错误信息 | 描述 |
|--------|----------|------|
| 400 | 请求参数错误 | 请求参数不合法 |
| 422 | 请求参数校验失败 | 请求参数格式错误 |
| 500 | 服务器内部错误 | 服务器处理请求时发生错误 |

## 使用示例

### cURL 示例

```bash
# 健康检查
curl http://localhost:3000/health

# 配置检查
curl http://localhost:3000/config-check

# 聊天完成（流式）
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下你自己"
      }
    ],
    "model": "glm-4v",
    "temperature": 0.7,
    "stream": true
  }'
```

### Python 示例

```python
import requests
import json

# 健康检查
response = requests.get("http://localhost:3000/health")
print(response.json())

# 聊天完成（流式）
response = requests.post(
    "http://localhost:3000/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": "你好，请介绍一下你自己"
            }
        ],
        "model": "glm-4v",
        "temperature": 0.7,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### JavaScript 示例

```javascript
// 健康检查
fetch('http://localhost:3000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// 聊天完成（流式）
const response = await fetch('http://localhost:3000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    messages: [
      {
        role: 'user',
        content: '你好，请介绍一下你自己'
      }
    ],
    model: 'glm-4v',
    temperature: 0.7,
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  console.log(chunk);
}
```

## 注意事项

1. **流式响应**: 聊天完成接口默认返回流式响应，需要特殊处理 SSE 格式。
2. **错误处理**: 所有接口都有完善的错误处理机制，会返回统一的错误格式。
3. **环境配置**: 确保 `.env` 文件中的 `ZAI_API_KEY` 配置正确。
4. **CORS**: 当前配置允许所有来源的跨域请求，生产环境应限制具体域名。
5. **性能**: 流式响应使用异步处理，适合实时对话场景。

## 更新日志

### v0.1.0 (2024-01-XX)
- 初始版本
- 实现基础健康检查接口
- 实现配置检查接口
- 实现聊天完成接口（流式）
- 实现测试接口
- 添加全局异常处理
- 支持 ZAI API 代理
