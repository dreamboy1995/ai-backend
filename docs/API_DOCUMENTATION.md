# AI Backend API 文档

## 概述

AI Backend 是一个基于 FastAPI 的后端服务，用于代理 ZAI 大模型 API，提供流式聊天功能。支持 JWT 认证，确保 API 安全访问。

## 基础信息

- **基础URL**: `http://localhost:3000`
- **认证方式**: JWT Bearer Token
- **API版本**: v1
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
AIDER_API_BASE=https://open.bigmodel.cn/api/paas/v4
ZAI_API_KEY=xxx
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

### 3. 获取访问令牌

**POST** `/auth/api-key`

使用 API Key 获取 JWT 访问令牌。

#### 请求头

```
Content-Type: application/json
```

#### 请求体

```json
{
  "api_key": "your_api_key_here",
  "model": "glm-4.5-air"
}
```

#### 字段说明

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| api_key | string | 是 | - | 用户的 ZAI API 密钥 |
| model | string | 否 | "glm-4.5-air" | 指定使用的模型 |

#### 响应

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| access_token | string | JWT 访问令牌 |
| token_type | string | 令牌类型，固定为 "bearer" |
| expires_in | number | 令牌有效期（秒），默认 24 小时 |

#### 状态码

- `200`: 令牌获取成功
- `401`: API Key 无效

---

### 4. 用户登出

**POST** `/auth/logout`

使当前 JWT 令牌失效，加入黑名单。

#### 请求头

```
Authorization: Bearer {access_token}
Content-Type: application/json
```

#### 响应

```json
{
  "message": "登出成功"
}
```

#### 状态码

- `200`: 登出成功
- `401`: 令牌无效

---

### 5. 聊天完成接口（流式）

**POST** `/v1/chat/completions`

代理 ZAI API 的聊天完成接口，支持流式响应。需要 JWT 认证。

#### 请求头

```
Authorization: Bearer {access_token}
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
  "model": "glm-4.5-air",
  "temperature": 0.7,
  "stream": true,
  "max_tokens": null
}
```

#### 字段说明

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| messages | Array[ChatMessage] | 是 | - | 消息列表 |
| model | string | 否 | "glm-4.5-air" | 模型名称 |
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
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4.5-air","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4.5-air","choices":[{"index":0,"delta":{"content":"，"},"finish_reason":null}]}
...
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4.5-air","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

#### 响应数据结构

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1699999999,
  "model": "glm-4.5-air",
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
- `401`: 未认证或令牌无效
- `400`: 请求参数错误
- `500`: 服务器内部错误

---

### 6. 测试接口

#### 6.1 测试业务异常

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

#### 6.2 测试参数校验

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

#### 6.3 测试未处理异常

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
| 401 | 未认证或令牌无效 | 缺少或无效的 JWT 令牌 |
| 422 | 请求参数校验失败 | 请求参数格式错误 |
| 500 | 服务器内部错误 | 服务器处理请求时发生错误 |

## 使用示例

### cURL 示例

```bash
# 1. 获取访问令牌
curl -X POST "http://localhost:3000/auth/api-key" \
     -H "Content-Type: application/json" \
     -d '{"api_key": "your_api_key_here"}'

# 响应示例
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 86400
}

# 2. 使用令牌调用聊天接口
curl -X POST "http://localhost:3000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
     -d '{
         "messages": [
             {
                 "role": "user",
                 "content": "你好，请介绍一下你自己"
             }
         ],
         "model": "glm-4.5-air",
         "temperature": 0.7,
         "stream": true
     }'

# 3. 用户登出
curl -X POST "http://localhost:3000/auth/logout" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Python 示例

```python
import requests
import json

# 1. 获取访问令牌
token_response = requests.post(
    "http://localhost:3000/auth/api-key",
    json={"api_key": "your_api_key_here"}
)
access_token = token_response.json()["access_token"]

# 2. 聊天完成（流式）
response = requests.post(
    "http://localhost:3000/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    },
    json={
        "messages": [
            {
                "role": "user",
                "content": "你好，请介绍一下你自己"
            }
        ],
        "model": "glm-4.5-air",
        "temperature": 0.7,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

# 3. 用户登出
requests.post(
    "http://localhost:3000/auth/logout",
    headers={"Authorization": f"Bearer {access_token}"}
)
```

### JavaScript 示例

```javascript
// 1. 获取访问令牌
const tokenResponse = await fetch('http://localhost:3000/auth/api-key', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    api_key: 'your_api_key_here'
  })
});

const { access_token } = await tokenResponse.json();

// 2. 聊天完成（流式）
const response = await fetch('http://localhost:3000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({
    messages: [
      {
        role: 'user',
        content: '你好，请介绍一下你自己'
      }
    ],
    model: 'glm-4.5-air',
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

// 3. 用户登出
await fetch('http://localhost:3000/auth/logout', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`
  }
});
```

## 注意事项

1. **JWT 认证**: 除了健康检查和文档接口外，所有接口都需要 JWT 认证
2. **令牌有效期**: JWT 令牌默认有效期为 24 小时
3. **流式响应**: 聊天完成接口默认返回流式响应，需要特殊处理 SSE 格式
4. **错误处理**: 所有接口都有完善的错误处理机制，会返回统一的错误格式
5. **环境配置**: 确保 `.env` 文件中的 `ZAI_API_KEY` 和 `JWT_SECRET` 配置正确
6. **CORS**: 当前配置允许所有来源的跨域请求，生产环境应限制具体域名
7. **性能**: 流式响应使用异步处理，适合实时对话场景

## 更新日志

### v0.1.0 (2024-01-XX)
- 初始版本
- 实现基础健康检查接口
- 实现配置检查接口
- 实现聊天完成接口（流式）
- 实现 JWT 认证机制
- 添加全局异常处理
- 支持 ZAI API 代理
