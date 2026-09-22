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

**POST** `/auth/token`

使用 API Key 获取 JWT 访问令牌。

#### 请求头

```
Content-Type: application/json
```

#### 请求体

```json
{
  "apiKey": "your_api_key_here",
  "model": "glm-4.5-air"
}
```

#### 字段说明

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| apiKey | string | 是 | - | 用户的 ZAI API 密钥 |
| model | string | 否 | "glm-4.5-air" | 指定使用的模型 |

#### 响应

```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expiresIn": 86400
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| accessToken | string | JWT 访问令牌 |
| expiresIn | number | 令牌有效期（秒），默认 24 小时 |

#### 状态码

- `200`: 令牌获取成功
- `401`: API Key 无效

---

### 4. 用户登出

**POST** `/auth/logout`

使当前 JWT 令牌失效，加入黑名单。

#### 请求头

```
Authorization: Bearer {accessToken}
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
Authorization: Bearer {accessToken}
Content-Type: application/json
```

#### 请求体

```json
{
  "session_id": "sess_abc123",
  "messages": [
    {
      "role": "user",
      "content": "@main.py 这个函数是干嘛的？"
    }
  ],
  "model": "glm-4.5-air",
  "temperature": 0.7,
  "stream": true,
  "max_tokens": null,
  "contexts": [
    {
      "type": "file",
      "file_path": "src/main.py",
      "content_snippet": "def hello():\n    print('hello')\n",
      "language": "python"
    },
    {
      "type": "selection",
      "file_path": "src/utils.js",
      "content_snippet": "function formatDate(d) { return d.toISOString(); }",
      "language": "javascript"
    },
    {
      "type": "implicit",
      "file_path": "src/utils.js",
      "content_snippet": "// 当前激活文件全文（截断后）",
      "language": "javascript"
    }
  ]
}
```

#### 字段说明

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| session_id | string | 否 | null | 会话 ID。携带时后端维护多轮对话历史并使用滑动窗口裁剪（保留 System + 最近 5 轮，Token 预算 8000）；不携带时为无状态模式，直接透传 messages |
| messages | Array[ChatMessage] | 是 | - | 消息列表。携带 session_id 时通常只需传入本次用户提问，后端会自动拼接历史 |
| model | string | 否 | "glm-4.5-air" | 模型名称 |
| temperature | number | 否 | 0.7 | 温度参数，控制随机性 |
| stream | boolean | 否 | true | 是否流式输出 |
| max_tokens | number | 否 | null | 最大令牌数 |
| contexts | Array[ContextItem] | 否 | null | S2 新增。上下文数组，承载 @文件 / @选中代码 / 隐式上下文。ContextBuilder 将其格式化为结构化 XML 标签插入 System Prompt，并纳入 Token 裁剪计算 |

#### ContextItem 结构（S2 第 13-14 天新增）

```json
{
  "type": "file",
  "file_path": "src/main.py",
  "content_snippet": "def hello():\n    print('hello')\n",
  "language": "python"
}
```

#### 字段说明

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| type | string | 是 | - | 上下文类型，枚举值：`file`（用户主动 @ 的整个文件）、`selection`（编辑器选中的代码片段）、`implicit`（插件自动附带的当前激活文件，不显示 @ 标签） |
| file_path | string | 是 | - | 文件路径，非空字符串。用于在 System Prompt 中标注来源 |
| content_snippet | string | 是 | - | 已截断的文件/代码片段内容。后端对单条 snippet 上限做防御性校验（50000 字符，约 12500 token），超过返回 422 拒绝 |
| language | string | 否 | null | 文件语言（如 `python` / `javascript`），用于代码块语言标签 |

#### 上下文截断规则（关键，研发必读）

⚠️ **大文件截断必须在插件端完成**，后端只做防御性校验：

- **截断策略**：只取头尾各 200 行 + 光标所在行附近 50 行，绝对禁止把 1 万行的文件全量塞进去，否则后端 Token 直接爆炸。
- **后端防御**：单条 `content_snippet` 超过 50000 字符时返回 422 错误，提示客户端先做截断。
- **Token 计数**：后端使用 `tiktoken` 的 `cl100k_base` 编码精确计数（S2 关键技术预研要求，不可用字符串长度 / 4 估算，DeepSeek/中文场景 Token 比例差异大）。
- **拼装方式**：ContextBuilder 将上下文拼装为 `<context_files><file path="..." lang="...">...</file></context_files>` 结构化 XML 标签，与基础系统提示词合并后置于消息列表首位。系统提示词 Token 纳入裁剪预算，保证总 Token 不超限。
- **丢弃优先级**：当总 Token 超限时，按文件优先级丢弃——用户主动 `@` 的（file/selection）保留，自动附带的 implicit 当前文件优先截断或丢弃。
- **系统提示词内容**：包含助手角色定义、代码块输出规范（要求使用 ```language 围栏，配合 Apply 功能）、以及上下文 XML。

#### 会话管理说明（S2 第 11-12 天）

- **多轮记忆**：携带 `session_id` 后，后端会将每条用户消息和助手回复存入会话，后续请求自动携带历史上下文。
- **滑动窗口裁剪**：为避免 Token 无限膨胀，后端始终保留 System 消息 + 最近 5 轮对话（user+assistant），超出部分从最旧消息开始丢弃。
- **Token 预算**：总预算 8000 tokens，预留 20% 余量（实际 6400 触发裁剪），使用 `tiktoken` 精确计数。
- **会话过期**：会话默认 1 小时无活动后自动清理（TTL=3600s）。
- **无状态兼容**：不携带 `session_id` 时，行为与此前完全一致，按请求中的 `messages` 直接透传。

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

### 6. 用户用量查询（S3 第 21-22 天新增）

**GET** `/v1/user/usage`

查询当前用户今日 Token 消耗量与配额信息。需要 JWT 认证。此接口不受限频限制（状态查询接口）。

#### 请求头

```
Authorization: Bearer {accessToken}
```

#### 响应

```json
{
  "user_id": "a1b2c3d4e5f6...",
  "used_tokens_today": 12500,
  "quota_limit_per_day": 100000,
  "percentage": 0.125,
  "reset_at": "2026-09-23T00:00:00Z"
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| user_id | string | 用户标识（JWT sub 字段的 SHA256 哈希，避免原始 API Key 暴露） |
| used_tokens_today | number | 今日已消耗 Token 数（每次聊天完成后累加） |
| quota_limit_per_day | number | 每日 Token 配额上限（默认 100000） |
| percentage | number | 已用比例（0.0 ~ 1.0） |
| reset_at | string | 配额重置时间（ISO 8601，UTC 午夜） |

#### 状态码

- `200`: 查询成功
- `401`: 未认证或令牌无效

#### 限频说明（S3 第 21-22 天）

后端对所有 `/v1/` 前缀的业务接口实施滑动窗口限频（`/v1/user/usage` 除外）：

| 限制维度 | 限制值 | 说明 |
|----------|--------|------|
| 每分钟 | 20 次 | 按 user_id（JWT sub 字段哈希）限频 |
| 每天 | 500 次 | 按 user_id 限频 |

超限时返回 HTTP 429，并在 Response Header 中携带：

| Header | 说明 |
|--------|------|
| X-RateLimit-Reset | 剩余重置秒数 |
| X-RateLimit-Remaining | 剩余请求数 |
| X-RateLimit-Limit | 当前窗口的请求上限 |
| Retry-After | 建议重试等待秒数（同 X-RateLimit-Reset） |

#### 限频响应

```json
{
  "code": 429,
  "message": "请求频率超限，请稍后再试",
  "path": "/v1/chat/completions"
}
```

---

### 7. 测试接口

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
curl -X POST "http://localhost:3000/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"apiKey": "your_api_key_here"}'

# 响应示例
{
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expiresIn": 86400
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
    "http://localhost:3000/auth/token",
    json={"apiKey": "your_api_key_here"}
)
access_token = token_response.json()["accessToken"]

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
const tokenResponse = await fetch('http://localhost:3000/auth/token', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    apiKey: 'your_api_key_here'
  })
});

const { accessToken } = await tokenResponse.json();

// 2. 聊天完成（流式）
const response = await fetch('http://localhost:3000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${accessToken}`
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
    'Authorization': `Bearer ${accessToken}`
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
