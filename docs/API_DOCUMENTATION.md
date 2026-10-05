# AI Backend API 文档

## 概述

AI Backend 是一个基于 FastAPI 的 AI 编程助手后端服务，代理多厂商大模型 API（ZAI / DeepSeek / OpenAI / Anthropic），提供流式聊天、代码索引、符号搜索、Agent 任务规划与执行、沙箱安全等完整能力。支持 JWT 认证、滑动窗口限频、每日 Token 配额、熔断与自修复。

## 基础信息

- **基础URL**: `http://localhost:3000`
- **认证方式**: JWT Bearer Token（全局中间件校验，除健康检查和文档外均需认证）
- **API版本**: v1
- **响应格式**: JSON / SSE（流式）/ WebSocket
- **字段命名**: 后端 Pydantic model 默认使用 Python `snake_case` 风格，所有请求体和响应体的 JSON 字段名均为 `snake_case`（如 `session_id`、`workspace_root`、`total_symbols`）。
- **Auth 接口例外**: `/auth/token` 接口特殊配置了 `alias_generator=to_camel + response_model_by_alias=True`，请求字段和响应字段使用 `camelCase`（`apiKey` / `accessToken` / `expiresIn`）。

> ⚠️ **命名约定说明**：本文档所有 JSON 示例中的字段名均与后端实际行为保持一致。绝大多数接口使用 `snake_case`，仅 `/auth/token` 接口使用 `camelCase`（后端 auth.py 单独配置了 alias generator）。此前版本曾混淆，已根据后端 OpenAPI schema + Pydantic 源码逐接口修正。前端插件代码已按此约定与后端对齐。

## 环境配置

### 必需的环境变量

```bash
ZAI_API_KEY=your_zai_api_key_here    # ZAI 厂商 API Key
JWT_SECRET=your_jwt_secret_here      # JWT 签名密钥
PORT=3000
```

### 可选的多厂商 API Key

```bash
DEEPSEEK_API_KEY=xxx     # DeepSeek 厂商 Key
OPENAI_API_KEY=xxx       # OpenAI 厂商 Key
ANTHROPIC_API_KEY=xxx    # Anthropic 厂商 Key
```

### 配置文件示例 (.env)

```env
PORT=3000
HOST=127.0.0.1
ZAI_API_KEY=xxx
JWT_SECRET=xxx
DEEPSEEK_API_KEY=xxx
ENVIRONMENT=development
SANDBOX_MODE=auto       # docker / host / auto
REDIS_ENABLED=false     # 默认关闭，设为 true 启用 Redis
```

---

## 接口列表

### 1. 健康检查

**GET** `/health`

检查服务是否正常运行。无需认证。

#### 响应

```json
{
  "status": "ok",
  "port": 3000
}
```

---

### 2. 配置检查（仅非生产环境）

**GET** `/config-check`

检查环境配置是否正确。仅在 `ENVIRONMENT != production` 时注册。

#### 响应

```json
{
  "port": 3000,
  "zai_api_key_configured": true,
  "jwt_secret_configured": false,
  "environment": "development"
}
```

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

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| apiKey | string | 是 | - | 用户的 API 密钥（sub 字段，用于限频、配额） |
| model | string | 否 | "glm-4.5-air" | 指定默认模型，写入 JWT payload |

#### 响应

```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expiresIn": 86400
}
```

| 字段 | 类型 | 描述 |
|------|------|------|
| accessToken | string | JWT 访问令牌 |
| expiresIn | number | 令牌有效期（秒），默认 24 小时 |

#### 状态码

- `200`: 令牌获取成功
- `401`: API Key 无效

---

### 4. 验证令牌

**GET** `/auth/validate`

验证当前 Token 是否有效（不强制认证，直接从 Authorization Header 解析）。

#### 请求头

```
Authorization: Bearer {accessToken}
```

#### 响应（有效）

```json
{
  "valid": true,
  "payload": {
    "sub": "a1b2c3d4...",
    "model": "glm-4.5-air",
    "iat": 1699999999,
    "exp": 1700086399
  }
}
```

#### 响应（无效）

```json
{
  "valid": false,
  "detail": "无效的认证凭据"
}
```

---

### 5. 用户登出

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

---

### 6. 聊天完成接口（流式）

**POST** `/v1/chat/completions`

代理多厂商 LLM API 的聊天完成接口，支持流式响应、会话管理、上下文注入、自动检索、Inline Chat、多文件 JSON Mode 与 Diff。

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
  "mode": "chat",
  "contexts": [
    {
      "type": "file",
      "file_path": "src/main.py",
      "content_snippet": "def hello():\n    print('hello')\n",
      "language": "python"
    }
  ],
  "retrieval_config": {
    "auto_context": true,
    "top_k": 5,
    "include_references": true
  },
  "inline_selection": null,
  "response_format": null
}
```

#### 字段说明

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| messages | Array[ChatMessage] | 是 | - | 消息列表 |
| model | string | 否 | "glm-4.5-air" | 模型名称 |
| temperature | number | 否 | 0.7 | 温度参数，控制随机性 |
| stream | boolean | 否 | true | 是否流式输出（非流式暂不支持） |
| max_tokens | number | 否 | null | 最大令牌数，不传则根据 mode 自动设置 |
| session_id | string | 否 | null | 会话 ID。携带时后端维护多轮对话历史并滑动窗口裁剪；不携带时为无状态模式 |
| mode | string | 否 | "chat" | 请求模式：`chat` 普通对话(60s) / `new` 单文件生成(120s) / `inline` Inline Chat(120s) / `builder` 预留 |
| contexts | Array[ContextItem] | 否 | null | 上下文数组（@文件 / @选中代码 / 隐式上下文） |
| retrieval_config | RetrievalConfig | 否 | null | 自动检索配置，开启后后端用混合检索器注入相关代码片段 |
| inline_selection | InlineSelection | 否 | null | Inline Chat 选中代码范围（mode='inline' 时使用） |
| response_format | ResponseFormat | 否 | null | 强制结构化输出，目前仅 `{type: "json_object"}` 用于多文件修改场景 |

#### ChatMessage 结构

```json
{
  "role": "user",
  "content": "你好"
}
```

- `role`: `"system"` / `"user"` / `"assistant"`
- `content`: 消息内容

#### ContextItem 结构

```json
{
  "type": "file",
  "file_path": "src/main.py",
  "content_snippet": "...",
  "language": "python"
}
```

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| type | string | 是 | `file` 用户主动 @ 的文件 / `selection` 选中的代码片段 / `implicit` 插件附带的当前激活文件 |
| file_path | string | 是 | 文件相对路径，非空字符串 |
| content_snippet | string | 是 | 已截断的文件/代码片段内容，单条上限 50000 字符 |
| language | string | 否 | 文件语言（如 python / javascript） |

#### RetrievalConfig 结构

```json
{
  "auto_context": true,
  "top_k": 5,
  "include_references": true
}
```

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| auto_context | boolean | true | 是否自动从索引中检索相关代码注入 Prompt |
| top_k | number | 5 | 注入 Prompt 的片段数（1~20） |
| include_references | boolean | true | SSE 流中是否返回 references 元数据块 |

#### InlineSelection 结构（mode='inline'）

```json
{
  "file_path": "src/main.py",
  "selected_text": "def hello(): pass",
  "start_line": 10,
  "end_line": 12
}
```

#### ResponseFormat 结构

```json
{
  "type": "json_object"
}
```

#### 上下文截断规则

⚠️ **大文件截断必须在插件端完成**，后端只做防御性校验：

- **截断策略**：只取头尾各 200 行 + 光标所在行附近 50 行
- **后端防御**：单条 `content_snippet` 超过 50000 字符时返回 422
- **Token 计数**：使用 `tiktoken` 的 `cl100k_base` 编码精确计数
- **丢弃优先级**：用户主动 @ 的（file/selection）保留，implicit 自动附带的优先截断
- **JSON Mode 风险**：`response_format=json_object` 时独立超时 30 秒，超过返回友好提示

#### 会话管理

- **滑动窗口裁剪**：保留 System 消息 + 最近 5 轮对话
- **Token 预算**：总预算 8000 tokens，20% 余量触发裁剪
- **会话过期**：默认 1 小时无活动后自动清理
- **无状态兼容**：不携带 `session_id` 时按请求 messages 直接透传

#### 流式响应（SSE）

SSE 流可能推送四种数据块，按顺序：

**1. type:meta（可选，第一个 chunk 之前）**

```
data: {"type":"meta","references":[
  {"file":"src/main.py","lines":"12-45","score":0.92,"symbol":"foo"}
]}
```

**2. 标准 chat.completion.chunk**

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4.5-air","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699999999,"model":"glm-4.5-air","choices":[{"index":0,"delta":{"content":"世界"},"finish_reason":"stop"}]}
```

**3. type:diff（可选，多文件 JSON Mode 时 [DONE] 之前）**

```
data: {"type":"diff","files":[
  {"path":"src/main.py","old_content":"def hello(): pass","new_content":"def hello(): print('hi')","diff":"@@ ..."}
]}
```

超过 3 个文件时会拆分为多个 type:diff 块。

**4. [DONE]**

```
data: [DONE]
```

#### 错误块

模型调用失败时返回：

```
data: {"error":{"code":10201,"msg":"服务器内部错误"}}
```

#### 状态码

- `200`: 请求成功（流式响应）
- `400`: 请求参数错误
- `401`: 未认证或令牌无效
- `422`: 参数校验失败（如 content_snippet 过长）
- `429`: 限频超限
- `500`: 服务器内部错误
- `503`: 适配器不可用

---

### 7. 用户用量查询

**GET** `/v1/user/usage`

查询当前用户今日 Token 消耗量与配额信息。此接口不受限频限制。

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
  "reset_at": "2026-10-06T00:00:00Z"
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| user_id | string | 用户标识（JWT sub 字段的哈希值） |
| used_tokens_today | number | 今日已消耗 Token 数 |
| quota_limit_per_day | number | 每日 Token 配额上限（默认 100000） |
| percentage | number | 已用比例（0.0 ~ 1.0） |
| reset_at | string | 配额重置时间（ISO 8601，UTC 午夜） |

---

### 8. 模型列表

**GET** `/v1/models`

获取可用模型列表，供设置面板下拉框使用。

#### 请求头

```
Authorization: Bearer {accessToken}
```

#### 响应

```json
[
  {"id": "glm-4.5-air", "label": "ZAI GLM-4.5 Air", "context_window": 128000},
  {"id": "deepseek-v3", "label": "DeepSeek V3", "context_window": 64000},
  {"id": "gpt-4o", "label": "GPT-4o", "context_window": 128000}
]
```

---

### 9. 代码索引 - 触发全量索引

**POST** `/v1/index/start`

触发全量索引，后台线程异步执行，立即返回 job_id。

#### 请求头

```
Authorization: Bearer {accessToken}
Content-Type: application/json
```

#### 请求体

```json
{
  "workspace_root": "/path/to/workspace",
  "force_rebuild": false,
  "priority_files": ["src/main.py"]
}
```

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| workspace_root | string | 是 | - | 工作区根路径 |
| force_rebuild | boolean | 否 | false | 是否强制重建索引 |
| priority_files | Array[string] | 否 | null | 优先索引的文件列表（即用即索引） |

#### 响应

```json
{
  "job_id": "idx_abc123",
  "total_files": 1523
}
```

---

### 10. 代码索引 - 查询进度

**GET** `/v1/index/status`

查询当前索引进度、状态、符号总数等。

#### 响应

```json
{
  "status": "indexing",
  "total": 1523,
  "processed": 876,
  "percentage": 0.575,
  "total_symbols": 3420,
  "message": null
}
```

#### status 枚举值

- `"idle"`: 空闲，尚未开始
- `"indexing"`: 正在索引
- `"done"`: 完成
- `"error"`: 出错

---

### 11. 代码索引 - 增量更新

**POST** `/v1/index/update`

由插件在文件保存/删除/重命名时调用，仅重新处理该文件。

#### 请求体

```json
{
  "file_path": "src/main.py",
  "action": "modified"
}
```

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| file_path | string | 是 | 变更的文件路径（相对路径） |
| action | string | 是 | `modified` / `deleted` / `renamed` |

#### 响应

```json
{
  "success": true,
  "message": "更新成功",
  "symbols_count": 24
}
```

---

### 12. 代码语义搜索

**GET** `/v1/search?q=<query>&top_k=<n>`

通过 LanceDB 向量检索返回语义最相关的代码切片。

#### 请求参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | string | - | 搜索关键词（必填） |
| top_k | number | 10 | 返回数量（1~100） |

#### 响应

```json
{
  "query": "排序函数",
  "top_k": 10,
  "total": 5,
  "results": [
    {
      "id": "chunk_001",
      "file_path": "src/utils.py",
      "symbol_name": "sort_data",
      "chunk_type": "function",
      "content": "def sort_data(arr): return sorted(arr)",
      "start_line": 10,
      "end_line": 12,
      "distance": 0.34,
      "score": 0.82
    }
  ]
}
```

> 若代码库尚未索引，返回 `409`: "代码库尚未建立向量索引，请先调用 POST /v1/index/start"

---

### 13. 依赖关系图 - 关联文件

**GET** `/v1/graph/related?file=<path>&depth=<n>`

BFS 遍历查询与指定文件强关联的上下游文件。

#### 请求参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| file | string | - | 查询起点文件（相对路径，必填） |
| depth | number | 2 | BFS 遍历深度（1~5） |

#### 响应

```json
{
  "file_path": "src/main.py",
  "depth": 2,
  "upstream": [
    {"file_path": "src/utils.py", "direction": "upstream", "depth": 1, "edge": {"source": "src/main.py", "target": "src/utils.py", "edge_type": "import", "line": 3}}
  ],
  "downstream": [],
  "total": 1
}
```

---

### 14. 依赖关系图 - 文件 import 列表

**GET** `/v1/graph/imports?file=<path>`

返回指定文件直接 import 的所有本地模块名。

#### 响应

```json
{
  "file_path": "src/main.py",
  "imports": [
    {"file_path": "src/utils.py", "module_name": "utils", "line": 3, "resolved": true},
    {"file_path": null, "module_name": "os", "line": 2, "resolved": false}
  ]
}
```

- `resolved=true`: 已解析为本地文件
- `resolved=false`: 未解析为本地文件（标准库/三方包）

---

### 15. 依赖关系图 - 统计信息

**GET** `/v1/graph/stats`

返回依赖图统计信息（节点数、边数等，调试用）。

---

### 16. 符号实时补全

**GET** `/v1/symbols/search?q=<prefix>&limit=<n>`

用户输入 `#` 后触发的符号实时补全（300ms 内响应）。

#### 请求参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | string | "" | 符号名前缀，为空返回空列表 |
| limit | number | 10 | 候选数量（1~50） |

#### 响应

```json
{
  "query": "Data",
  "limit": 10,
  "total": 3,
  "symbols": [
    {"name": "DataProcessor", "type": "class", "file_path": "src/processor.py", "line": 10},
    {"name": "DatabaseConnector", "type": "class", "file_path": "src/db.py", "line": 20}
  ]
}
```

---

### 17. 符号定义定位

**GET** `/v1/symbols/definition?name=<symbol>`

精确定位符号定义的文件路径与行号范围。

#### 响应

```json
{
  "name": "DataProcessor",
  "type": "class",
  "file_path": "src/processor.py",
  "start_line": 10,
  "end_line": 45
}
```

---

### 18. 符号反向依赖查询

**GET** `/v1/symbols/callers?name=<symbol>&limit=<n>`

查询"谁调用了这个符号"，利用 Call Graph 反向遍历。

#### 响应

```json
{
  "symbol_name": "save",
  "total": 2,
  "callers": [
    {"file_path": "src/main.py", "caller_symbol": "process_data", "line": 15, "raw": "db.save()"},
    {"file_path": "src/utils.py", "caller_symbol": "", "line": 88, "raw": "repo.save(obj)"}
  ]
}
```

---

### 19. Cue 编辑位置预测

**POST** `/v1/cue/suggest`

插件将最近的编辑上下文发给后端，返回"可能受影响的文件列表"，用于编辑器灰色箭头提示。

#### 请求体

```json
{
  "file_path": "src/UserService.js",
  "modified_line": 22,
  "action": "rename",
  "symbol_name": "getUser"
}
```

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| file_path | string | 是 | 用户当前编辑的文件路径 |
| modified_line | number | 是 | 被改动的行号（1-based） |
| action | string | 是 | `rename` / `delete` / `add_method` |
| symbol_name | string | 是 | 被改名的符号名 |

#### 三种 action 对应规则

- **rename**: 依赖跟随 — 返回所有调用该符号的跨文件位置
- **delete**: 类似 rename，但调用方需替换或移除
- **add_method**: 相似结构补全 — 返回其他文件中同名方法

#### 响应

```json
{
  "action": "rename",
  "symbol_name": "getUser",
  "total": 1,
  "suggestions": [
    {"file_path": "src/AdminService.js", "line": 18, "reason": "此函数调用了被改名的符号"}
  ]
}
```

> 风险预警：启发式规则必然有误报。后端限制返回上限（默认 20 条），并排除当前编辑文件和编辑行。

---

### 20. Agent - 创建任务计划

**POST** `/v1/agent/plan`

创建 Agent 任务计划，由大模型将自然语言需求拆解为 DAG 任务图。

#### 请求体

```json
{
  "goal": "给项目添加用户登录功能",
  "workspace_root": "/path/to/workspace",
  "model": "glm-4.5-air",
  "plan": null
}
```

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| goal | string | 是 | 自然语言需求描述 |
| workspace_root | string | 是 | 工作区根路径 |
| model | string | 否 | 使用的模型 |
| plan | Array[Step] | 否 | 预拆解的步骤列表（不传则由 Planner 自动生成） |

#### 响应

```json
{
  "session_id": "agent_sess_001",
  "plan_preview": [
    {"id": "step_1", "description": "分析现有代码结构"},
    {"id": "step_2", "description": "创建 User 模型"}
  ]
}
```

---

### 21. Agent - 查询状态

**GET** `/v1/agent/status/{session_id}`

查询 Agent 会话完整状态，供 Builder 面板轮询。

#### 路径参数

- `session_id`: Agent 会话 ID

#### 响应（部分字段）

```json
{
  "session_id": "agent_sess_001",
  "user_goal": "给项目添加用户登录功能",
  "workspace_root": "/path/to/workspace",
  "current_step_index": 2,
  "progress": 0.4,
  "progress_percent": 40,
  "is_executing": true,
  "is_paused": false,
  "pending_question": null,
  "pending_confirmation_id": null,
  "pending_confirmation_tool": null,
  "end_reason": null,
  "total_retries_used": 0,
  "test_results": null
}
```

#### end_reason 枚举（终态）

- `"completed"`: 正常完成
- `"max_iter"`: 超过最大迭代次数
- `"timeout"`: 总超时
- `"error"`: 执行错误
- `"fused"`: 熔断触发

---

### 22. Agent - 启动执行

**POST** `/v1/agent/start/{session_id}`

启动 ReAct 执行循环（后台 asyncio.Task，立即返回）。

#### 响应

```json
{
  "success": true,
  "message": "Agent 执行已启动"
}
```

---

### 23. Agent - 暂停执行

**POST** `/v1/agent/pause/{session_id}`

暂停 ReAct 循环。

---

### 24. Agent - 恢复执行

**POST** `/v1/agent/resume/{session_id}`

恢复 ReAct 循环。

---

### 25. Agent - 人工介入回复

**POST** `/v1/agent/ask/respond`

Agent 返回 ask_user 工具暂停后，用户提交回答。

#### 请求体

```json
{
  "session_id": "agent_sess_001",
  "answer": "好的，使用 JWT 认证"
}
```

---

### 26. Agent - 工具执行确认

**POST** `/v1/agent/confirm`

Agent 执行 write_file / run_command / git_commit 等需要确认的工具时，用户 allow/deny。

#### 请求体

```json
{
  "session_id": "agent_sess_001",
  "confirmation_id": "confirm_abc123",
  "action": "allow"
}
```

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| session_id | string | 是 | Agent 会话 ID |
| confirmation_id | string | 是 | 待确认操作的 ID |
| action | string | 是 | `allow` 允许 / `deny` 拒绝 |

#### 响应

```json
{
  "success": true,
  "message": "已允许工具执行，Agent 继续运行",
  "result_summary": "写入 src/main.py 成功"
}
```

---

### 27. 工具执行 - execute

**POST** `/v1/tool/execute`

执行工具调用。只读工具直接执行并返回；写操作（write_file / run_command / git_commit）返回 `requires_confirmation=true` + confirmation_id。

#### 请求体

```json
{
  "session_id": "agent_sess_001",
  "tool_call": {
    "name": "read_file",
    "arguments": {"file_path": "src/main.py"}
  }
}
```

#### 响应（只读工具）

```json
{
  "success": true,
  "requires_confirmation": false,
  "tool_name": "read_file",
  "output": {"content": "...", "lines": 150}
}
```

#### 响应（需确认工具）

```json
{
  "success": false,
  "requires_confirmation": true,
  "confirmation_id": "confirm_xyz789",
  "tool_name": "write_file",
  "output": null
}
```

---

### 28. 工具执行 - confirm

**POST** `/v1/tool/confirm`

处理用户对工具的确认操作。与 `/v1/agent/confirm` 类似，区别是不恢复 Agent 循环，仅执行/拒绝工具。

#### 请求体

```json
{
  "session_id": "agent_sess_001",
  "confirmation_id": "confirm_xyz789",
  "action": "allow"
}
```

---

### 29. 终端日志流式（WebSocket）

**WS** `/v1/agent/stream/{session_id}`

命令执行时的实时 stdout/stderr/system 消息流。前端在调用 tool/confirm 前先建立连接。

#### 连接

```
ws://localhost:3000/v1/agent/stream/{session_id}
```

#### 消息格式

```json
{"type": "system", "content": "$ echo hello\n[PID=1234, 超时=60s]", "timestamp": "2026-10-05T10:00:00.000Z"}
{"type": "stdout", "content": "hello\n", "timestamp": "2026-10-05T10:00:00.123Z"}
{"type": "system", "content": "[进程退出码=0]", "timestamp": "2026-10-05T10:00:00.456Z"}
```

- **type**: `stdout` / `stderr` / `system`
- 支持多客户端并发订阅同一 session
- 无订阅者时命令仍能正常执行，output 在 ToolResult 中返回

---

### 30. 沙箱状态查询

**GET** `/v1/sandbox/status`

查询沙箱运行时配置与降级状态。只读接口。

#### 响应

```json
{
  "configured_mode": "docker",
  "active_mode": "host",
  "docker_available": false,
  "fuse_limit": 3,
  "safety_info": {
    "default_timeout_seconds": 60,
    "max_timeout_seconds": 300,
    "danger_patterns_count": 5,
    "network_disabled": true,
    "memory_limit_mb": 512,
    "cpu_limit": 0.5,
    "allow_host_gateway": false
  },
  "degradation_warning": "配置了 SANDBOX_MODE=docker 但 Docker 不可用，已自动降级到 host 模式"
}
```

---

### 31. 调试接口（仅非生产环境）

以下接口仅在 `ENVIRONMENT != production` 时注册，用于测试。

#### GET `/test-error` → HTTP 400

#### POST `/test-validation` → 参数校验回显

#### GET `/test-500` → HTTP 500

---

## 限频说明

后端对所有 `/v1/` 前缀的业务接口实施滑动窗口限频（`/v1/user/usage` 除外）：

| 限制维度 | 限制值 | 说明 |
|----------|--------|------|
| 每分钟 | 20 次 | 按 user_id 限频 |
| 每天 | 500 次 | 按 user_id 限频 |

超限时返回 HTTP 429，Response Header：

| Header | 说明 |
|--------|------|
| X-RateLimit-Limit | 当前窗口的请求上限 |
| X-RateLimit-Remaining | 剩余请求数 |
| X-RateLimit-Reset | 剩余重置秒数 |
| Retry-After | 建议重试等待秒数 |

---

## 错误码说明

| HTTP | 场景 |
|------|------|
| 400 | 请求参数错误 / 业务异常 |
| 401 | 未认证或令牌无效 / API Key 无效 |
| 404 | 资源不存在（如 Agent 会话 / 符号定义） |
| 409 | 资源冲突（如代码库尚未索引） |
| 422 | 请求参数校验失败（如 content_snippet 过长） |
| 429 | 限频超限 |
| 500 | 服务器内部错误 |
| 502 | Planner 生成任务计划失败 |
| 503 | LLM 适配器不可用 |

SSE 流内部错误码：

| code | 描述 |
|------|------|
| 10201 | 服务器内部错误 |
| 各厂商适配器原始错误码 | 透传 |

---

## 中间件说明

请求经过以下中间件链执行（由外到内）：

1. **Request ID** — 为每个请求生成 X-Request-ID，便于链路追踪
2. **JWT Auth** — 校验 Bearer Token，将 payload 存入 `request.state.user_payload`
3. **Rate Limiter** — 从 user_id 读取并更新限频计数
4. **Handler** — 业务处理

所有响应（含 SSE）会携带 `X-Request-ID` Header。

---

## 使用示例

### cURL 示例

```bash
# 1. 获取访问令牌
curl -X POST "http://localhost:3000/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"apiKey": "your_api_key_here"}'

# 2. 验证令牌
curl -X GET "http://localhost:3000/auth/validate" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."

# 3. 使用令牌调用聊天接口（Inline Chat + JSON Mode）
curl -X POST "http://localhost:3000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
     -d '{
         "messages": [
             {"role": "user", "content": "给 main.py 加一个 greet 函数"}
         ],
         "mode": "inline",
         "inline_selection": {
             "file_path": "src/main.py",
             "selected_text": "def hello(): pass",
             "start_line": 10,
             "end_line": 10
         },
         "response_format": {"type": "json_object"},
         "stream": true
     }'

# 4. 触发代码索引
curl -X POST "http://localhost:3000/v1/index/start" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
     -H "Content-Type: application/json" \
     -d '{"workspace_root": "/path/to/project"}'

# 5. 查询用量
curl -X GET "http://localhost:3000/v1/user/usage" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."

# 6. Agent 完整流程
curl -X POST "http://localhost:3000/v1/agent/plan" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
     -H "Content-Type: application/json" \
     -d '{"goal": "给项目加日志功能", "workspace_root": "/path/to/project"}'

curl -X POST "http://localhost:3000/v1/agent/start/{session_id}" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."

curl -X GET "http://localhost:3000/v1/agent/status/{session_id}" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

### JavaScript 示例（SSE 流）

```javascript
async function chatInlineChat() {
  const response = await fetch('http://localhost:3000/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${accessToken}`
    },
    body: JSON.stringify({
      messages: [{ role: 'user', content: '把 hello 改成 greet' }],
      mode: 'inline',
      inline_selection: {
        file_path: 'src/main.py',
        selected_text: 'def hello(): pass',
        start_line: 10,
        end_line: 10
      },
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // 保留未完成的行

    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') return;

      const chunk = JSON.parse(data);

      // type:meta — references（在第一个 content chunk 之前）
      if (chunk.type === 'meta') {
        console.log('References:', chunk.references);
        continue;
      }

      // type:diff — 多文件修改的 Diff
      if (chunk.type === 'diff') {
        console.log('Diff files:', chunk.files);
        continue;
      }

      // 标准 content chunk
      if (chunk.choices?.[0]?.delta?.content) {
        process.stdout.write(chunk.choices[0].delta.content);
      }

      // error chunk
      if (chunk.error) {
        console.error('Error:', chunk.error);
      }
    }
  }
}
```

---

## 注意事项

1. **JWT 认证**: 除 `/health` 和 `/auth/*` 外，所有 `/v1/` 接口均需 JWT 认证
2. **流式响应**: 聊天接口默认流式，使用 SSE 格式；非流式暂不支持
3. **生产环境**: `ENVIRONMENT=production` 时禁用 `/docs`、`/redoc`、`/config-check`、调试接口
4. **CORS**: 当前允许 `http://localhost:3000` 和 `http://localhost:5173`，生产环境应限制具体域名
5. **Gzip**: 中间件已开启 Gzip 压缩（≥1KB 响应），降低 SSE 包体积
6. **Redis**: 默认关闭，开启后可提升限频、配额、索引进度的多实例一致性；未开启自动降级内存
7. **沙箱**: 默认 Docker 模式，不可用时自动降级 host；熔断阈值 3 次连续失败
8. **熔断**: Agent 连续 3 次失败自动暂停循环，全局 20 次自修复总上限强制终止

---

## 更新日志

### v0.1.0 ~ v0.12.0 (2024-2026)

- **S1 基础**: 健康检查、配置检查、JWT 认证、聊天代理、全局异常处理
- **S2 会话**: 会话管理、滑动窗口裁剪、上下文注入（@文件/选中/隐式）、token 精确计数
- **S3 安全**: 用户用量查询、多厂商模型列表、全局限频、模式化请求（chat/new）
- **S4 代码索引**: 全量/增量索引、向量语义搜索、依赖关系图（related/imports/stats）
- **S5 符号系统**: 符号实时补全、符号定义定位、反向依赖查询、混合检索（向量+BM25+符号+RRF+Cross-Encoder）
- **S6 Inline & JSON**: Inline Chat 模式、多文件 JSON Mode + Diff、Cue 编辑位置预测
- **S7 Agent 骨架**: Planner 任务规划、状态机、ReAct 执行循环、start/pause/resume/ask_user
- **S8 工具层**: 工具 execute/confirm、Write 安全确认链路、终端日志 WebSocket、Docker 沙箱、熔断机制
- **S9 自修复**: 自修复循环引擎、测试沙箱集成（run_tests）、修复无效检测
