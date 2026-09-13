## Sprint 1 核心目标（S1）
启动基建：完成 VS Code 插件壳子加载 Webview，后端网关启动，打通"插件 -> 后端 -> 第三方大模型（ZAI）"的流式 SSE 链路，实现"你好，世界"的问答返回。


### 👥 人员分工假设
- 后端开发（1人）：负责 API 网关、鉴权、代理第三方模型。
- 插件开发（1人）：负责 VS Code 扩展、Webview UI、通信桥梁。


### 📅 每日拆解任务（第 1-10 天）
#### 第 1-2 天（项目初始化 & 骨架搭建）
| 角色	 | 具体任务（粒度到文件/命令）                                                                                                                                                                                                                                                           | 	验收标准                                                                          |
|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 后端	 | 1. 初始化项目：`mkdir ai-backend && cd ai-backend`，使用 FastAPI（Python）或 NestJS（TS）。<br>2. 配置 `.env` 文件（`PORT=3000`, `ZAI_API_KEY=xxx`, `JWT_SECRET=xxx`）。<br>3. 编写基础 `main.py` / `app.module.ts`，挂载全局异常过滤器。<br>4. 生成 OpenAPI（Swagger）文档路由（`/docs`）。 | 	服务启动后，浏览器访问 `http://localhost:3000/docs` 能看到 Swagger 页面。|
| 插件	 | 1. 全局安装 `npm install -g yo generator-code`。<br>2. 生成项目：`yo code`，选择 `TypeScript`，配置 `webview` 模板。<br>3. 安装 UI 依赖：`npm install react @types/react @types/vscode`。<br>4. 编写 `package.json` 的 `contributes.viewsContainers`，将 Webview 注册到侧边栏。                     | 按 `F5` 启动调试，VS Code 侧边栏出现你的插件图标，点击能打开一个空白 Webview 面板。 |


#### 第 3-4 天（通信管道搭建）
| 角色 | 	具体任务（核心代码/接口定义）	                                                                                                                                                                                                                                                                                               | 验收标准                                                                                            |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| 后端	 | 1. 定义标准 SSE 流式 DTO（数据传输对象）：`python<br># Pydantic 模型<br>class ChatChunk:<br> id: str<br> choices: List[Dict] # [{"delta": {"content": "你好"}}]<br> usage: Optional[Dict] = None<br>`<br>2. 编写 `/v1/chat/completions` 路由，接收 `{messages, model, stream}`，先返回 Mock 流式数据（每 200ms 吐一个字）。	 | 使用 Postman 或 curl 请求该接口，`stream=true` 时能收到 `data: {"choices":...}\n\n` 格式的 SSE 流。 |
| 插件	 | 1. 在 Webview 的 `App.tsx` 中实现 `sendMessage` 函数。<br>2. 使用 `vscode.postMessage` 将用户输入发送给 Extension Host（插件的后台脚本）。<br>3. 在 `extension.ts` 中监听 `onDidReceiveMessage`，暂时只做日志打印（`console.log`）。                                                                                        | 	在 Webview 输入框打字点击发送，查看 VS Code 的调试控制台（Terminal），能看到打印出的用户消息对象。  |


#### 第 5-6 天（流式 UI 渲染 & 真实代理）
|角色	|具体任务（核心逻辑）|	验收标准|
|---|---|---|
|后端|	1. 替换 Mock 逻辑，编写 `zai_adapter.py`。<br>2. 使用 `httpx.AsyncClient` 异步请求 ZAI 官方 API。<br>3. 关键：实现后端 SSE 与 ZAI 官方 SSE 格式的透传转换（统一转成 OpenAI 标准格式），并增加 `request_id` 用于后续链路追踪。|	后端日志显示成功拿到 ZAI 返回的 `data: [DONE]`，且没有报错。|
|插件|	1. 在 Extension Host 侧，使用 `Node.js` 原生 `fetch` 请求后端 `/v1/chat/completions`。<br>2. 利用 `ReadableStream` 解析 SSE 数据块，通过 `postMessage` 将增量文本（`delta.content`）逐字推给 Webview。3. Webview 使用 `setState` 实现打字机效果（逐字追加到消息气泡）。|	插件中输入 `"Hello"`，界面上 1-2 秒后开始逐个蹦出 ZAI 返回的文字。|


#### 第 7-8 天（鉴权与配置持久化）
|角色	|具体任务（产品化细节）	|验收标准|
|---|---|---|
|后端	|1. 实现 JWT 鉴权中间件（拦截除 `/health` 外的所有路由）。<br>2. 增加 `/auth/api-key` 接口，前端传递用户填的 API Key，后端校验有效性（可先 Ping 一下模型接口）并颁发 JWT。<br>3. 引入 Redis 或内存 Cache 存储用户的 Token 黑名单（用于注销）。|	未携带 JWT 的请求返回 `401 Unauthorized`；携带正确 Token 的请求正常通过。|
|插件	|1. 利用 `vscode.ExtensionContext.globalState` 存储用户的 API Key 和 模型选择（如 zai/glm-4.5-air）。<br>2. 在 Webview 加载时，先检查 `globalState` 中是否有 Key，若无则弹出一个输入框（`vscode.window.showInputBox`）。<br>3. 将用户输入的 Key 发送给后端换取 JWT，并存回 `globalState`（下次启动免登录）。	|关闭 VS Code 再打开，插件能自动读取 Key 并显示"已登录"状态，无需反复输入。|


#### 第 9-10 天（异常处理、日志与联调交付）
|角色	|具体任务（稳定性保障）	|验收标准|
|---|---|---|
|后端	|1. 封装全局错误码（如 10001: 模型限流, 10002: Token 超长）。<br>2. 在 SSE 流中增加错误块：`data: {"error": {"code": 10001, "msg": "Rate limit"}}`，确保流中断时 UI 能感知。<br>3. 增加超时熔断（·httpx.timeout=60s·）。	|手动断网或让模型超时，前端能收到明确的错误提示，而不是卡死。|
|插件	|1. Webview 增加停止生成按钮（点击后发信号给后端取消  `httpx` 请求）。<br>2. 处理连接断开重试：若 SSE 断开，自动重试 1 次。<br>3. 编写 Sprint 1 演示脚本（生成一个简单的 Python 斐波那契函数）。|	所有异常情况（Key错误、网络超时、取消）均有友好的 Toast 通知，无白屏/红屏。|


### 📂 关键接口定义（代码级契约）
为了让前后端并行不悖，S1 第一天就必须锁定以下接口，建议使用 OpenAPI 生成 TypeScript 类型供插件端直接 import。

1. 获取 Token 接口
```typescript
POST /auth/token
Request: { apiKey: string; model?: string }
Response: { accessToken: string; expiresIn: number }
```

2. 核心对话流式接口
```typescript
POST /v1/chat/completions
Headers: { Authorization: "Bearer {accessToken}" }
Request: {
  messages: Array<{ role: "user"|"assistant"|"system", content: string }>;
  stream: true; // S1 强制 true
  temperature?: number;
}
// Response: text/event-stream
// 数据块格式（严格对齐 OpenAI）：
// data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}
// data: [DONE]
```

3. VS Code 插件内部通信协议（Webview <-> Extension）
```typescript
// Webview -> Extension
interface WebviewMessage {
  type: 'send-prompt';
  payload: { text: string; fileContext?: string };
}

// Extension -> Webview
interface ExtensionMessage {
  type: 'stream-chunk' | 'stream-end' | 'stream-error';
  payload: { content?: string; errorCode?: number; msg?: string };
}
```


### 🔧 研发环境准备清单（S1 启动会必做）
- 后端：申请 ZAI 的 API Key（建议准备 2 个测试账号防限流）。
- 插件：配置 VS Code 的 launch.json，确保 "runtimeExecutable" 指向本地编译好的插件宿主。
- 联调：使用 whistle 或 Charles 配置代理，让插件请求转发到本地 localhost:3000，避免 CORS 问题（或后端直接配 allow_origins=["*"]）。


### ✅ Sprint 1 结束时的 Demo 检查清单（Showcase）
- [ ] 用户在 VS Code 侧边栏打开插件，输入 API Key 完成登录。
- [ ] 输入"用 Python 写一个快速排序"，AI 以打字机效果逐字输出完整代码。
- [ ] 点击"停止生成"按钮，响应立即终止且无报错。
- [ ] 断开本地后端服务，界面出现友好的"服务连接失败，请检查网络"重试按钮。
- [ ] 关闭 VS Code 再打开，登录态依然保持（无需二次输入 Key）。
