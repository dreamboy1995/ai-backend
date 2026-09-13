
## Sprint 3
这个 Sprint 的核心任务是 "产品化闭环"：让用户可以通过一条指令（`/new`）从零生成一个完整的、可保存的代码文件，同时把"选择模型"和"用量感知"这些产品体验补全，让插件从"技术Demo"变成"可内测的工具"。

### 核心目标（S3）
单文件生成 & 产品配置化：实现 `/new` 指令（一键生成完整单文件并自动保存）；插件端增加模型切换和用量展示；后端增加限频、计费统计和配额管理，为后续小范围内测做准备。


### 👥 人员分工（延续 S1-S2）
- 后端开发（1人）：限频中间件（Redis）、用量统计、模型列表接口、配额管理。
- 插件开发（1人）：`/new` 命令工作流、设置面板（UI）、文件冲突处理、用量进度条。


### 📅 每日拆解任务（第 21-30 个工作日）
#### 第 21-22 天（后端：限频与配额系统）
| 角色 | 具体任务（粒度到代码模块） | 验收标准 |
|------|--------------------------|---------|
| 后端 | 1. 集成 Redis，编写 `RateLimiter` 中间件。<br>2. 实现滑动窗口限频（Lua 脚本保证原子性）：<br>  - 按 `user_id` 限频：每分钟 20 次请求，每天 500 次请求。<br>  - 限频超限时返回 HTTP 429，并在 Response Header 中附带 `X-RateLimit-Reset`（剩余重置秒数）。<br>3. 创建 `QuotaService`，在 Redis 中存储用户当日的 Token 消耗量（`incrby`），并提供 `/v1/user/usage` 接口返回 `{ used_tokens, quota_limit, percentage }`。 | 用脚本在 1 秒内连发 30 次请求，第 21 次起返回 429，且 Header 中带正确重置时间。 |
| 插件 | 1. 在 Webview 底部预留状态栏区域（待第 23-24 天接入用量数据）。 | 状态栏占位 UI 已渲染（显示 "Loading..."）。 |

⚠️ 注意：限频的 `user_id` 在 MVP 阶段可以用 JWT 中的 `sub` 字段，如果还未接入真实用户系统，可以用 `api_key` 的哈希值作为临时标识。


#### 第 23-24 天（插件端：设置面板 & 模型切换）
| 角色 | 具体任务（产品交互） | 验收标准 |
|------|---------------------|---------|
| 插件 | 1. 在 Webview 侧边栏顶部增加齿轮图标（⚙️），点击弹出下拉设置面板（或跳转到独立设置页）。<br>2. 设置面板包含：<br>  - 模型选择下拉框（从后端 `/v1/models` 接口获取列表，硬编码兜底：`deepseek-v3`, `gpt-4o`, `claude-3.5-sonnet`）。<br>  - 温度（Temperature）滑块（0~1，步长 0.1）。<br>  - 显示今日 Token 消耗进度条（调用第 21-22 天开发的 `/v1/user/usage`）。<br>3. 所有设置项存至 `vscode.ExtensionContext.globalState`，每次发送请求时自动带上 `model` 和 `temperature` 参数。 | 切换模型后，发送消息，后端日志中 `model` 字段随之改变。进度条随对话次数增长而更新。 |
| 后端 | 1. 实现 `/v1/models` 接口，返回可用模型列表（含展示名称和上下文长度）。<br>2. 修改 `/v1/chat/completions`，从请求体中读取 `model` 字段，动态路由到对应的第三方 API（适配不同厂商的请求格式差异）。 | 调用 `/v1/models` 返回 `[{id:'deepseek-v3', label:'DeepSeek V3', context_window:64000}]`。 |

🔧 适配器模式：后端需要为每个模型厂商写一个轻量级适配器（`DeepSeekAdapter`、`OpenAIAdapter`、`AnthropicAdapter`），统一转成内部标准格式再透传。


#### 第 25-27 天（插件端核心：/new 指令完整工作流）
这是 S3 最关键的三天，也是 P1 阶段的"王牌功能"。

| 角色 | 具体任务（核心逻辑） | 验收标准 |
|------|---------------------|---------|
| 插件 | 1. 在 Chat 输入框识别 `/new` 斜杠命令：当用户输入以 `/new` 开头时，自动切换为"生成模式"。<br>2. 构建专门的新文件生成 System Prompt（由插件硬编码，不用后端改）：<br>  - 要求模型只输出纯代码，不要 Markdown 围栏（或者只用 ``` 围栏，但插件自动剥离）。<br>  - 要求模型在代码顶部自动添加 shebang（如 `#!/usr/bin/env python3`）和必要的 import。<br>  - 强制要求：输出必须是可直接运行的完整文件，不要省略号（`...`）或注释占位符。<br>3. 生成后的保存流程：<br>  a. 流式输出结束后，解析完整代码文本。<br>  b. 根据代码语言自动推断文件名（如 Python → `main.py`，JavaScript → `index.js`，HTML → `index.html`）。<br>  c. 调用 `vscode.window.showSaveDialog` 让用户选择保存路径（或直接保存到当前工作区根目录，并提供快速覆盖确认）。<br>  d. 使用 `vscode.workspace.fs.writeFile` 写入文件，并自动打开该文件。 | 在输入框输入 `/new` 写一个Python脚本，读取当前目录下的csv文件并打印前5行：<br>1. AI 生成完整的、含 `import csv` 的脚本。<br>2. 弹出保存对话框，默认文件名 `script.py`。<br>3. 保存成功后文件自动在编辑器中打开。<br>4. 文件内容无 `...` 或 `# 此处省略` 等占位符。 |
| 后端 | 1. 无需额外改动，但需要在 System Prompt 中强化"完整输出"的指令（可让插件端在请求末尾追加 `[IMPORTANT] 生成完整可运行代码，禁止使用省略号`）。 | - |

⚠️ 大坑预警：大模型天生倾向于生成"核心片段"而非完整文件，尤其在上下文窗口紧张时。插件端必须在 User Message 尾部强制追加"请输出完整文件内容，不要使用 '...' 或 '// 其余代码不变' 等缩写"，并在流式输出结束后做后置校验：若代码中包含 `...` 或 `省略` 等关键词，自动触发"重新生成"（Retry）一次。


#### 第 28-29 天（冲突处理 & 下载体验打磨）
| 角色 | 具体任务（边界情况） | 验收标准 |
|------|---------------------|---------|
| 插件 | 1. 处理文件重名冲突：若用户选择的保存路径下已存在同名文件，弹出选择框：<br>  - 覆盖（直接覆盖）<br>  - 重命名（自动加 `_1`, `_2` 后缀）<br>  - 取消<br>2. 增加生成中 Loading 动画：在 AI 流式输出时，显示进度条或闪烁光标，并禁用 `/new` 按钮防止连点。<br>3. 增加复制到剪贴板按钮（作为 Apply 之外的备选）。<br>4. 修复 `/new` 与普通 Chat 的上下文污染问题：`/new` 命令应该开启全新的会话上下文（清空历史），避免之前的闲聊干扰代码生成。 | 连续用 `/new` 生成 3 个同名文件，每次都弹出正确的冲突处理对话框。 |
| 后端 | 1. 为 `/new` 请求增加单独的更长超时时间（120秒，普通 Chat 为 60秒），因为单文件生成往往比对话需要更多推理时间。 | - |


#### 第 30 天（联调回归 & S3 Demo 封板）
| 角色 | 具体任务（质量保障） | 验收标准 |
|------|---------------------|---------|
| 全员 | 1. 端到端回归测试：走通 S1（普通对话）→ S2（`@文件` + Apply）→ S3（`/new` 生成 + 保存）全链路。<br>2. 编写插件端错误边界（Error Boundary）：当后端返回 429 限频时，在 Webview 顶部显示黄色横幅"今日免费额度已用完，请明天再试"。<br>3. 更新 `README.md` 和插件 `package.json` 的版本号（`0.1.0`），准备首个内测包（`.vsix`）。 | 完整跑通以下测试用例：<br>1. 普通对话：问"Python 的 list 和 tuple 区别"→ 正常流式输出。<br>2. `@文件`：`@当前文件` + "这个类有什么问题"→ 正确引用。<br>3. `/new` 生成：`/new 写一个flask hello world` → 生成完整 `app.py`，保存后可直接 `python app.py` 运行。<br>4. 限频测试：快速发 25 条消息 → 第 21 条起显示限频提示。 |


### 📂 关键接口/数据结构变更（S3 新增）
1. 后端新增用量查询接口
```typescript
GET /v1/user/usage
Headers: { Authorization: "Bearer {accessToken}" }
Response: {
  user_id: string;
  used_tokens_today: number;
  quota_limit_per_day: number;  // 如 100000
  percentage: number;           // 0.0 ~ 1.0
  reset_at: string;            // ISO 8601 日期，如 "2026-09-10T00:00:00Z"
}
```

2. 插件端 GlobalState 存储结构（新增字段）
```typescript
interface GlobalState {
  apiKey: string;
  jwtToken: string;
  sessionId: string;
  // S3 新增
  selectedModel: string;          // 如 "deepseek-v3"
  temperature: number;           // 0.7
  dailyUsage: { used: number; quota: number; lastUpdated: string };
}
```

3. /new 命令的内部消息构造（插件端硬编码）
```typescript
// 当用户输入以 /new 开头时
function buildNewFilePrompt(userRawInput: string): Message[] {
  const systemPrompt = `
你是一个专业的代码生成助手。用户将通过 "/new" 命令要求你生成一个完整的、可独立运行的代码文件。
**严格要求**：
1. 只输出纯代码内容，不要加 Markdown 围栏（除非用户明确要求）。
2. 必须包含所有必要的 import 语句和依赖声明。
3. 严禁使用 "..." 或 "// 此处省略" 等占位符，必须输出完整代码。
4. 如果用户未指定语言，根据代码内容自动推断。
5. 代码中应包含基础的错误处理（如 try/except）。
`;
  const userPrompt = userRawInput.replace(/^\/new\s*/, ''); // 去掉 /new 前缀
  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt }
  ];
}
```


### 🔧 S3 关键技术预研（研发必读）
- **文件语言推断**：不一定依赖文件后缀，也可以使用 `tree-sitter` 或正则快速判断：如果代码包含 `def` → Python，`function` / `const` → JavaScript，`package main` → Go。插件端必须实现一个简单的 `detectLanguage(code: string): string` 函数。

- **大模型输出截断问题**：DeepSeek 等模型在 `max_tokens` 设置过小时会自动截断代码，后端 `/v1/chat/completions` 的 `max_tokens` 参数要为 `/new` 请求单独调整为 8192 或更高（普通对话可保留 4096）。

- **VS Code 文件写入权限**：`vscode.workspace.fs.writeFile` 在未打开的工作区（没有文件夹）中会报错，此时应降级为 `vscode.window.showSaveDialog`（让用户选路径），这是更稳妥的做法。

### ✅ Sprint 3 结束时的 Demo 检查清单（Showcase）
- [ ] 用户在侧边栏点击齿轮图标，能顺利切换模型（如从 DeepSeek 切到 GPT-4o），且后续对话使用新模型。
- [ ] 设置面板中的 Token 消耗进度条随每一次对话实时更新（后端接口返回准确数据）。
- [ ] 输入 `/new 生成一个快速排序的Python脚本`，AI 输出完整代码，自动弹出保存对话框，默认文件名正确。
- [ ] 保存后的 `.py` 文件在编辑器中打开，无语法错误（可手动 `python` 运行验证）。
- [ ] 当用户疯狂刷消息触发限频时，界面出现友好的黄色提示横幅，而不是白屏或崩溃。
- [ ] 插件打包为 `.vsix` 文件，能在另一台纯净的 VS Code 上成功安装并运行（验证依赖完整性）。

S3 做完后，你的插件就具备了 "开箱即用" 的产品雏形：用户装好插件、填好 Key，就可以用 `/new` 快速生成脚本，用 `@` 询问现有代码，用 Apply 插入代码片段。P1（MVP）阶段至此全部完成，可以开始招募 10-20 名种子用户进行内测了。

S4 将正式进入 P2 阶段，开始做 "tree-sitter AST 解析 + 代码库向量化索引"，这是决定产品"懂代码"能力上限的关键基建，技术难度会比前三个 Sprint 明显提升。
