
## Sprint 2
这个 Sprint 的核心任务是 "对话上下文基础 & 文件操作"，目标是让 AI 能看到用户当前正在编辑的文件内容，并能把生成的代码一键插入到编辑器里。

### 核心目标（S2）
上下文注入与代码落地：实现 @文件 / @选中代码 的上下文引用；后端建立会话记忆（多轮对话裁剪）；前端实现“Apply”按钮，将 AI 回复中的代码块插入光标处。


### 👥 人员分工（延续 S1）
- 后端开发（1人）：会话管理（Session）、历史消息滑动窗口裁剪、上下文拼装策略。
- 插件开发（1人）：编辑器内容提取、@提及 UI 交互、代码块解析与 Apply 编辑器操作。


### 📅 每日拆解任务（第 11-20 个工作日）
#### 第 11-12 天（后端会话管理 & 历史记忆）
| 角色 | 具体任务（粒度到代码模块） | 验收标准 |
|------|--------------------------|---------|
| 后端 | 1. 安装 Redis 客户端（或使用内存 Map 做简易版，但生产建议 Redis）。<br>2. 创建 `SessionService` 类，提供 `create(session_id)`、`append(session_id, msg)`、`get_history(session_id)` 方法。<br>3. 实现滑动窗口裁剪算法：<br>  - 设定总 Token 预算（如 8000 tokens，为 DeepSeek 留余量）。<br>  - 使用 `tiktoken` 库精确计数。<br>  - 策略：保留 System Prompt + 最近 5 轮对话，超出部分从最旧的消息开始丢弃（保留用户最新追问）。 | 后端日志能打印出每次请求裁剪后的消息条数。连续对话 20 轮后，请求体中的 `messages` 数组长度稳定在 12 条左右（5轮 user+assistant+system），不无限膨胀。 |
| 插件 | 1. 在 Webview 的 Chat UI 中，为每条消息增加时间戳显示（便于用户感知会话连续性）。<br>2. 前端每次发送请求时，携带 `session_id`（可简单用 uuid 生成，并存于 `globalState` 中）。 | 刷新 Webview（不关闭 VS Code）后，依然能带同一 `session_id` 请求，后端能识别出是同一会话。 |


#### 第 13-14 天（插件端上下文提取器 - @文件 & @选中）
| 角色 | 具体任务（核心交互） | 验收标准 |
|------|---------------------|---------|
| 插件 | 1. 在 Webview 输入框中监听键盘事件，识别用户输入 `@` 符号。<br>2. 弹出快速选择菜单（QuickPick），列出当前工作区打开的所有 Tab 文件名和当前光标选中的代码片段。<br>3. 选中的文件/代码以 Chip 标签（如 `@main.py` 蓝色圆角矩形）显示在输入框中，支持 Backspace 删除。<br>4. 构建 `ContextPayload` 对象：`{ type: 'file' \| 'selection', path: string, content: string, language: string }` | 输入 `@` 能弹出文件列表，选中 `main.py` 后，输入框出现蓝色标签。 |
| 后端 | 1. 预先调整 `/v1/chat/completions` 的 DTO，增加 `contexts` 可选字段（数组），为第 15-16 天的拼接做准备。 | API 文档（Swagger）更新了 `ContextItem` 的 Schema 定义。 |

⚠️ 关键技术点：读取大文件时，必须在插件端做截断（只取头尾各 200 行 + 光标所在行附近 50 行），绝对禁止把 1 万行的文件全量塞进去，否则后端 Token 直接爆炸。


#### 第 15-16 天（后端上下文拼装 & 系统提示词工程）
| 角色 | 具体任务（核心算法） | 验收标准 |
|------|---------------------|---------|
| 后端 | 1. 编写 `ContextBuilder` 模块。<br>2. 将前端传来的 `contexts` 数组，格式化为结构化 XML 标签插入 System Prompt 中：`<context_files><file path="main.py" lang="python">def hello():\n    print("hello")</file></context_files>`<br>3. 关键策略：将拼接后的 System Prompt 也纳入 Token 裁剪计算中（保证总 Token 不超限）。<br>4. 若上下文过长，按文件优先级丢弃（用户主动 `@` 的保留，自动附带的当前文件可截断）。 | 当用户 `@` 了 `a.py`，后端日志中打印出的 System Prompt 包含 `a.py` 的完整（或截断）内容，模型回答能基于该文件内容展开。 |
| 插件 | 1. 增加编辑器事件监听：用户切换 Tab 或移动光标时，自动将当前激活文件作为隐式上下文（不显示 `@` 标签，但默默传给后端）。<br>2. 在发送请求的 Payload 中，增加 `implicitContext` 字段。 | 用户不输入任何 `@`，直接问"这个文件里的函数是干嘛的"，AI 能根据当前编辑器的文件正确回答。 |


#### 第 17-18 天（Apply 功能 - 代码块插入编辑器）
| 角色 | 具体任务（编辑器操作核心） | 验收标准 |
|------|---------------------------|---------|
| 插件 | 1. 在 Webview 渲染 AI 回复时，用正则 ```` ```(\w+)\n([\s\S]*?)``` ```` 提取所有代码块。<br>2. 在每个代码块下方渲染"插入 (Apply)"按钮。<br>3. 点击按钮时，调用 VS Code 核心 API：`const edit = new vscode.WorkspaceEdit(); edit.insert(uri, position, codeContent); await vscode.workspace.applyEdit(edit);`<br>4. 处理边界情况：<br>  - 若当前无激活编辑器，自动用代码语言创建新文件（如 `Untitled-1.py`）。<br>  - 若用户选中了文本，Apply 是替换选中区域还是插入在后面？（默认替换）。 | 在 Chat 中生成一段 Python 代码，点击"插入"，代码精准出现在光标位置，且语法高亮正常。 |
| 后端 | 1. 无需改动，但需要确保模型返回的 Markdown 代码块格式严格（语言标签正确），否则前端正则匹配失败。可在 System Prompt 中强调"请始终使用 ```language 围栏"。 | - |

⚠️ VS Code 并发冲突警告：`applyEdit` 是异步的，如果在 AI 流式输出尚未结束时用户连点 3 次 Apply，可能引发编辑冲突。前端需做防抖（debounce），点击后按钮置灰，等编辑完成再恢复。


#### 第 19-20 天（异常处理、联调与 S2 Demo 准备）
| 角色 | 具体任务（产品质量） | 验收标准 |
|------|---------------------|---------|
| 插件 | 1. 处理超大文件 Apply 卡顿：若代码块超过 500 行，改为调用 `vscode.workspace.openTextDocument` 新建文件写入，而非在当前编辑器插入（防止 UI 冻结）。<br>2. 优化 Webview 渲染性能：长对话列表采用 `react-window` 虚拟滚动（若消息超过 30 条）。<br>3. 修复 S1 遗留的 SSE 断连后 UI 状态不一致问题（增加 `reconnecting` 状态）。 | 快速连发 10 条消息，Webview 滚动流畅不卡顿。 |
| 后端 | 1. 增加会话过期机制：Redis 中的 Session 设置 `TTL=1 小时`，过期自动清理。<br>2. 增加日志链路追踪：每个请求带上 `X-Request-ID`，前后端打通，方便排查问题。 | 运行 S2 完整测试用例：<br>1. `@main.py` + "给这个类加一个 `__str__` 方法"。<br>2. 生成的代码点击 Apply 插入成功。<br>3. 继续追问"再加一个属性"，AI 记得之前的对话背景。 |


### 📂 关键接口/数据结构变更（S2 新增）
1. 请求 DTO 增加上下文数组
```typescript
// POST /v1/chat/completions
interface ChatRequestV2 {
  session_id: string;           // S2 新增
  messages: Message[];
  contexts?: {                  // S2 新增
    type: 'file' | 'selection' | 'implicit';
    file_path: string;
    content_snippet: string;    // 截断后的内容
    language?: string;
  }[];
  stream: true;
}
```

2. 插件端 Webview 状态管理新增字段
```typescript
interface ChatState {
  sessionId: string;            // 持久化存储
  contexts: ContextChip[];      // @标签数组
  isStreaming: boolean;
  isApplying: boolean;          // 防抖用
}
```


### 🔧 S2 关键技术预研（研发必读）
- tiktoken 计数：后端必须引入 tiktoken 或 transformers 的 Tokenizer，绝对不能用字符串长度 / 4 来估算，DeepSeek 的 Token 比例跟英文/中文差异很大。建议在裁剪时 多留 20% 余量（目标 8000，实际 6400 就触发裁剪）。

- @ 提及的性能：vscode.workspace.textDocuments 在大型 Monorepo（1000+文件）中遍历会卡，需加 防抖（300ms） 和 最大匹配数限制（50个）。

- 代码块正则陷阱：Markdown 中可能有 ``` 嵌套（比如展示命令行），正则建议用 非贪婪匹配：/```(\w+)?\n([\s\S]*?)```/g，并过滤掉 json / shell 等非编程语言（避免把终端命令插进代码文件）。


### ✅ Sprint 2 结束时的 Demo 检查清单（Showcase）
- [ ] 用户打开一个新项目，在侧边栏输入 @，能弹出当前编辑的文件列表。
- [ ] 选中 utils.js 后，问 "这个文件的 formatDate 函数时区有问题吗？"，AI 能准确指出代码逻辑。
- [ ] 让 AI "写一个冒泡排序"，生成代码块下方出现 "Apply" 按钮，点击后代码插入到当前光标位置。
- [ ] 连续追问 8 轮，AI 依然记得最开始提到的文件名和变量名（验证滑动窗口裁剪没把关键上下文丢掉）。
- [ ] 断开网络重连后，重试机制正常，没有重复插入代码块。
- [ ] 关闭 VS Code 再打开，之前的会话记录清空（或提示 "新会话"），不保留过期 Session 占用内存。

S2 做完后，你的插件就已经具备了 "看懂当前代码 + 插入代码" 的基础能力，用户留存率会比纯聊天工具高一大截。

