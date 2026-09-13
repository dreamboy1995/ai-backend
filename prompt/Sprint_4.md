
## Sprint 4
这个 Sprint 是技术难度陡增的一环，标志着从 "插件玩具" 向 "代码理解引擎" 的跨越。我们要让 AI 不仅能看到你当前打开的文件，还能通过 AST 解析 + 向量化，"看懂"整个代码仓库的符号表（类、函数、变量）和依赖关系。

### 核心目标（S4）
代码库索引基建（本地优先）：搭建基于 `tree-sitter` 的多语言 AST 解析引擎；实现"语义代码切片"并按函数/类边界向量化；搭建本地向量数据库（推荐 LanceDB 免安装），完成代码库的首次静默索引。


### 👥 人员分工（S4 需增加算法侧人力）
- 后端/算法开发（1-2人）：tree-sitter 解析器封装、切片策略、向量化服务（Embedding）、向量数据库读写。
- 插件开发（1人）：索引进度反馈 UI、文件变更监听（触发增量索引）、`#符号搜索` 的输入框交互（为 S5 做准备）。


### 📅 每日拆解任务（第 31-40 个工作日）
#### 第 31-32 天（tree-sitter 多语言解析器基建）
| 角色 | 具体任务（粒度到代码模块） | 验收标准 |
|------|--------------------------|---------|
| 后端/算法 | 1. 初始化 Python 后端项目，安装 tree-sitter 核心库及各语言包：`pip install tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript tree-sitter-java tree-sitter-go`<br>2. 编写 `parser_factory.py`，实现语言自动检测（根据文件后缀返回对应 Parser）。<br>3. 编写 `ast_parser.py`，输入文件路径，输出符号表（Symbol Table）：<br>  - 类名（Class）、方法/函数名（Function）、全局变量（Variable）。<br>  - 记录每个符号的起始行号、结束行号。<br>4. 编写测试用例：用 `sample.py` 解析，打印出所有函数名和行号范围。 | 运行 `python test_parser.py --file sample.py`，控制台输出：<br>`Function: calculate_sum (Lines 5-12)<br>Function: main (Lines 15-22)<br>Class: DataProcessor (Lines 25-45)` |
| 插件 | 1. 在 Webview 设置面板中，预留"索引状态"显示区域（如"代码库索引中: 0/150 文件"）。<br>2. 编写 `fileWatcher.ts`，监听 `vscode.workspace.onDidSaveTextDocument` 和 `onDidCreateFiles` 事件，记录变更文件列表（暂不触发后端，等第 37-38 天集成）。 | 当用户保存文件时，插件后台 Console 能打印出 `[File Changed] /path/to/file.py`。 |

⚠️ 注意：`tree-sitter` 的 Python 绑定在不同操作系统上编译可能报错（尤其 Windows）。务必在 `requirements.txt` 中锁死版本（如 `tree-sitter==0.20.4`），并准备 Docker 开发环境统一构建，或使用 `tree-sitter` 官方提供的预编译 `wheel`。


#### 第 33-34 天（代码切片 & 语义分块策略）
这是决定检索质量（Recall）的关键环节，不是简单按 500 字符切块，而是按语法边界切分。

| 角色      | 具体任务（核心算法）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 验收标准 |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| 后端/算法 | 1. 编写 `code_chunker.py`，实现语义切片（Semantic Chunking）策略：<br>  - 最小单元：单个函数/方法体（若函数行数 < 50 行，则整个函数作为一个 Chunk）。<br>  - 大函数拆分：若函数超过 200 行，按逻辑块（循环/条件分支）进一步切分。<br>  - 文件头聚合：将文件顶部的 import 语句和全局变量作为独立 Chunk。<br>  - 每个 Chunk 记录元数据：`{ file_path, symbol_name, chunk_type: 'function'\|'class'\|'import', start_line, end_line }`。<br>2. 编写 `embedding_client.py`，加载轻量级本地 Embedding 模型（推荐 `all-MiniLM-L6-v2`，约 80MB，CPU 可跑），将每个 Chunk 的文本内容转为 384 维向量。<br>3. 测试单个文件：切片后向量维度正确，且向量值不全为 0。<br> | 对一个 300 行的 Python 文件跑切片，产出 5-8 个 Chunk，每个 Chunk 的 `embedding` 数组长度为 384，且数值有变化（非全零）。 |
| 后端/算法 | 4. （备选方案）若本地 Embedding 模型加载过慢（>2s），则切换到调用远程 API（如 OpenAI `text-embedding-3-small`），但需将向量缓存到本地 SQLite，避免重复计算。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |首次加载模型 < 3 秒，后续调用 < 200ms。|

💡 小技巧：函数体内的注释和文档字符串（docstring）对检索至关重要。切片时务必保留它们，甚至可以将函数签名（`def foo(x: int) -> bool`）单独提取作为"标题"，与函数体拼接后一起向量化，能大幅提升符号搜索的准确率。


#### 第 35-36 天（向量数据库选型与 CRUD 封装）
| 角色 | 具体任务（数据持久化） | 验收标准 |
|------|----------------------|---------|
| 后端/算法 | 1. 选型确定：由于 P2 阶段要求"免安装、嵌入式"，选定 LanceDB（基于 Lance 列式存储，纯文件，无外部依赖）作为向量库。`pip install lancedb`<br>2. 编写 `vector_store.py`，封装以下方法：<br>  - `create_table(table_name, schema)`<br>  - `insert_chunks(chunks: List[Chunk], embeddings: List[Array])`<br>  - `search(query_embedding, top_k=10)` 返回相似 Chunk 列表<br>  - `delete_by_file(file_path)`（文件删除/重命名时清理旧向量）<br>3. 设计数据表 Schema（LanceDB 支持 Pydantic 风格）：<br>```python<br>class CodeChunk(BaseModel):<br>    id: str            # 文件路径 + 符号名 + 行号的哈希<br>    file_path: str<br>    symbol_name: str<br>    chunk_type: str    # function, class, import<br>    content: str       # 切片文本<br>    start_line: int<br>    end_line: int<br>    embedding: List[float]  # 向量字段<br>``` | 写入 10 个测试 Chunk，然后用查询向量搜索，能按相似度排序返回正确结果（比如搜 "sort function" 能返回排序相关的函数）。 |
| 插件 | 1. 无新功能开发。配合后端验证文件路径传递格式（统一使用相对路径，如 `src/main.py`）。 | - |


#### 第 37-38 天（全仓库首次索引 & 增量更新机制）
这是 "索引基建" 的集成冲刺。

| 角色 | 具体任务（性能与工程化） | 验收标准 |
|------|-------------------------|---------|
| 后端/算法 | 1. 编写 `index_orchestrator.py`，驱动全仓库索引流程：<br>  a. 遍历工作区所有文件（排除 `.git`、`node_modules`、`__pycache__`、大于 1MB 的文件）。<br>  b. 每解析 10 个文件，将向量批量提交（Batch Insert）到 LanceDB（减少 IO 开销）。<br>  c. 将索引进度（`{ total: 500, processed: 120 }`）写入 Redis，供插件轮询展示。<br>2. 实现增量更新：当插件通过 HTTP 通知后端"文件 xxx.py 已变更"，后端仅重新解析该文件，删除旧向量，插入新向量。 | 在 100 个文件的测试仓库上运行索引，耗时 < 30 秒（不含 Embedding 模型加载时间）。修改其中一个文件后，增量索引 < 2 秒完成。 |
| 插件 | 1. 在 VS Code 启动时（`activate` 函数中），检测当前工作区是否有 `.ai_index` 缓存标记。若无，则发送 `POST /index/start` 触发首次索引。<br>2. 轮询 `GET /index/status`（每 2 秒），在 Webview 底部状态栏显示"索引中 45% (120/265 文件)"。<br>3. 首次索引完成后，在状态栏显示"✅ 代码库已索引 (265 个符号)"。 | 打开一个中型项目（如 React 脚手架），插件右下角出现进度条，从 0% 平滑走到 100%，完成后显示绿色对勾。 |

⚠️ 性能优化：Embedding 模型在 CPU 上跑批量向量化时，用 `sentence-transformers` 的 `encode` 方法时务必开启 `show_progress_bar=False` 和 `batch_size=32`，避免控制台刷屏和内存溢出。


#### 第 39-40 天（依赖关系图构建 & 联调回归）
为了让后续的 "代码搜索" 更聪明，我们不能只靠向量，还要建立符号引用图（Call Graph）。

| 角色 | 具体任务（进阶索引） | 验收标准 |
|------|---------------------|---------|
| 后端/算法 | 1. 在 AST 遍历过程中，额外提取符号引用关系：<br>  - 在 `a.py` 中 `import b` → 记录边 `a.py -> b.py`。<br>  - 在 `main()` 中调用了 `utils.parse()` → 记录边 `main -> utils.parse`。<br>2. 将依赖图存入 Neo4j（或简单存入 JSON 文件，MVP 阶段建议用 JSON + 内存缓存，避免引入过多中间件）。<br>3. 提供接口 `GET /graph/related?file=main.py&depth=2`，返回与该文件强关联的上下游文件列表。 | 对项目根目录的 `app.py` 调用依赖图接口，能返回它 import 的所有本地模块名。 |
| 插件 | 1. 为输入框增加 `#` 前缀触发（为 S5 的符号搜索做准备）：输入 `#` 时，弹出当前索引中的所有符号名（函数/类列表），用户选中后可作为上下文附加。<br>2. （仅 UI 层，暂不接入后端搜索逻辑） | 输入 `#` 后，下拉框弹出 `main`, `DataProcessor`, `calculate_sum` 等符号名称（数据由后端 mock 提供即可）。 |


### 📂 关键接口/数据结构变更（S4 新增）
1. 索引控制接口（后端新增）
```typescript
// 触发全量索引
POST /v1/index/start
Request: { workspace_root: string, force_rebuild?: boolean }
Response: { job_id: string, total_files: number }

// 查询索引进度
GET /v1/index/status
Response: { status: 'idle'|'indexing'|'done'|'error', total: number, processed: number, percentage: number }

// 增量更新通知（由插件在文件保存时调用）
POST /v1/index/update
Request: { file_path: string, action: 'modified'|'deleted'|'renamed' }
```

2. 插件端数据结构（GlobalState 新增）
```typescript
interface GlobalState {
  // ... 之前字段
  repoIndexed: boolean;           // 当前仓库是否已索引完成
  repoRootHash: string;           // 工作区路径的哈希，切换项目时重置索引状态
  totalSymbols: number;           // 索引符号总数，用于状态栏展示
}
```


### 🔧 S4 关键技术预研与风险预警（研发必读）
| 风险点 | 应对方案 |
|--------|---------|
| tree-sitter 解析失败 | 部分不规范的代码（残缺语法）会导致解析抛出异常。必须用 `try-except` 包裹，失败时降级为纯文本按行切块（正则分割空行），保证索引不中断。 |
| 本地 Embedding 模型内存占用 | `all-MiniLM-L6-v2` 约 80MB，对大多数开发机可接受。但若用户内存紧张（4GB），可在设置中提供开关："启用本地索引（消耗 ~100MB 内存）"默认开启。 |
| 首次索引耗时过长 | 大型 Monorepo（10000+文件）首次索引可能需要数分钟。务必实现"即用即索引"策略：优先索引用户当前打开的文件及其 import 的直接依赖，后台静默处理剩余文件。 |
| 向量维度不兼容 | 若未来切换 Embedding 模型（如升级到 BAAI/bge-large 1024 维），LanceDB 无法自动迁移。需要在 Chunk 表中预留 `embedding_version` 字段，支持版本重建。 |


### ✅ Sprint 4 结束时的 Demo 检查清单（Showcase）
- [ ] 后端能正确解析 Python / JavaScript / TypeScript / Java / Go 文件的 AST，提取完整符号表。
- [ ] 对工作区执行全量索引，进度条在状态栏平滑更新（无卡死）。
- [ ] 索引完成后，修改一个文件并保存，后端日志显示增量更新成功（旧向量删除，新向量插入）。
- [ ] 向量搜索 API（`/v1/search?q=排序&top_k=5`）能返回语义相关的函数/类切片（即使关键词不完全匹配）。
- [ ] 插件中输入 `#`，能弹出符号名下拉列表（UI 交互雏形）。
- [ ] 索引目录跳过了 `node_modules`、`.git` 等大型无关目录（验证 `.gitignore` 集成逻辑）。

S4 做完后，你的后端就拥有了一个"可查询的代码语义索引库"。虽然用户还感知不到它的存在（因为还没接入 Chat），但它为 S5（混合检索 RAG）和 S6（Inline Chat / 多文件 Diff）铺平了道路。
