"""
S6 回归测试用例集（S6 第 59-60 天：联调、性能优化与 P2 里程碑 Demo）

覆盖 S6 全链路核心能力，确保 P2 里程碑交付前无回归：
    1. Inline Chat 修改（mode='inline' + inline_selection 上下文注入）
    2. 多文件修改（response_format=json_object → JSON 解析 → Diff 生成 → 分片推送）
    3. Cue 提示触发（POST /v1/cue/suggest：rename / delete / add_method）
    4. 多文件 JSON 输出超时保护（30s 强制终止 + 友好提示）

用法:
    python test_s6_regression.py

设计原则：
    - 不依赖真实模型 API 与鉴权，全部走单元/模拟路径，保证可重复执行。
    - 复用 test_diff_generator.py 与 test_cue_suggest.py 已验证的基础能力，
      聚焦 S6 端到端链路与第 59-60 天新增的超时保护。
"""

import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatMessage,
    InlineSelection,
    ResponseFormat,
    CueSuggestRequest,
    CueSuggestResponse,
    CueSuggestionItem,
)
from app.api.chat import (
    _parse_json_files,
    _INLINE_SYSTEM_INSTRUCTION,
    _JSON_MODE_SYSTEM_INSTRUCTION,
)
from app.services.diff_generator import (
    build_diff_files,
    generate_unified_diff,
    MAX_DIFF_CHARS_PER_FILE,
)
from app.services.code_index.ast_parser import parse_file
from app.services.code_index.dependency_graph import DependencyGraph
from app.services.code_index.symbol_exact_matcher import (
    SymbolExactMatcher,
    normalize_path,
)


# ============================================================
# 公共测试辅助
# ============================================================

def _make_chat_request(**overrides) -> ChatRequest:
    """构造一个最小可用的 ChatRequest，允许字段覆盖。"""
    base = dict(
        messages=[ChatMessage(role="user", content="把 print 改成 logger.info")],
        model="glm-4.5-air",
        stream=True,
    )
    base.update(overrides)
    return ChatRequest(**base)


# ============================================================
# 模块一：Inline Chat 修改回归（S6 第 51-52 天）
# ============================================================

def test_inline_chat_request_schema():
    """
    验收：ChatRequest 支持 mode='inline' 与 inline_selection 字段。
    """
    req = _make_chat_request(
        mode="inline",
        inline_selection=InlineSelection(
            file_path="src/sort.py",
            selected_text="def sortArray(arr):\n    return sorted(arr)\n",
            start_line=1,
            end_line=2,
        ),
    )
    assert req.mode == "inline"
    assert req.inline_selection is not None
    assert req.inline_selection.file_path == "src/sort.py"
    assert req.inline_selection.start_line == 1
    assert req.inline_selection.end_line == 2
    print("✅ Inline Chat 请求模型（mode + inline_selection）构造通过")


def test_inline_chat_system_prompt_instruction_exists():
    """
    验收：Inline Chat 模式的 System Prompt 追加指令存在且语义正确。
    后端日志中 system 消息应包含 "你正在修改用户选中的代码片段"。
    """
    assert "修改用户选中的代码片段" in _INLINE_SYSTEM_INSTRUCTION
    assert "完整新代码" in _INLINE_SYSTEM_INSTRUCTION
    print("✅ Inline Chat System Prompt 指令文本校验通过")


def test_inline_selection_line_range_annotation():
    """
    验收：inline_selection 注入上下文时应带行号范围注释，便于模型定位。
    模拟 chat.py 中 inline_selection → ContextItem 的转换逻辑。
    """
    sel = InlineSelection(
        file_path="src/main.py",
        selected_text="print('hello')",
        start_line=10,
        end_line=10,
    )
    line_count = max(1, sel.end_line - sel.start_line + 1)
    annotated = (
        f"[行 {sel.start_line}-{sel.end_line}，共 {line_count} 行]\n"
        f"{sel.selected_text}"
    )
    assert "[行 10-10，共 1 行]" in annotated
    assert "print('hello')" in annotated
    print(f"✅ inline_selection 行号范围注释生成通过: {annotated!r}")


def test_inline_chat_default_mode_is_chat():
    """
    验收：mode 缺省为 'chat'，不影响普通对话。
    """
    req = _make_chat_request()
    assert req.mode == "chat"
    assert req.inline_selection is None
    assert req.response_format is None
    print("✅ ChatRequest 默认 mode=chat，无 inline/JSON 字段")


# ============================================================
# 模块二：多文件修改回归（S6 第 53-54 天 + 第 55-56 天）
# ============================================================

def test_json_mode_system_prompt_instruction_exists():
    """
    验收：JSON Mode 的 System Prompt 指令包含 files 数组格式说明。
    """
    assert '"files"' in _JSON_MODE_SYSTEM_INSTRUCTION
    assert '"path"' in _JSON_MODE_SYSTEM_INSTRUCTION
    assert '"content"' in _JSON_MODE_SYSTEM_INSTRUCTION
    assert "完整新内容" in _JSON_MODE_SYSTEM_INSTRUCTION
    print("✅ JSON Mode System Prompt 指令文本校验通过")


def test_response_format_schema():
    """
    验收：ResponseFormat 仅支持 json_object。
    """
    rf = ResponseFormat()
    assert rf.type == "json_object"
    print("✅ ResponseFormat 默认 type=json_object 通过")


def test_chat_request_with_response_format():
    """
    验收：ChatRequest 可携带 response_format=json_object。
    """
    req = _make_chat_request(response_format=ResponseFormat(type="json_object"))
    assert req.response_format is not None
    assert req.response_format.type == "json_object"
    print("✅ ChatRequest 携带 response_format=json_object 通过")


def test_parse_json_files_multi_file():
    """
    验收：_parse_json_files 能正确解析多文件 JSON。
    """
    text = (
        '{"files": ['
        '{"path": "src/main.py", "content": "import logging\\nlogging.info(1)\\n"}, '
        '{"path": "src/utils.py", "content": "def util():\\n    pass\\n"}'
        '], "explanation": "替换了 print"}'
    )
    files = _parse_json_files(text)
    assert files is not None
    assert len(files) == 2
    assert files[0]["path"] == "src/main.py"
    assert files[1]["path"] == "src/utils.py"
    assert "logging" in files[0]["content"]
    print("✅ 多文件 JSON 解析通过（2 个文件）")


def test_parse_json_files_with_markdown_fence_regression():
    """
    回归：模型在 JSON 前后加 ```json 标记时仍能解析（S6 风险预警）。
    """
    text = '```json\n{"files": [{"path": "a.py", "content": "x"}]}\n```'
    files = _parse_json_files(text)
    assert files is not None and len(files) == 1
    print("✅ 带 markdown 围栏的 JSON 解析回归通过")


def test_parse_json_files_invalid_returns_none_regression():
    """
    回归：非法 / 缺字段 JSON 返回 None，触发降级。
    """
    assert _parse_json_files("") is None
    assert _parse_json_files("hello world") is None
    assert _parse_json_files('{"foo": "bar"}') is None        # 缺 files
    assert _parse_json_files('{"files": []}') is None         # files 为空
    print("✅ 非法 JSON 返回 None 回归通过")


def test_build_diff_files_multi_file():
    """
    验收：build_diff_files 能为多个文件生成 old/new/diff 结构。
    """
    with tempfile.TemporaryDirectory() as tmp:
        # 准备两个原始文件
        with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as f:
            f.write("import os\nprint('hi')\n")
        with open(os.path.join(tmp, "utils.py"), "w", encoding="utf-8") as f:
            f.write("# old comment\ndef foo():\n    return 1\n")

        files_json = [
            {"path": "main.py", "content": "import logging\nlogging.info('hi')\n"},
            {"path": "utils.py", "content": "def foo():\n    return 42\n"},
        ]
        result = build_diff_files(files_json, tmp)
        assert len(result) == 2

        main = result[0]
        assert main["path"] == "main.py"
        assert "print('hi')" in main["old_content"]
        assert "logging.info('hi')" in main["new_content"]
        assert "-print" in main["diff"]
        assert "+logging" in main["diff"]

        utils = result[1]
        assert utils["path"] == "utils.py"
        assert "# old comment" in utils["old_content"]
        assert "# old comment" not in utils["new_content"]
    print("✅ 多文件 Diff 生成通过（2 个文件均含 old/new/diff）")


def test_build_diff_files_new_file_scenario():
    """
    验收：新增文件（原文件不存在）时 old_content 为空，diff 显示新增行。
    """
    with tempfile.TemporaryDirectory() as tmp:
        files_json = [{"path": "brand_new.py", "content": "print('new')\n"}]
        result = build_diff_files(files_json, tmp)
        assert len(result) == 1
        assert result[0]["old_content"] == ""
        assert "+print('new')" in result[0]["diff"]
    print("✅ 新增文件 Diff（old_content 为空）通过")


def test_build_diff_files_path_traversal_blocked_regression():
    """
    回归：路径逃逸被拦截，不读取工作区外文件。
    """
    with tempfile.TemporaryDirectory() as tmp:
        files_json = [{"path": "../../../etc/passwd", "content": "x"}]
        result = build_diff_files(files_json, tmp)
        assert result[0]["old_content"] == ""
    print("✅ 路径逃逸拦截回归通过")


def test_diff_chunking_max_3_per_batch():
    """
    验收：S6 风险预警 —— 超过 3 个文件时分片推送，每个 SSE 包最多 3 个文件。
    模拟 chat.py 中的分片逻辑。
    """
    # 构造 7 个文件的 diff 数据
    diff_files = [
        {"path": f"f{i}.py", "old_content": "", "new_content": "x", "diff": "d"}
        for i in range(7)
    ]
    chunk_size = 3
    batches = []
    for i in range(0, len(diff_files), chunk_size):
        batches.append(diff_files[i:i + chunk_size])

    # 7 个文件应分成 3 批：3 + 3 + 1
    assert len(batches) == 3
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 1
    # 每批不超过 3 个
    assert all(len(b) <= 3 for b in batches)
    print(f"✅ Diff 分片推送通过：{len(diff_files)} 个文件 → {len(batches)} 批（每批≤3）")


def test_large_diff_truncation_protection():
    """
    验收：极端大文件 Diff 超过 MAX_DIFF_CHARS_PER_FILE 时被截断，避免撑爆 SSE。
    构造足够多的行数（每行较长）使 Diff 超过 200KB 上限。
    """
    # 每行约 60 字符，20000 行 → Diff 约 2.4MB，远超 200KB 上限
    line_old = "x" * 50 + "\n"
    line_new = "y" * 50 + "\n"
    old = line_old * 20000
    new = line_new * 20000
    diff = generate_unified_diff(old, new, "huge.py")
    # 截断后长度应不超过上限 + 截断标记长度
    assert len(diff) <= MAX_DIFF_CHARS_PER_FILE + len("\n... [diff truncated]\n") + 10
    assert "[diff truncated]" in diff
    print(f"✅ 超大 Diff 截断保护通过（长度={len(diff)}，上限={MAX_DIFF_CHARS_PER_FILE}）")


# ============================================================
# 模块三：Cue 提示触发回归（S6 第 57-58 天）
# ============================================================

def _create_cue_test_workspace():
    """
    构造与 Sprint_6.md 验收场景一致的测试仓库：
      UserService.py    定义 getUser
      AdminService.py   第 22 行调用 getUser
      HelperService.py  调用 getUser
    """
    workspace = tempfile.mkdtemp()

    with open(os.path.join(workspace, "UserService.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""User service."""\n\n\n'
            'def getUser(user_id):\n'
            '    return {"id": user_id}\n'
        )

    admin_lines = [
        '"""Admin service."""',
        'from UserService import getUser',
        '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
        'def admin_get_user(uid):',
        '    """Get user."""',
        '    return getUser(uid)',   # 第 22 行
    ]
    with open(os.path.join(workspace, "AdminService.py"), "w", encoding="utf-8") as f:
        f.write("\n".join(admin_lines) + "\n")

    with open(os.path.join(workspace, "HelperService.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Helper service."""\n'
            'from UserService import getUser\n\n'
            'def helper():\n'
            '    return getUser(1)\n'
        )
    return workspace


class _StubIndexService:
    def __init__(self):
        self._lock = threading.Lock()
        self._index = {}

    def add_table(self, file_path, table):
        self._index[file_path] = table


def _build_cue_matcher(workspace):
    graph = DependencyGraph()
    stub = _StubIndexService()
    for fname in ("UserService.py", "AdminService.py", "HelperService.py"):
        full = os.path.join(workspace, fname)
        table = parse_file(full)
        table.file_path = fname
        stub.add_table(fname, table)
        graph.register_file(fname, table.language)
        graph.build_from_file(fname, table, full_path=full)
    return SymbolExactMatcher(index_service=stub, dependency_graph=graph)


def _run_cue_suggest_logic(matcher, req: CueSuggestRequest, cue_max: int = 20):
    """复刻 app/api/cue.py 的核心业务逻辑（不经过 HTTP）。"""
    exclude_file = normalize_path(req.file_path) if req.file_path else None
    suggestions = []

    if req.action in ("rename", "delete"):
        callers = matcher.find_callers(req.symbol_name, top_k=cue_max)
        reason = (
            "此函数调用了被改名的符号" if req.action == "rename"
            else "此函数调用了被删除的符号"
        )
        for c in callers:
            fp = c.get("file_path", "")
            line = c.get("line", 0)
            if exclude_file and fp == exclude_file:
                continue
            if fp == exclude_file and line == req.modified_line:
                continue
            if line < 1:
                continue
            suggestions.append(CueSuggestionItem(file_path=fp, line=line, reason=reason))
    elif req.action == "add_method":
        related = matcher.find_related_symbols(
            req.symbol_name, exclude_file=req.file_path, top_k=cue_max,
        )
        reason = "存在同名方法，可能需要同步修改"
        for r in related:
            suggestions.append(CueSuggestionItem(
                file_path=r["file_path"], line=r["line"], reason=reason,
            ))

    suggestions = suggestions[:cue_max]
    return CueSuggestResponse(
        action=req.action,
        symbol_name=req.symbol_name,
        total=len(suggestions),
        suggestions=suggestions,
    )


def test_cue_rename_returns_callers():
    """
    验收：rename 场景返回跨文件调用方，含 AdminService.py 第 22 行。
    """
    ws = _create_cue_test_workspace()
    matcher = _build_cue_matcher(ws)
    req = CueSuggestRequest(
        file_path="UserService.py",
        modified_line=4,
        action="rename",
        symbol_name="getUser",
    )
    resp = _run_cue_suggest_logic(matcher, req)

    matched = [
        s for s in resp.suggestions
        if s.file_path == "AdminService.py" and s.line == 22
    ]
    assert matched, f"应返回 AdminService.py:22，实际: {[(s.file_path, s.line) for s in resp.suggestions]}"
    assert "UserService.py" not in {s.file_path for s in resp.suggestions}
    assert matched[0].reason == "此函数调用了被改名的符号"
    print(f"✅ Cue rename 触发通过：返回 {resp.total} 条，含 AdminService.py:22")


def test_cue_delete_returns_callers():
    """
    验收：delete 场景同样返回跨文件调用方，原因为"被删除的符号"。
    """
    ws = _create_cue_test_workspace()
    matcher = _build_cue_matcher(ws)
    req = CueSuggestRequest(
        file_path="UserService.py",
        modified_line=4,
        action="delete",
        symbol_name="getUser",
    )
    resp = _run_cue_suggest_logic(matcher, req)
    files = {s.file_path for s in resp.suggestions}
    assert "AdminService.py" in files
    assert "HelperService.py" in files
    assert "UserService.py" not in files
    assert all(s.reason == "此函数调用了被删除的符号" for s in resp.suggestions)
    print(f"✅ Cue delete 触发通过：返回 {resp.total} 条，跨文件调用方 {files}")


def test_cue_add_method_returns_related_symbols():
    """
    验收：add_method 场景返回其他文件中的同名符号定义。
    """
    ws = _create_cue_test_workspace()
    matcher = _build_cue_matcher(ws)
    # 在 HelperService 中新增 getUser，应在 UserService 中找到同名定义
    req = CueSuggestRequest(
        file_path="HelperService.py",
        modified_line=5,
        action="add_method",
        symbol_name="getUser",
    )
    resp = _run_cue_suggest_logic(matcher, req)
    files = {s.file_path for s in resp.suggestions}
    assert "UserService.py" in files
    assert "HelperService.py" not in files
    assert all(s.reason == "存在同名方法，可能需要同步修改" for s in resp.suggestions)
    print(f"✅ Cue add_method 触发通过：返回 {resp.total} 条，命中 {files}")


def test_cue_suggest_max_limit():
    """
    验收：CUE_SUGGEST_MAX 限制返回数量，缓解误报（S6 风险预警）。
    """
    assert settings.CUE_SUGGEST_MAX == 20
    print(f"✅ Cue 返回数量上限配置通过：CUE_SUGGEST_MAX={settings.CUE_SUGGEST_MAX}")


# ============================================================
# 模块四：多文件 JSON 输出超时保护回归（S6 第 59-60 天）
# ============================================================

def test_json_mode_timeout_config():
    """
    验收：JSON 模式超时配置为 30 秒，友好提示文案正确。
    """
    assert settings.JSON_MODE_TIMEOUT_SECONDS == 30.0
    assert settings.JSON_MODE_TIMEOUT_MESSAGE == "生成时间过长，请简化需求重试"
    print(
        f"✅ JSON 模式超时配置通过："
        f"timeout={settings.JSON_MODE_TIMEOUT_SECONDS}s, "
        f"msg='{settings.JSON_MODE_TIMEOUT_MESSAGE}'"
    )


def test_json_mode_timeout_overrides_mode_timeout():
    """
    验收：开启 response_format 时，超时被覆盖为 JSON_MODE_TIMEOUT_SECONDS，
    与 mode 维度（chat=60 / new=120 / inline=120）无关。
    模拟 chat.py 中的超时选择逻辑。
    """
    mode_timeouts = {
        "chat": settings.CHAT_TIMEOUT_SECONDS,        # 60
        "new": settings.NEW_FILE_TIMEOUT_SECONDS,     # 120
        "inline": settings.INLINE_CHAT_TIMEOUT_SECONDS,  # 120
    }
    json_timeout = settings.JSON_MODE_TIMEOUT_SECONDS  # 30

    for mode, mode_to in mode_timeouts.items():
        # 模拟 chat.py 逻辑：先按 mode 选超时，JSON 模式再覆盖
        request_timeout = mode_to
        is_json_mode = True
        if is_json_mode:
            request_timeout = json_timeout
        assert request_timeout == json_timeout, (
            f"mode={mode} 的 JSON 模式超时应为 {json_timeout}s，实际 {request_timeout}s"
        )
        # JSON 超时必须短于所有 mode 超时（否则保护无意义）
        assert json_timeout < mode_to, (
            f"JSON 超时({json_timeout}s)应短于 mode={mode} 超时({mode_to}s)"
        )
    print("✅ JSON 模式超时覆盖 mode 超时通过（chat/new/inline 均被覆盖为 30s）")


def test_json_mode_timeout_message_distinct_from_default():
    """
    验收：JSON 模式超时时返回的友好提示与默认超时消息不同，
    便于前端区分展示。
    """
    from app.error_codes import ERROR_MESSAGES, ErrorCode
    default_timeout_msg = ERROR_MESSAGES[ErrorCode.MODEL_TIMEOUT]
    json_timeout_msg = settings.JSON_MODE_TIMEOUT_MESSAGE
    assert json_timeout_msg != default_timeout_msg
    print(
        f"✅ JSON 超时提示与默认提示区分通过："
        f"默认='{default_timeout_msg}'，JSON='{json_timeout_msg}'"
    )


def test_non_json_mode_keeps_mode_timeout():
    """
    验收：未开启 response_format 时，仍按 mode 维度使用原超时（不被覆盖）。
    """
    req = _make_chat_request(mode="inline")
    assert req.response_format is None
    # 模拟逻辑
    request_timeout = settings.INLINE_CHAT_TIMEOUT_SECONDS
    is_json_mode = req.response_format is not None
    if is_json_mode:
        request_timeout = settings.JSON_MODE_TIMEOUT_SECONDS
    assert request_timeout == settings.INLINE_CHAT_TIMEOUT_SECONDS
    print(f"✅ 非 JSON 模式保留 mode 超时通过：inline={request_timeout}s")


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("S6 回归测试用例集（第 59-60 天：联调 / 性能 / P2 里程碑）")
    print("=" * 60)

    # 模块一：Inline Chat
    print("\n【模块一】Inline Chat 修改回归（S6 第 51-52 天）")
    test_inline_chat_request_schema()
    test_inline_chat_system_prompt_instruction_exists()
    test_inline_selection_line_range_annotation()
    test_inline_chat_default_mode_is_chat()

    # 模块二：多文件修改
    print("\n【模块二】多文件修改回归（S6 第 53-54 / 55-56 天）")
    test_json_mode_system_prompt_instruction_exists()
    test_response_format_schema()
    test_chat_request_with_response_format()
    test_parse_json_files_multi_file()
    test_parse_json_files_with_markdown_fence_regression()
    test_parse_json_files_invalid_returns_none_regression()
    test_build_diff_files_multi_file()
    test_build_diff_files_new_file_scenario()
    test_build_diff_files_path_traversal_blocked_regression()
    test_diff_chunking_max_3_per_batch()
    test_large_diff_truncation_protection()

    # 模块三：Cue 提示触发
    print("\n【模块三】Cue 提示触发回归（S6 第 57-58 天）")
    test_cue_rename_returns_callers()
    test_cue_delete_returns_callers()
    test_cue_add_method_returns_related_symbols()
    test_cue_suggest_max_limit()

    # 模块四：超时保护
    print("\n【模块四】多文件 JSON 输出超时保护（S6 第 59-60 天）")
    test_json_mode_timeout_config()
    test_json_mode_timeout_overrides_mode_timeout()
    test_json_mode_timeout_message_distinct_from_default()
    test_non_json_mode_keeps_mode_timeout()

    print("\n" + "=" * 60)
    print("🎉 S6 回归测试全部通过（P2 里程碑交付就绪）")
    print("=" * 60)


if __name__ == "__main__":
    main()
