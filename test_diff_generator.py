"""
S6 第 53-54 天：Diff 生成器与 JSON 解析单元测试
S6 第 55-56 天：新增大文件动态上下文行数测试

验证：
1. generate_unified_diff 能正确生成 Unified Diff（含新增/修改/删除行）。
2. 新旧内容相同时返回空串。
3. build_diff_files 能读取原文件、组装含 old/new/diff 的结构。
4. 路径逃逸（../../../etc/passwd）被拦截，返回空 old_content。
5. _parse_json_files 能从带 ```json 标记的文本中提取 files 数组。
6. _parse_json_files 对非法/缺字段输入返回 None。
7. determine_context_lines：大文件(>500行)返回 3，小文件返回 SMALL_FILE_CONTEXT_LINES。
8. build_diff_files 默认按文件大小动态选择上下文行数。
9. 大文件 Diff 只包含变更行附近 3 行上下文，远离变更的行不出现。
"""
import os
import tempfile

from app.services.diff_generator import (
    generate_unified_diff,
    build_diff_files,
    read_original_content,
    determine_context_lines,
    DEFAULT_DIFF_CONTEXT_LINES,
    LARGE_FILE_CONTEXT_LINES,
    LARGE_FILE_LINE_THRESHOLD,
    SMALL_FILE_CONTEXT_LINES,
)
from app.api.chat import _parse_json_files


def test_unified_diff_basic():
    old = "import os\nprint('hello')\n"
    new = "import logging\nlogging.info('hello')\n"
    diff = generate_unified_diff(old, new, "src/main.py")
    assert diff
    assert "--- a/src/main.py" in diff
    assert "+++ b/src/main.py" in diff
    assert "-import os" in diff
    assert "+import logging" in diff
    print("✅ 基础 Unified Diff 生成通过")


def test_unified_diff_identical_returns_empty():
    content = "line1\nline2\n"
    assert generate_unified_diff(content, content, "a.py") == ""
    print("✅ 相同内容返回空 Diff 通过")


def test_unified_diff_new_file():
    diff = generate_unified_diff("", "print('hi')\n", "new.py")
    assert diff
    assert "+print('hi')" in diff
    print("✅ 新增文件 Diff 通过")


def test_build_diff_files_with_real_file():
    with tempfile.TemporaryDirectory() as tmp:
        # 写一个原始文件
        orig = "def foo():\n    return 1\n"
        with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as f:
            f.write(orig)

        files_json = [
            {"path": "main.py", "content": "def foo():\n    return 42\n"}
        ]
        result = build_diff_files(files_json, tmp)
        assert len(result) == 1
        item = result[0]
        assert item["path"] == "main.py"
        assert item["old_content"] == orig
        assert item["new_content"] == "def foo():\n    return 42\n"
        assert "return 1" in item["diff"]
        assert "return 42" in item["diff"]
    print("✅ build_diff_files 读取原文件并生成 Diff 通过")


def test_build_diff_files_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        files_json = [
            {"path": "../../../etc/passwd", "content": "hacked"}
        ]
        result = build_diff_files(files_json, tmp)
        assert len(result) == 1
        # 路径逃逸被拦截，old_content 应为空（不读取真实文件）
        assert result[0]["old_content"] == ""
    print("✅ 路径逃逸拦截通过")


def test_build_diff_files_skip_invalid_items():
    with tempfile.TemporaryDirectory() as tmp:
        files_json = [
            {"path": "a.py", "content": "x"},
            {"content": "no path"},           # 缺 path
            {"path": "b.py"},                  # 缺 content
            "not a dict",                      # 非 dict
        ]
        result = build_diff_files(files_json, tmp)
        assert len(result) == 1
        assert result[0]["path"] == "a.py"
    print("✅ 无效条目跳过通过")


def test_read_original_content_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        assert read_original_content("nonexistent.py", tmp) == ""
    print("✅ 不存在文件返回空串通过")


def test_parse_json_files_clean():
    text = '{"files": [{"path": "a.py", "content": "x"}], "explanation": "ok"}'
    files = _parse_json_files(text)
    assert files is not None
    assert len(files) == 1
    assert files[0]["path"] == "a.py"
    print("✅ 干净 JSON 解析通过")


def test_parse_json_files_with_markdown_fence():
    # 模拟模型在 JSON 前后加 ```json 标记
    text = '```json\n{"files": [{"path": "a.py", "content": "x"}]}\n```'
    files = _parse_json_files(text)
    assert files is not None
    assert len(files) == 1
    print("✅ 带 markdown 围栏的 JSON 解析通过")


def test_parse_json_files_with_prefix_text():
    text = '好的，这是修改结果：\n{"files": [{"path": "a.py", "content": "x"}]}'
    files = _parse_json_files(text)
    assert files is not None
    print("✅ 带前置解释文本的 JSON 解析通过")


def test_parse_json_files_invalid_returns_none():
    assert _parse_json_files("") is None
    assert _parse_json_files("hello world") is None
    assert _parse_json_files('{"foo": "bar"}') is None          # 缺 files
    assert _parse_json_files('{"files": []}') is None           # files 为空
    assert _parse_json_files('not json {{{') is None
    print("✅ 非法 JSON 返回 None 通过")


def test_diff_context_lines_default():
    # 大文件只保留变更行附近 3 行上下文
    old = "\n".join([f"line{i}" for i in range(20)]) + "\n"
    new = old.replace("line10", "CHANGED")
    diff = generate_unified_diff(old, new, "big.py", context_lines=DEFAULT_DIFF_CONTEXT_LINES)
    # 上下文行数=3，变更行(第11行)附近应保留 line7-line13 范围
    assert "line7" in diff or "line8" in diff
    # 远离变更的行不应出现
    assert "line0" not in diff
    print("✅ 大文件上下文行数限制通过")


# ============================================================
# S6 第 55-56 天：动态上下文行数测试
# ============================================================

def test_determine_context_lines_small_file():
    # 小文件（<= 500 行）应返回 SMALL_FILE_CONTEXT_LINES
    small = "\n".join([f"line{i}" for i in range(100)]) + "\n"
    assert determine_context_lines(small) == SMALL_FILE_CONTEXT_LINES
    print("✅ 小文件返回大上下文行数通过")


def test_determine_context_lines_large_file():
    # 超大文件（> 500 行）应返回 LARGE_FILE_CONTEXT_LINES（3）
    large = "\n".join([f"line{i}" for i in range(LARGE_FILE_LINE_THRESHOLD + 1)]) + "\n"
    assert determine_context_lines(large) == LARGE_FILE_CONTEXT_LINES
    print("✅ 超大文件返回 3 行上下文通过")


def test_determine_context_lines_boundary():
    # 恰好 500 行视为小文件，501 行视为大文件
    exactly_threshold = "\n".join([f"line{i}" for i in range(LARGE_FILE_LINE_THRESHOLD)]) + "\n"
    assert determine_context_lines(exactly_threshold) == SMALL_FILE_CONTEXT_LINES

    over_threshold = "\n".join([f"line{i}" for i in range(LARGE_FILE_LINE_THRESHOLD + 1)]) + "\n"
    assert determine_context_lines(over_threshold) == LARGE_FILE_CONTEXT_LINES
    print("✅ 边界值（500/501 行）判断通过")


def test_determine_context_lines_empty():
    # 空内容视为 0 行，返回小文件上下文
    assert determine_context_lines("") == SMALL_FILE_CONTEXT_LINES
    print("✅ 空内容处理通过")


def test_large_file_diff_uses_3_context_lines():
    """
    大文件（>500 行）的 Diff 只包含变更行附近 3 行上下文，
    远离变更点的行不应出现在 Diff 中。
    """
    # 构建 600 行的文件，在第 300 行附近做修改
    lines = [f"line{i}" for i in range(600)]
    old = "\n".join(lines) + "\n"
    lines[300] = "CHANGED_LINE"
    new = "\n".join(lines) + "\n"

    diff = generate_unified_diff(old, new, "large.py", context_lines=LARGE_FILE_CONTEXT_LINES)

    # 变更行附近的行应出现
    assert "CHANGED_LINE" in diff
    assert "line297" in diff or "line298" in diff  # 变更前 3 行内
    # 远离变更的行不应出现
    assert "line0" not in diff
    assert "line599" not in diff
    print("✅ 大文件 Diff 仅含变更行附近 3 行上下文通过")


def test_build_diff_files_dynamic_context():
    """
    build_diff_files 默认（context_lines=None）按文件大小动态选择上下文行数：
    - 小文件保留完整上下文
    - 大文件只保留 3 行上下文
    """
    with tempfile.TemporaryDirectory() as tmp:
        # 小文件：20 行
        small_orig = "\n".join([f"s{i}" for i in range(20)]) + "\n"
        with open(os.path.join(tmp, "small.py"), "w", encoding="utf-8") as f:
            f.write(small_orig)

        # 大文件：600 行
        large_lines = [f"l{i}" for i in range(600)]
        large_orig = "\n".join(large_lines) + "\n"
        with open(os.path.join(tmp, "large.py"), "w", encoding="utf-8") as f:
            f.write(large_orig)

        files_json = [
            {"path": "small.py", "content": small_orig.replace("s10", "SMALL_CHANGED")},
            {"path": "large.py", "content": large_orig.replace("l300", "LARGE_CHANGED")},
        ]
        result = build_diff_files(files_json, tmp)
        assert len(result) == 2

        small_diff = result[0]["diff"]
        large_diff = result[1]["diff"]

        # 小文件：完整上下文，首尾行都应出现
        assert "s0" in small_diff
        assert "s19" in small_diff

        # 大文件：只有变更行附近 3 行，首尾行不应出现
        assert "LARGE_CHANGED" in large_diff
        assert "l0" not in large_diff
        assert "l599" not in large_diff

    print("✅ build_diff_files 动态上下文行数通过")


def test_build_diff_files_explicit_context_override():
    """
    显式传入 context_lines 时，所有文件统一使用该值，不做动态判断。
    """
    with tempfile.TemporaryDirectory() as tmp:
        small_orig = "\n".join([f"s{i}" for i in range(20)]) + "\n"
        with open(os.path.join(tmp, "small.py"), "w", encoding="utf-8") as f:
            f.write(small_orig)

        files_json = [
            {"path": "small.py", "content": small_orig.replace("s10", "CHANGED")}
        ]
        # 显式指定 context_lines=1，即使是小文件也只保留 1 行上下文
        result = build_diff_files(files_json, tmp, context_lines=1)
        diff = result[0]["diff"]
        assert "CHANGED" in diff
        # 只保留 1 行上下文，s0（距变更 10 行）不应出现
        assert "s0" not in diff

    print("✅ 显式 context_lines 覆盖动态判断通过")


if __name__ == "__main__":
    test_unified_diff_basic()
    test_unified_diff_identical_returns_empty()
    test_unified_diff_new_file()
    test_build_diff_files_with_real_file()
    test_build_diff_files_path_traversal_blocked()
    test_build_diff_files_skip_invalid_items()
    test_read_original_content_missing_file()
    test_parse_json_files_clean()
    test_parse_json_files_with_markdown_fence()
    test_parse_json_files_with_prefix_text()
    test_parse_json_files_invalid_returns_none()
    test_diff_context_lines_default()
    # S6 第 55-56 天新增测试
    test_determine_context_lines_small_file()
    test_determine_context_lines_large_file()
    test_determine_context_lines_boundary()
    test_determine_context_lines_empty()
    test_large_file_diff_uses_3_context_lines()
    test_build_diff_files_dynamic_context()
    test_build_diff_files_explicit_context_override()
    print("\n" + "=" * 50)
    print("全部测试通过 ✅")
