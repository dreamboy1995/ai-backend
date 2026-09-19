"""
上下文 DTO 验证脚本（S2 第 13-14 天）

验证：
1. ContextItem 能正确解析 file / selection / implicit 三种类型
2. file_path 必填且非空
3. content_snippet 超过 50000 字符时被防御性拒绝
4. ChatRequest.contexts 为 None / 空数组 / 多条上下文 时均能正常解析
5. type 取值非法时被拒绝
"""
from pydantic import ValidationError

from app.models.schemas import ChatRequest, ContextItem


def _make_context(**overrides) -> dict:
    base = {
        "type": "file",
        "file_path": "src/main.py",
        "content_snippet": "def hello():\n    print('hello')\n",
        "language": "python",
    }
    base.update(overrides)
    return base


def test_context_item_valid_types():
    """三种合法 type 都能正常解析"""
    for t in ("file", "selection", "implicit"):
        item = ContextItem(**_make_context(type=t))
        assert item.type == t
        assert item.file_path == "src/main.py"
        assert item.language == "python"
    print("✅ ContextItem 三种 type 解析通过")


def test_context_item_invalid_type_rejected():
    """非法 type 必须被拒绝"""
    try:
        ContextItem(**_make_context(type="directory"))
        raise AssertionError("非法 type 'directory' 应被拒绝")
    except ValidationError:
        pass
    print("✅ 非法 type 拒绝通过")


def test_context_item_file_path_required():
    """file_path 必填且非空"""
    # 缺字段
    try:
        ContextItem(type="file", content_snippet="x")
        raise AssertionError("缺 file_path 应被拒绝")
    except ValidationError:
        pass
    # 空字符串
    try:
        ContextItem(**_make_context(file_path=""))
        raise AssertionError("空 file_path 应被拒绝")
    except ValidationError:
        pass
    print("✅ file_path 必填校验通过")


def test_context_item_snippet_too_long_rejected():
    """content_snippet 超过 50000 字符必须被拒绝"""
    long_snippet = "x" * 50_001
    try:
        ContextItem(**_make_context(content_snippet=long_snippet))
        raise AssertionError("超长 content_snippet 应被拒绝")
    except ValidationError as e:
        # 确认是因为长度校验触发
        assert "content_snippet" in str(e)
    # 边界：恰好 50000 字符应通过
    ContextItem(**_make_context(content_snippet="x" * 50_000))
    print("✅ content_snippet 长度防御通过（上限 50000 字符）")


def test_context_item_language_optional():
    """language 可选"""
    item = ContextItem(**{k: v for k, v in _make_context().items() if k != "language"})
    assert item.language is None
    print("✅ language 可选通过")


def test_chat_request_contexts_none_or_empty():
    """contexts 为 None 或空数组时正常解析"""
    req_none = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert req_none.contexts is None

    req_empty = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        contexts=[],
    )
    assert req_empty.contexts == []
    print("✅ contexts=None / [] 解析通过")


def test_chat_request_with_contexts():
    """ChatRequest 能携带多条上下文正常解析"""
    req = ChatRequest(
        messages=[{"role": "user", "content": "@main.py 这个函数是干嘛的？"}],
        session_id="sess_test",
        contexts=[
            _make_context(type="file", file_path="src/main.py"),
            _make_context(
                type="selection",
                file_path="src/utils.js",
                content_snippet="function f(){}",
                language="javascript",
            ),
            _make_context(type="implicit", file_path="src/utils.js"),
        ],
    )
    assert req.contexts is not None
    assert len(req.contexts) == 3
    assert req.contexts[0].type == "file"
    assert req.contexts[1].type == "selection"
    assert req.contexts[2].type == "implicit"
    # model_dump 能正确序列化（用于后续传给 adapter / ContextBuilder）
    dumped = req.contexts[0].model_dump()
    assert dumped["file_path"] == "src/main.py"
    assert dumped["language"] == "python"
    print("✅ ChatRequest 携带 contexts 解析通过")


if __name__ == "__main__":
    test_context_item_valid_types()
    test_context_item_invalid_type_rejected()
    test_context_item_file_path_required()
    test_context_item_snippet_too_long_rejected()
    test_context_item_language_optional()
    test_chat_request_contexts_none_or_empty()
    test_chat_request_with_contexts()
    print("\n" + "=" * 50)
    print("全部上下文 DTO 测试通过！")
    print("=" * 50)
