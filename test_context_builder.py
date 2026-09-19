"""
ContextBuilder 验证脚本（S2 第 15-16 天：后端上下文拼装 & 系统提示词工程）

验证：
1. XML 格式化：contexts 被正确格式化为 <context_files><file ...>...</file></context_files>
2. 系统提示词拼装：包含基础提示词 + 上下文 XML
3. 优先级裁剪：file/selection 保留，implicit 优先截断
4. Token 预算：裁剪后总 Token 不超过上下文预算
5. 与 SessionService 协同：reserved_tokens 正确从历史阈值中扣除
6. 已有 system 内容合并：请求中带 system 消息时能正确合并
"""

from app.models.schemas import ContextItem
from app.services.context_builder import ContextBuilder, BASE_SYSTEM_PROMPT
from app.services.session import SessionService, count_tokens


def _make_ctx(**overrides) -> ContextItem:
    base = {
        "type": "file",
        "file_path": "src/main.py",
        "content_snippet": "def hello():\n    print('hello')\n",
        "language": "python",
    }
    base.update(overrides)
    return ContextItem(**base)


def test_xml_format():
    """验证 contexts 被格式化为正确的 XML 结构"""
    builder = ContextBuilder(total_token_budget=8000, token_margin=0.2, context_ratio=0.5)
    contexts = [
        _make_ctx(type="file", file_path="a.py", content_snippet="def foo(): pass"),
        _make_ctx(type="selection", file_path="b.js", content_snippet="const x = 1;", language="javascript"),
    ]
    prompt = builder.build_system_prompt(contexts)

    # 必须包含 context_files 根标签
    assert "<context_files>" in prompt
    assert "</context_files>" in prompt
    # 必须包含每个文件的标签
    assert '<file path="a.py" lang="python">def foo(): pass</file>' in prompt
    assert '<file path="b.js" lang="javascript">const x = 1;</file>' in prompt
    print("✅ XML 格式化验证通过")


def test_system_prompt_contains_base():
    """验证系统提示词包含基础提示词内容"""
    builder = ContextBuilder()
    prompt = builder.build_system_prompt([])
    assert BASE_SYSTEM_PROMPT in prompt
    assert "<context_files>" not in prompt  # 无上下文时不应有 context_files 标签
    print("✅ 基础系统提示词验证通过")


def test_existing_system_content_merged():
    """验证已有 system 内容与上下文 XML 正确合并"""
    builder = ContextBuilder()
    existing = "你是一个 Rust 专家"
    contexts = [_make_ctx(file_path="lib.rs", content_snippet="fn main() {}")]
    prompt = builder.build_system_prompt(contexts, existing_system_content=existing)

    assert existing in prompt
    assert "<context_files>" in prompt
    # 已有内容应在上下文之前
    assert prompt.index(existing) < prompt.index("<context_files>")
    print("✅ 已有 system 内容合并验证通过")


def test_priority_truncation_implicit_dropped():
    """验证 implicit 类型在超预算时被优先丢弃，file 保留"""
    # 预算精确设置：file("a")=7 tokens, context_budget=7, 留给 implicit 的剩余=0 → 丢弃
    builder = ContextBuilder(total_token_budget=16, token_margin=0.1, context_ratio=0.5)
    contexts = [
        _make_ctx(type="implicit", file_path="implicit.py", content_snippet="x" * 500),
        _make_ctx(type="file", file_path="user_file.py", content_snippet="a"),
    ]
    prompt = builder.build_system_prompt(contexts)

    # file 应保留，implicit 应被丢弃（剩余预算为 0，无法截断）
    assert "user_file.py" in prompt
    assert "implicit.py" not in prompt
    print("✅ 优先级裁剪（implicit 丢弃）验证通过")


def test_priority_file_kept_over_implicit():
    """验证 file 优先级高于 implicit，即使 file 在列表后面"""
    # 预算精确设置：file("a")=7 tokens, context_budget=7, implicit 被丢弃
    builder = ContextBuilder(total_token_budget=16, token_margin=0.1, context_ratio=0.5)
    contexts = [
        _make_ctx(type="implicit", file_path="implicit.py", content_snippet="x" * 500),
        _make_ctx(type="file", file_path="user_file.py", content_snippet="a"),
    ]
    prompt = builder.build_system_prompt(contexts)

    # 虽然 implicit 在列表前面，但 file 优先级更高，最终只有 file 被保留
    assert "user_file.py" in prompt
    assert "implicit.py" not in prompt
    print("✅ file 优先级高于 implicit 验证通过")


def test_implicit_truncated_when_partial_fit():
    """验证 implicit 在部分能放下时被截断而非完全丢弃"""
    builder = ContextBuilder(total_token_budget=400, token_margin=0.1, context_ratio=0.5)
    contexts = [
        _make_ctx(type="file", file_path="a.py", content_snippet="def foo(): pass"),
        _make_ctx(type="implicit", file_path="big.py", content_snippet="x" * 2000),
    ]
    prompt = builder.build_system_prompt(contexts)

    # file 保留
    assert "a.py" in prompt
    # implicit 被截断保留（含截断标记），而非完全丢弃
    assert "big.py" in prompt
    assert "内容已截断" in prompt
    print("✅ implicit 部分截断验证通过")


def test_xml_escaping():
    """验证 XML 特殊字符被转义，不破坏标签结构"""
    builder = ContextBuilder()
    contexts = [
        _make_ctx(
            file_path="a<b>.py",
            content_snippet='if x < 0 and y > 0: print("&")',
        )
    ]
    prompt = builder.build_system_prompt(contexts)
    # 转义后不应出现未转义的 < 在内容中（标签除外）
    # 检查路径被转义
    assert 'path="a&lt;b&gt;.py"' in prompt
    print("✅ XML 转义验证通过")


def test_token_budget_respected():
    """验证裁剪后上下文 Token 不超过预算"""
    builder = ContextBuilder(total_token_budget=1000, token_margin=0.2, context_ratio=0.5)
    # 构造大量上下文
    contexts = [
        _make_ctx(type="file", file_path=f"f{i}.py", content_snippet=f"# file {i}\n" + "x" * 500)
        for i in range(20)
    ]
    prompt = builder.build_system_prompt(contexts)
    prompt_tokens = count_tokens([{"role": "system", "content": prompt}])

    # 系统提示词总 Token 应不超过阈值（含基础提示词）
    threshold = int(1000 * (1 - 0.2))
    assert prompt_tokens <= threshold, f"系统提示词 Token {prompt_tokens} 超过阈值 {threshold}"
    print(f"✅ Token 预算验证通过（系统提示词 Token≈{prompt_tokens} <= 阈值 {threshold}）")


def test_session_reserved_tokens():
    """验证 SessionService 正确扣除 reserved_tokens"""
    svc = SessionService(token_budget=1000, token_margin=0.1, max_rounds=50)
    sid = "test-reserved"
    svc.create(sid)
    # 塞入大量消息
    for i in range(10):
        svc.append(sid, {"role": "user", "content": "x" * 300})
        svc.append(sid, {"role": "assistant", "content": "y" * 300})

    # 不预留时的裁剪结果
    history_no_reserve = svc.get_history(sid, reserved_tokens=0)
    tokens_no_reserve = count_tokens(history_no_reserve)

    # 预留 500 Token 时，历史应更小
    history_reserved = svc.get_history(sid, reserved_tokens=500)
    tokens_reserved = count_tokens(history_reserved)

    threshold = int(1000 * (1 - 0.1))
    # 预留后历史 Token + 预留 Token 不应超过阈值
    assert tokens_reserved + 500 <= threshold + 50, (
        f"预留后总 Token {tokens_reserved + 500} 超过阈值 {threshold}"
    )
    # 预留后的历史 Token 应小于不预留的
    assert tokens_reserved <= tokens_no_reserve
    print(
        f"✅ Session reserved_tokens 验证通过 "
        f"(无预留 Token≈{tokens_no_reserve}, 预留500后 Token≈{tokens_reserved})"
    )


def test_language_optional_in_xml():
    """验证 language 为 None 时 XML 中不出现 lang 属性"""
    builder = ContextBuilder()
    contexts = [_make_ctx(language=None, file_path="a.py")]
    prompt = builder.build_system_prompt(contexts)
    assert '<file path="a.py">' in prompt
    assert 'lang=' not in prompt
    print("✅ language 可选验证通过")


if __name__ == "__main__":
    test_xml_format()
    test_system_prompt_contains_base()
    test_existing_system_content_merged()
    test_priority_truncation_implicit_dropped()
    test_priority_file_kept_over_implicit()
    test_implicit_truncated_when_partial_fit()
    test_xml_escaping()
    test_token_budget_respected()
    test_session_reserved_tokens()
    test_language_optional_in_xml()
    print("\n" + "=" * 50)
    print("全部 ContextBuilder 测试通过！")
    print("=" * 50)
