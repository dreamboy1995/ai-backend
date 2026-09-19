"""
会话管理裁剪逻辑验证脚本
验证：连续对话 20 轮后，裁剪后的消息条数稳定在 12 条左右
"""
from app.services.session import SessionService, count_tokens


def test_sliding_window_trimming():
    # 使用默认配置：token_budget=8000, max_rounds=5
    svc = SessionService(ttl_seconds=3600, token_budget=8000, token_margin=0.2, max_rounds=5)
    sid = "test-session-trim"
    svc.create(sid)

    # 模拟 20 轮对话（每轮 user + assistant）
    for i in range(20):
        svc.append(sid, {"role": "user", "content": f"第{i+1}轮用户提问：请帮我解决问题{i+1}"})
        svc.append(sid, {"role": "assistant", "content": f"第{i+1}轮助手回答：这是解决方案{i+1}的详细说明"})

    history = svc.get_history(sid)
    print(f"20轮对话后，裁剪后的消息条数: {len(history)}")
    print(f"预期约 10 条 (system=0 + 5轮*2=10): 实际 {len(history)}")
    assert len(history) <= 12, f"裁剪后消息数 {len(history)} 超过预期上限 12"
    print("✅ 轮数裁剪验证通过")

    print("\n裁剪后的消息角色序列:")
    for m in history:
        print(f"  - {m['role']}: {m['content'][:30]}...")


def test_system_message_preserved():
    """验证 System 消息始终保留"""
    svc = SessionService(max_rounds=2)
    sid = "test-sys"
    svc.create(sid)
    svc.append(sid, {"role": "system", "content": "你是一个编程助手"})
    for i in range(10):
        svc.append(sid, {"role": "user", "content": f"问题{i}"})
        svc.append(sid, {"role": "assistant", "content": f"回答{i}"})

    history = svc.get_history(sid)
    roles = [m["role"] for m in history]
    assert "system" in roles, "System 消息丢失！"
    assert roles[0] == "system", "System 消息应在最前面"
    print(f"\n✅ System 消息保留验证通过（总条数 {len(history)}，system 在首位）")


def test_token_budget_trimming():
    """验证 Token 预算裁剪：超长消息会被丢弃"""
    svc = SessionService(token_budget=100, token_margin=0.1, max_rounds=50)
    sid = "test-token"
    svc.create(sid)
    # 塞入大量长消息
    for i in range(20):
        svc.append(sid, {"role": "user", "content": "x" * 500})
        svc.append(sid, {"role": "assistant", "content": "y" * 500})

    history = svc.get_history(sid)
    tokens = count_tokens(history)
    threshold = int(100 * (1 - 0.1))
    print(f"\nToken 裁剪后条数: {len(history)}, Token数: {tokens}, 阈值: {threshold}")
    assert tokens <= threshold, f"Token {tokens} 超过阈值 {threshold}"
    print("✅ Token 预算裁剪验证通过")


def test_ttl_expiry():
    """验证会话过期机制"""
    svc = SessionService(ttl_seconds=1)
    sid = "test-ttl"
    svc.create(sid)
    svc.append(sid, {"role": "user", "content": "hello"})
    assert svc.exists(sid) is True
    import time
    time.sleep(1.2)
    assert svc.exists(sid) is False, "会话应已过期"
    print("\n✅ 会话过期(TTL)验证通过")


if __name__ == "__main__":
    test_sliding_window_trimming()
    test_system_message_preserved()
    test_token_budget_trimming()
    test_ttl_expiry()
    print("\n" + "=" * 50)
    print("全部会话管理测试通过！")
    print("=" * 50)
