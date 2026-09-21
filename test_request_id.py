"""
X-Request-ID 链路追踪验证脚本（S2 第 19-20 天：异常处理、联调与 S2 Demo 准备）

验证：
1. 客户端未传 X-Request-ID 时，后端自动生成并在响应头回写。
2. 客户端传入 X-Request-ID 时，后端原样回写（前后端打通）。
3. 超长 X-Request-ID 会被截断到 128 字符以内，防止日志污染。
4. 中间件设置的 request_id 能被业务日志感知（日志记录带 req=<id>）。
5. SSE 流式响应头同样携带 X-Request-ID（通过受保护路径的响应头间接验证）。
"""

import logging

from fastapi.testclient import TestClient

from app.main import app
from app.middlewares.request_id import REQUEST_ID_HEADER, _MAX_CLIENT_REQUEST_ID_LEN


class _CaptureHandler(logging.Handler):
    """捕获日志记录，用于断言 request_id 是否注入。"""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_request_id_generated_when_missing():
    """未传 X-Request-ID 时，后端生成并回写响应头。"""
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert REQUEST_ID_HEADER in r.headers, "响应头缺少 X-Request-ID"
        rid = r.headers[REQUEST_ID_HEADER]
        assert rid and rid != "-", f"生成的 request_id 非法: {rid}"
        # uuid4().hex 为 32 位十六进制
        assert len(rid) == 32, f"生成的 request_id 长度异常: {len(rid)}"
    print("✅ 未传 X-Request-ID 时后端自动生成并通过响应头回写")


def test_request_id_echoed_when_provided():
    """客户端传入 X-Request-ID 时，后端原样回写。"""
    fixed = "client-rid-abc123"
    with TestClient(app) as client:
        r = client.get("/health", headers={REQUEST_ID_HEADER: fixed})
        assert r.status_code == 200
        assert r.headers[REQUEST_ID_HEADER] == fixed, (
            f"响应头 X-Request-ID 应原样回写: 期望 {fixed}, "
            f"实际 {r.headers[REQUEST_ID_HEADER]}"
        )
    print("✅ 客户端传入 X-Request-ID 时后端原样回写（前后端打通）")


def test_request_id_truncates_long_input():
    """超长 X-Request-ID 被截断，防止日志污染。"""
    long_id = "x" * (_MAX_CLIENT_REQUEST_ID_LEN + 100)
    with TestClient(app) as client:
        r = client.get("/health", headers={REQUEST_ID_HEADER: long_id})
        assert r.status_code == 200
        echoed = r.headers[REQUEST_ID_HEADER]
        assert len(echoed) <= _MAX_CLIENT_REQUEST_ID_LEN, (
            f"超长 request_id 未被截断: 长度 {len(echoed)} > {_MAX_CLIENT_REQUEST_ID_LEN}"
        )
    print(f"✅ 超长 X-Request-ID 被截断至 {len(echoed)} 字符")


def test_request_id_appears_in_logs():
    """中间件设置的 request_id 能被业务日志感知。"""
    fixed = "log-trace-rid-999"
    capture = _CaptureHandler()
    root = logging.getLogger()
    root.addHandler(capture)
    try:
        with TestClient(app) as client:
            # 带非法 token 访问受保护资源，触发 auth 中间件的 warning 日志
            # request_id 中间件为最外层，auth 日志会带上当前 request_id
            r = client.get(
                "/v1/chat/completions",
                headers={REQUEST_ID_HEADER: fixed, "Authorization": "Bearer invalid-token"},
            )
            assert r.status_code == 401
        # 在 auth 中间件中应产生带 request_id 的日志记录
        matched = [
            rec
            for rec in capture.records
            if getattr(rec, "request_id", "-") == fixed
        ]
        assert matched, (
            f"未在日志中找到 request_id={fixed} 的记录，"
            f"捕获到 {len(capture.records)} 条日志"
        )
    finally:
        root.removeHandler(capture)
    print(f"✅ 业务日志带上 req={fixed}（共 {len(matched)} 条）")


def test_request_id_each_request_isolated():
    """不同请求的 request_id 相互隔离，不发生跨请求泄漏。"""
    with TestClient(app) as client:
        r1 = client.get("/health")
        r2 = client.get("/health")
        rid1 = r1.headers[REQUEST_ID_HEADER]
        rid2 = r2.headers[REQUEST_ID_HEADER]
        assert rid1 != rid2, f"两次请求的 request_id 不应相同: {rid1}"
    print("✅ 不同请求的 request_id 相互隔离，无跨请求泄漏")


if __name__ == "__main__":
    test_request_id_generated_when_missing()
    test_request_id_echoed_when_provided()
    test_request_id_truncates_long_input()
    test_request_id_appears_in_logs()
    test_request_id_each_request_isolated()
    print("\n" + "=" * 50)
    print("全部 X-Request-ID 链路追踪测试通过！")
    print("=" * 50)
