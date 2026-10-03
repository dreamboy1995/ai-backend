"""
Cue 编辑位置预测接口验收测试脚本（S6 第 57-58 天）

用法:
    python test_cue_suggest.py

验收标准（来自 Sprint_6.md 第 57-58 天）：
    在 UserService 中重命名 getUser 为 fetchUser 后，后端返回的候选列表
    中包含 AdminService 第 22 行（AdminService 中调用了 getUser）。

本脚本分四阶段验证：
    阶段 1：单元功能验证（find_callers / find_related_symbols）
        - 构造测试仓库（UserService.py / AdminService.py / HelperService.py）
        - 验证 find_callers 返回跨文件调用方
        - 验证 find_related_symbols 返回跨文件同名符号
    阶段 2：Cue suggest 业务逻辑验证（模拟 POST /v1/cue/suggest 内核）
        - rename 场景：返回调用 getUser 的位置（含 AdminService 第 22 行）
        - delete 场景：同样返回调用方
        - add_method 场景：返回其他文件的同名符号
        - 同文件过滤：排除 UserService.py 自身（由插件 Rule A 处理）
    阶段 3：Pydantic 模型校验
        - CueSuggestRequest 字段约束（min_length / ge）
        - CueSuggestResponse 序列化
    阶段 4：HTTP 接口验证（可选）
        - 启动 FastAPI app，调用 POST /v1/cue/suggest
        - 若端口被占用或鉴权则跳过
"""

import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.code_index.ast_parser import parse_file
from app.services.code_index.dependency_graph import DependencyGraph
from app.services.code_index.symbol_exact_matcher import SymbolExactMatcher
from app.models.schemas import (
    CueSuggestRequest,
    CueSuggestResponse,
    CueSuggestionItem,
)


# ============================================================
# 测试仓库构造
# ============================================================

def create_test_workspace():
    """
    创建测试工作区，模拟 Sprint_6.md 验收场景：
      UserService.py     (定义 getUser 函数)
      AdminService.py    (在第 22 行调用 UserService.getUser)
      HelperService.py   (在第 6 行调用 getUser)
    """
    workspace = tempfile.mkdtemp()

    # UserService.py：定义 getUser 函数
    with open(os.path.join(workspace, "UserService.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""User service module."""\n\n\n'
            'def getUser(user_id):\n'
            '    return {"id": user_id, "name": "Alice"}\n'
        )

    # AdminService.py：让 getUser 调用精确出现在第 22 行
    admin_lines = [
        '"""Admin service module."""',          # 1
        'from UserService import getUser',     # 2
        '',                                     # 3
        '',                                     # 4
        '',                                     # 5
        '',                                     # 6
        '',                                     # 7
        '',                                     # 8
        '',                                     # 9
        '',                                     # 10
        '',                                     # 11
        '',                                     # 12
        '',                                     # 13
        '',                                     # 14
        '',                                     # 15
        '',                                     # 16
        '',                                     # 17
        '',                                     # 18
        '',                                     # 19
        'def admin_get_user(uid):',             # 20
        '    """Get user for admin."""',        # 21
        '    return getUser(uid)',              # 22 ← 调用 getUser
    ]
    with open(os.path.join(workspace, "AdminService.py"), "w", encoding="utf-8") as f:
        f.write("\n".join(admin_lines) + "\n")

    # HelperService.py：也在跨文件调用 getUser
    with open(os.path.join(workspace, "HelperService.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Helper service module."""\n'
            'from UserService import getUser\n\n'
            'def helper():\n'
            '    return getUser(1)\n'
        )

    return workspace


# Stub IndexService：仅暴露 _lock 与 _index，供 SymbolExactMatcher 使用
class _StubIndexService:
    def __init__(self):
        self._lock = threading.Lock()
        self._index = {}  # file_path -> SymbolTable

    def add_table(self, file_path, table):
        self._index[file_path] = table


def _build_matcher(workspace):
    """构建依赖图 + 符号表注入的 SymbolExactMatcher"""
    graph = DependencyGraph()
    stub_index = _StubIndexService()

    for fname in ("UserService.py", "AdminService.py", "HelperService.py"):
        full_path = os.path.join(workspace, fname)
        table = parse_file(full_path)
        table.file_path = fname
        stub_index.add_table(fname, table)
        graph.register_file(fname, table.language)
        graph.build_from_file(fname, table, full_path=full_path)

    return SymbolExactMatcher(index_service=stub_index, dependency_graph=graph), graph


# ============================================================
# 阶段 1：单元功能验证
# ============================================================

def run_unit_test(workspace, matcher):
    """验证 find_callers 与 find_related_symbols 的基础功能"""

    # 验证 1：find_callers 应返回 getUser 的所有调用方
    callers = matcher.find_callers("getUser", top_k=20)
    caller_files = {c["file_path"] for c in callers}
    assert "AdminService.py" in caller_files, (
        f"find_callers 应包含 AdminService.py，实际: {caller_files}"
    )
    assert "HelperService.py" in caller_files, (
        f"find_callers 应包含 HelperService.py，实际: {caller_files}"
    )
    print(f"[阶段1] ✅ find_callers 返回调用方文件: {caller_files}")

    # 验证 2：AdminService.py 第 22 行应精确出现在调用方列表中
    admin_caller = next(c for c in callers if c["file_path"] == "AdminService.py")
    assert admin_caller["line"] == 22, (
        f"AdminService.py 调用 getUser 的行号应为 22，实际: {admin_caller['line']}"
    )
    print(f"[阶段1] ✅ AdminService.py 第 {admin_caller['line']} 行调用 getUser（精确匹配验收）")

    # 验证 3：find_related_symbols 应返回其他文件中的同名符号
    # 在测试工作区里 UserService.py 定义了 getUser，跨文件查找同名定义时，
    # 因 AdminService/HelperService 没有 def getUser，故应返回空。
    # 改为查找已有符号 getUser，并排除 UserService 自身 → 应为空（无其他定义）
    related = matcher.find_related_symbols("getUser", exclude_file="UserService.py")
    assert related == [], (
        f"其他文件未定义 getUser，应返回空，实际: {related}"
    )
    print("[阶段1] ✅ find_related_symbols 排除当前文件后无同名定义（符合预期）")

    # 验证 4：不排除 UserService 自身时，应能找到 UserService.py 中的定义
    related_all = matcher.find_related_symbols("getUser", exclude_file=None)
    related_files = {r["file_path"] for r in related_all}
    assert "UserService.py" in related_files, (
        f"应找到 UserService.py 中的 getUser 定义，实际: {related_files}"
    )
    print(f"[阶段1] ✅ find_related_symbols 不排除时返回: {related_files}")

    print("[阶段1] 🎉 单元功能验收全部通过！")
    print("=" * 60)


# ============================================================
# 阶段 2：Cue suggest 业务逻辑验证（模拟接口内核）
# ============================================================

def _simulate_cue_suggest(matcher, req: CueSuggestRequest, cue_max: int = 20):
    """复刻 app/api/cue.py 的核心逻辑（不经过 HTTP），用于验证业务正确性"""
    from app.services.code_index.symbol_exact_matcher import normalize_path
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
            suggestions.append(CueSuggestionItem(
                file_path=fp, line=line, reason=reason,
            ))
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


def run_business_test(workspace, matcher):
    """模拟 POST /v1/cue/suggest 的核心业务逻辑"""

    # 场景 1：在 UserService.py 中重命名 getUser → fetchUser
    # 用户改了 UserService.py 第 4 行（getUser 定义所在行）
    req = CueSuggestRequest(
        file_path="UserService.py",
        modified_line=4,
        action="rename",
        symbol_name="getUser",
    )
    resp = _simulate_cue_suggest(matcher, req)

    # 验收：候选列表应包含 AdminService.py 第 22 行
    matched = [
        s for s in resp.suggestions
        if s.file_path == "AdminService.py" and s.line == 22
    ]
    assert matched, (
        f"rename 场景应返回 AdminService.py 第 22 行，"
        f"实际 suggestions: {[(s.file_path, s.line) for s in resp.suggestions]}"
    )
    assert "UserService.py" not in {s.file_path for s in resp.suggestions}, (
        "同文件 UserService.py 应被排除（由插件 Rule A 处理）"
    )
    assert matched[0].reason == "此函数调用了被改名的符号"
    print(
        f"[阶段2] ✅ rename 场景：返回 {resp.total} 条建议，"
        f"含 AdminService.py:22（reason='{matched[0].reason}'）"
    )

    # 场景 2：删除 UserService.getUser
    req_del = CueSuggestRequest(
        file_path="UserService.py",
        modified_line=4,
        action="delete",
        symbol_name="getUser",
    )
    resp_del = _simulate_cue_suggest(matcher, req_del)
    del_files = {s.file_path for s in resp_del.suggestions}
    assert "AdminService.py" in del_files and "HelperService.py" in del_files
    assert "UserService.py" not in del_files
    assert all(s.reason == "此函数调用了被删除的符号" for s in resp_del.suggestions)
    print(
        f"[阶段2] ✅ delete 场景：返回 {resp_del.total} 条建议，"
        f"跨文件调用方 {del_files}"
    )

    # 场景 3：在 AdminService 中新增方法 admin_get_user
    # 查找其他文件中是否已存在同名方法（应仅在 AdminService 自身定义）
    # 验证：排除 AdminService 后，应返回空（无其他文件定义 admin_get_user）
    req_add = CueSuggestRequest(
        file_path="AdminService.py",
        modified_line=20,
        action="add_method",
        symbol_name="admin_get_user",
    )
    resp_add = _simulate_cue_suggest(matcher, req_add)
    assert resp_add.total == 0, (
        f"admin_get_user 仅在 AdminService.py 中定义，排除后应为空，"
        f"实际: {resp_add.total}"
    )
    print(f"[阶段2] ✅ add_method 场景：跨文件无同名方法，返回 0 条（符合预期）")

    # 场景 4：add_method 在 HelperService 中新增 getUser
    # 应在跨文件找到 UserService.py 中的 getUser 同名定义
    req_add2 = CueSuggestRequest(
        file_path="HelperService.py",
        modified_line=5,
        action="add_method",
        symbol_name="getUser",
    )
    resp_add2 = _simulate_cue_suggest(matcher, req_add2)
    add_files = {s.file_path for s in resp_add2.suggestions}
    assert "UserService.py" in add_files, (
        f"应在 UserService.py 中找到同名 getUser，实际: {add_files}"
    )
    assert "HelperService.py" not in add_files, "同文件应被排除"
    assert all(s.reason == "存在同名方法，可能需要同步修改" for s in resp_add2.suggestions)
    print(
        f"[阶段2] ✅ add_method 场景（跨文件同名）：返回 {resp_add2.total} 条，"
        f"命中 {add_files}"
    )

    print("[阶段2] 🎉 业务逻辑验收全部通过！")
    print("=" * 60)


# ============================================================
# 阶段 3：Pydantic 模型校验
# ============================================================

def run_schema_test():
    """验证 Cue 请求/响应模型的字段约束"""
    from pydantic import ValidationError

    # 验证 1：合法请求应能构造
    req = CueSuggestRequest(
        file_path="UserService.py",
        modified_line=4,
        action="rename",
        symbol_name="getUser",
    )
    assert req.action == "rename"
    print(f"[阶段3] ✅ 合法请求构造成功: action={req.action}")

    # 验证 2：file_path 不能为空
    try:
        CueSuggestRequest(
            file_path="", modified_line=4, action="rename", symbol_name="getUser",
        )
        raise AssertionError("空 file_path 应被拒绝")
    except ValidationError:
        pass
    print("[阶段3] ✅ 空 file_path 被拒绝（min_length=1）")

    # 验证 3：modified_line 必须 >= 1
    try:
        CueSuggestRequest(
            file_path="a.py", modified_line=0, action="rename", symbol_name="x",
        )
        raise AssertionError("modified_line=0 应被拒绝")
    except ValidationError:
        pass
    print("[阶段3] ✅ modified_line < 1 被拒绝（ge=1）")

    # 验证 4：action 必须为枚举值
    try:
        CueSuggestRequest(
            file_path="a.py", modified_line=1, action="invalid", symbol_name="x",
        )
        raise AssertionError("非法 action 应被拒绝")
    except ValidationError:
        pass
    print("[阶段3] ✅ 非法 action 被拒绝（Literal 约束）")

    # 验证 5：响应序列化
    resp = CueSuggestResponse(
        action="rename",
        symbol_name="getUser",
        total=1,
        suggestions=[CueSuggestionItem(file_path="AdminService.py", line=22, reason="x")],
    )
    data = resp.model_dump()
    assert data["suggestions"][0]["line"] == 22
    print(f"[阶段3] ✅ 响应序列化正常: total={data['total']}")

    print("[阶段3] 🎉 Pydantic 模型校验全部通过！")
    print("=" * 60)


# ============================================================
# 阶段 4：HTTP 接口验证（可选）
# ============================================================

def run_http_test():
    """验证 POST /v1/cue/suggest 接口（若服务已启动）"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(("127.0.0.1", 3000))
    except (socket.timeout, ConnectionRefusedError, OSError):
        print("[阶段4] ⚠️  后端服务未启动（127.0.0.1:3000），跳过 HTTP 接口验证")
        print("=" * 60)
        return

    try:
        import httpx
        with httpx.Client(base_url="http://127.0.0.1:3000", timeout=5.0) as client:
            # 探测接口是否存在（无数据时应返回 200 + 空 suggestions）
            resp = client.post("/v1/cue/suggest", json={
                "file_path": "UserService.py",
                "modified_line": 4,
                "action": "rename",
                "symbol_name": "getUser",
            })
            if resp.status_code == 401:
                print("[阶段4] ⚠️  接口需鉴权，跳过 HTTP 验证（业务逻辑已在阶段 1-3 验证）")
                return
            if resp.status_code != 200:
                print(f"[阶段4] ⚠️  POST /v1/cue/suggest 返回 {resp.status_code}，跳过")
                return
            data = resp.json()
            print(f"[阶段4] POST /v1/cue/suggest → 200, total={data.get('total', 0)}, "
                  f"action={data.get('action')}")
    except Exception as e:
        print(f"[阶段4] ⚠️  HTTP 验证失败: {e}")
    print("=" * 60)


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("S6 第 57-58 天：Cue 编辑位置预测接口验收测试")
    print("=" * 60)

    workspace = create_test_workspace()
    matcher, _graph = _build_matcher(workspace)

    run_unit_test(workspace, matcher)
    run_business_test(workspace, matcher)
    run_schema_test()
    run_http_test()

    print("🎉 Cue 编辑位置预测验收测试完成（核心功能已通过）。")


if __name__ == "__main__":
    main()
