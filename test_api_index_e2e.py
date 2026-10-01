"""API 端到端集成测试（S4 第 37-38 天）"""
import os
import tempfile
import time

from fastapi.testclient import TestClient

from app.auth import create_access_token
from app.main import app

# 生成测试 Token
token = create_access_token({"sub": "test-user", "user_id": "test-user"})
headers = {"Authorization": f"Bearer {token}"}

client = TestClient(app)

# 创建临时工作区
workspace = tempfile.mkdtemp()
with open(os.path.join(workspace, "main.py"), "w", encoding="utf-8") as f:
    f.write(
        'def hello():\n'
        '    return "world"\n'
        '\n'
        'class App:\n'
        '    def run(self):\n'
        '        pass\n'
    )
with open(os.path.join(workspace, "utils.py"), "w", encoding="utf-8") as f:
    f.write(
        'def add(a, b):\n'
        '    return a + b\n'
    )

print("=== 1. POST /v1/index/start ===")
resp = client.post(
    "/v1/index/start",
    json={
        "workspace_root": workspace,
        "force_rebuild": True,
        "priority_files": ["main.py"],
    },
    headers=headers,
)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")
assert resp.status_code == 200

print()
print("=== 2. GET /v1/index/status (轮询) ===")
status = None
for i in range(60):
    resp = client.get("/v1/index/status", headers=headers)
    if resp.status_code == 429:
        # 触发限频，等待后重试
        print(f"  [{i}] 429 限频，等待 2s...")
        time.sleep(2)
        continue
    status = resp.json()
    print(
        f"  [{i}] status={status['status']} "
        f"processed={status['processed']}/{status['total']} "
        f"({status['percentage'] * 100:.1f}%)"
    )
    if status["status"] in ("done", "error"):
        break
    time.sleep(1.0)  # 1s 间隔，避免触发分钟限频（20次/分钟）
assert status["status"] == "done", f"索引未完成: {status}"
print(f"  符号总数: {status['total_symbols']}")

print()
print("=== 3. POST /v1/index/update (modified) ===")
with open(os.path.join(workspace, "main.py"), "w", encoding="utf-8") as f:
    f.write(
        'def hello():\n'
        '    return "world"\n'
        '\n'
        'def goodbye():\n'
        '    return "bye"\n'
    )
resp = client.post(
    "/v1/index/update",
    json={"file_path": "main.py", "action": "modified"},
    headers=headers,
)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")
assert resp.status_code == 200
assert resp.json()["success"] is True

print()
print("=== 4. POST /v1/index/update (deleted) ===")
resp = client.post(
    "/v1/index/update",
    json={"file_path": "utils.py", "action": "deleted"},
    headers=headers,
)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")
assert resp.status_code == 200

print()
print("=== 5. 验证标记文件 ===")
marker = os.path.join(workspace, ".ai_index", "index_done")
print(f"标记文件存在: {os.path.exists(marker)}")
assert os.path.exists(marker)

print()
print("✅ API 端到端集成测试通过！")
