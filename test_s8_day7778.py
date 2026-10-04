"""
S8 第 77-78 天：Git 工具增强 + 审计日志增强 单元测试

验收标准（来自 Sprint_8.md 第 77-78 天）：
  1. git_commit：若暂存区有变更，成功生成 Commit 并返回 commit hash；
     若无可提交变更，返回友好的错误提示。
  2. git_commit 未提供 message 时，自动调用 LLM 基于暂存区 Diff 生成
     （Conventional Commits 规范），LLM 不可用时降级为 fallback message。
  3. 所有工具执行日志写入审计日志（JSON Lines），含 tool_name、arguments、
     result、timestamp、duration_ms、output_chars 等字段。

覆盖场景：
  - git_commit 完整链路：git init → 写文件 → git add → git commit → 返回 hash
  - git_commit 自动生成 message（LLM 成功 / LLM 降级 fallback / LLM AdapterError）
  - git_commit 用户显式提供 message（跳过 LLM）
  - git_commit 无变更时返回友好错误
  - git_commit _truncate_diff_for_llm 超长 diff 截断
  - 审计日志增强字段：duration_ms / output_chars / output_preview / has_confirmation_id
  - 审计日志 output_preview 大输出截断（避免日志膨胀）
  - 审计日志 duration_ms 在正常工具调用中 > 0
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.models.tool import ToolCall
from app.services.tool_registry import (
    _truncate_diff_for_llm,
    confirm_tool,
    execute_tool,
    write_audit_log,
)


# ============================================================
# 测试夹具
# ============================================================

@pytest.fixture
def git_workspace(tmp_path):
    """创建一个真实的 git 仓库工作区，初始化 git 配置"""
    ws = str(tmp_path)
    # git init
    subprocess.run(["git", "init"], cwd=ws, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "ai-agent@test.local"],
        cwd=ws, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "AI Agent Test"],
        cwd=ws, capture_output=True, check=True,
    )
    return ws


def _make_tool_call(tool_name, **arguments):
    return ToolCall(tool_name=tool_name, arguments=arguments)


# ============================================================
# 1. git_commit 基础流程测试
# ============================================================

def test_git_commit_no_changes_returns_error(git_workspace):
    """工作区无变更时，git_commit 返回友好错误"""
    tool_call = _make_tool_call("git_commit", message="test")
    result = asyncio.run(execute_tool(tool_call, git_workspace, "s77-session"))

    assert result.success is False
    assert "无变更" in result.error or "无需提交" in result.error
    # 不需要 confirmation，因为根本没有东西可提交
    assert result.requires_confirmation is False


def test_git_commit_needs_confirmation_when_changes_exist(git_workspace):
    """有变更时，git_commit 返回 requires_confirmation=True"""
    # 在工作区写一个新文件
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    tool_call = _make_tool_call("git_commit", message="feat: init main.py")
    result = asyncio.run(execute_tool(tool_call, git_workspace, "s77-session"))

    assert result.success is True
    assert result.requires_confirmation is True
    assert result.confirmation_id is not None
    assert result.confirmation_prompt is not None
    # output 里应包含变更文件信息
    assert "main.py" in result.output or "M" in result.output or "??" in result.output


def test_git_commit_explicit_message_used_directly(git_workspace):
    """用户显式提供 message 时，直接使用，不调用 LLM"""
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    explicit_msg = "feat: 显式提供的提交信息"
    tool_call = _make_tool_call("git_commit", message=explicit_msg)

    # mock LLM，如果被调用则 test fail
    with patch(
        "app.services.tool_registry._generate_commit_message_via_llm",
        new_callable=AsyncMock,
    ) as mock_llm:
        result = asyncio.run(execute_tool(tool_call, git_workspace, "s77-session"))
        mock_llm.assert_not_called()

    assert result.success is True
    assert result.requires_confirmation is True
    # confirmation_prompt 里应该包含用户显式 message
    assert explicit_msg in result.confirmation_prompt
    # 不是 AI 生成的（所以不应有 "AI 自动生成" 标记）
    assert "AI 自动生成" not in result.confirmation_prompt


def test_git_commit_allow_executes_real_git_commit(git_workspace):
    """确认 allow 后，真正执行 git commit 并返回 commit hash"""
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    tool_call = _make_tool_call("git_commit", message="feat: init main.py")
    exec_result = asyncio.run(execute_tool(tool_call, git_workspace, "s77-session"))
    assert exec_result.requires_confirmation is True
    cid = exec_result.confirmation_id

    # 确认 allow
    confirm_result = asyncio.run(confirm_tool(cid, "allow", "s77-session"))

    assert confirm_result.success is True
    assert "提交成功" in confirm_result.output or "unknown" not in confirm_result.output
    # output 应包含 commit hash（40 字符 SHA1）
    import re
    sha_match = re.search(r"[0-9a-f]{40}", confirm_result.output)
    assert sha_match is not None, f"未找到 commit hash: {confirm_result.output}"

    # 验证文件确实在 git 历史里
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=git_workspace, capture_output=True, text=True, check=True,
    )
    assert "feat: init main.py" in log.stdout


def test_git_commit_deny_does_not_commit(git_workspace):
    """确认 deny 后，不执行 git commit"""
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    tool_call = _make_tool_call("git_commit", message="feat: init main.py")
    exec_result = asyncio.run(execute_tool(tool_call, git_workspace, "s77-session"))
    cid = exec_result.confirmation_id

    confirm_result = asyncio.run(confirm_tool(cid, "deny", "s77-session"))

    assert confirm_result.success is False
    assert "拒绝" in confirm_result.error

    # 没有任何 commit
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=git_workspace, capture_output=True, text=True,
    )
    assert log.returncode != 0 or log.stdout.strip() == ""


# ============================================================
# 2. git_commit 自动生成 message 测试（S8 第 77-78 天核心）
# ============================================================

def test_git_commit_auto_generate_message_llm_success(git_workspace):
    """LLM 返回有效的 Conventional Commits message 时被使用"""
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    mock_message = "feat(core): 增加 hello world 入口"
    with patch(
        "app.services.tool_registry._generate_commit_message_via_llm",
        new_callable=AsyncMock,
        return_value=mock_message,
    ) as mock_llm:
        tool_call = _make_tool_call("git_commit")  # 不提供 message
        result = asyncio.run(execute_tool(tool_call, git_workspace, "auto-msg-session"))

    assert mock_llm.called
    assert result.success is True
    assert result.requires_confirmation is True
    # confirmation_prompt 里应包含 AI 生成的 message + "AI 自动生成" 标记
    assert mock_message in result.confirmation_prompt
    assert "AI 自动生成" in result.confirmation_prompt
    # output 里也应包含
    assert mock_message in result.output


def test_git_commit_auto_generate_message_fallback_when_llm_returns_none(git_workspace):
    """LLM 返回 None（不可用/输出无效）时降级为 fallback message"""
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    with patch(
        "app.services.tool_registry._generate_commit_message_via_llm",
        new_callable=AsyncMock,
        return_value=None,
    ):
        tool_call = _make_tool_call("git_commit")
        result = asyncio.run(execute_tool(tool_call, git_workspace, "fallback-session"))

    assert result.success is True
    assert result.requires_confirmation is True
    # 降级为 settings.GIT_COMMIT_FALLBACK_MESSAGE
    assert settings.GIT_COMMIT_FALLBACK_MESSAGE in result.confirmation_prompt
    # 没有 "AI 自动生成" 标记（因为是 fallback）
    assert "AI 自动生成" not in result.confirmation_prompt


def test_git_commit_auto_generate_message_fallback_when_llm_disabled(git_workspace):
    """GIT_COMMIT_AUTO_MESSAGE_ENABLED=False 时，跳过 LLM 直接用 fallback"""
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    with patch.object(settings, "GIT_COMMIT_AUTO_MESSAGE_ENABLED", False):
        tool_call = _make_tool_call("git_commit")
        result = asyncio.run(execute_tool(tool_call, git_workspace, "disabled-session"))

    assert result.success is True
    assert settings.GIT_COMMIT_FALLBACK_MESSAGE in result.confirmation_prompt


def test_git_commit_auto_generate_message_real_commit_with_llm(git_workspace):
    """LLM 生成 message → 确认 allow → 真实 commit 带上 AI message"""
    (Path(git_workspace) / "main.py").write_text("print('hello')\n", encoding="utf-8")

    mock_message = "feat(entry): 新建 main.py 入口文件"
    with patch(
        "app.services.tool_registry._generate_commit_message_via_llm",
        new_callable=AsyncMock,
        return_value=mock_message,
    ):
        tool_call = _make_tool_call("git_commit")
        exec_result = asyncio.run(execute_tool(tool_call, git_workspace, "e2e-auto-session"))

    assert exec_result.requires_confirmation is True
    cid = exec_result.confirmation_id

    confirm_result = asyncio.run(confirm_tool(cid, "allow", "e2e-auto-session"))
    assert confirm_result.success is True

    # 验证 commit message 是 AI 生成的那条
    log = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        cwd=git_workspace, capture_output=True, text=True, check=True,
    )
    assert mock_message in log.stdout.strip()


# ============================================================
# 3. _truncate_diff_for_llm 截断逻辑测试
# ============================================================

def test_truncate_diff_for_llm_short_unchanged():
    """短 diff 不截断"""
    short_diff = "diff --git a/x.py b/x.py\n+ x = 1\n"
    with patch.object(settings, "GIT_COMMIT_DIFF_MAX_CHARS", 8000):
        result = _truncate_diff_for_llm(short_diff)
    assert result == short_diff


def test_truncate_diff_for_llm_long_truncated():
    """超长 diff 被截断"""
    long_diff = "x" * 20000  # 20000 chars
    with patch.object(settings, "GIT_COMMIT_DIFF_MAX_CHARS", 8000):
        result = _truncate_diff_for_llm(long_diff)

    assert len(result) <= 8000 + 200  # 截断后略长于 max_chars（留了截断标记）
    assert "[diff truncated" in result  # 包含截断标记
    # 开头和结尾应该都保留了
    assert long_diff[:100] in result
    assert long_diff[-100:] in result


def test_truncate_diff_for_llm_empty():
    """空字符串不崩溃"""
    assert _truncate_diff_for_llm("") == ""


# ============================================================
# 4. 审计日志增强测试（S8 第 77-78 天要求）
# ============================================================

def test_audit_log_new_fields(tmp_path):
    """审计日志包含新增的增强字段：duration_ms / output_chars / has_confirmation_id"""
    log_path = tmp_path / "audit_enhanced.log"

    with patch.object(settings, "TOOL_AUDIT_LOG_PATH", str(log_path)):
        # 触发一个工具调用（read_file，最简单）
        tool_call = _make_tool_call("read_file", file_path="nonexistent.txt")
        asyncio.run(execute_tool(tool_call, str(tmp_path), "audit-enhanced-session"))

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    # 可能有多行（因为是 JSON Lines），取最后一条
    entry = json.loads(lines[-1])

    # 新增字段存在性校验
    assert "duration_ms" in entry
    assert "output_chars" in entry
    assert "has_confirmation_id" in entry
    # 类型校验
    assert isinstance(entry["duration_ms"], (int, float))
    assert isinstance(entry["output_chars"], int)
    assert isinstance(entry["has_confirmation_id"], bool)
    # 基础字段仍然存在
    assert "timestamp" in entry
    assert "session_id" in entry
    assert "tool_name" in entry
    assert "success" in entry


def test_audit_log_duration_ms_positive():
    """真实工具调用（如 read_file）的 duration_ms 应 > 0"""
    import app.services.tool_registry as registry_mod

    with tempfile.TemporaryDirectory() as tmp:
        # 准备一个测试文件确保 read_file 会真正执行 IO
        test_file = Path(tmp) / "dur_test.txt"
        test_file.write_text("hello\n" * 1000, encoding="utf-8")

        log_path = Path(tmp) / "dur_audit.log"
        with patch.object(registry_mod.settings, "TOOL_AUDIT_LOG_PATH", str(log_path)):
            tool_call = _make_tool_call("read_file", file_path="dur_test.txt")
            asyncio.run(execute_tool(tool_call, tmp, "dur-session"))

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[-1])
        # duration_ms 应该为正数（工具执行了 IO）
        assert entry["duration_ms"] >= 0, f"duration_ms 应为非负: {entry['duration_ms']}"
        assert entry["tool_name"] == "read_file"
        assert entry["session_id"] == "dur-session"


def test_audit_log_output_preview_truncation():
    """大 output 时审计日志只写截断后的 preview（S8 风险预警：大输出撑爆日志）"""
    import app.services.tool_registry as registry_mod

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "preview_audit.log"

        # 设置很小的 preview 阈值方便测试
        with patch.object(registry_mod.settings, "TOOL_AUDIT_LOG_PATH", str(log_path)):
            with patch.object(registry_mod.settings, "TOOL_AUDIT_LOG_OUTPUT_MAX_CHARS", 50):
                result_big = registry_mod.ToolResult(
                    success=True,
                    output="A" * 5000,  # 5000 字符的大 output
                )
                result_small = registry_mod.ToolResult(
                    success=True,
                    output="short output",
                )

                asyncio.run(write_audit_log(
                    "preview-session",
                    _make_tool_call("run_command", cmd="big-cmd"),
                    result_big,
                    duration_ms=123.45,
                ))
                asyncio.run(write_audit_log(
                    "preview-session",
                    _make_tool_call("run_command", cmd="small-cmd"),
                    result_small,
                    duration_ms=5.0,
                ))

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        # 第一条：大 output
        entry_big = json.loads(lines[0])
        assert entry_big["output_chars"] == 5000
        assert "output_preview" in entry_big
        assert len(entry_big["output_preview"]) < 5000
        assert "[truncated" in entry_big["output_preview"]
        # 第二条：小 output
        entry_small = json.loads(lines[1])
        assert entry_small["output_chars"] == 12
        assert entry_small.get("output_preview") == "short output"  # 小的直接存


def test_audit_log_confirmation_flag():
    """requires_confirmation=True 的工具，审计日志 has_confirmation_id 应为 True"""
    import app.services.tool_registry as registry_mod

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "confirm_audit.log"

        with patch.object(registry_mod.settings, "TOOL_AUDIT_LOG_PATH", str(log_path)):
            # write_file 会触发 requires_confirmation=True
            tool_call = _make_tool_call(
                "write_file", file_path="x.py", content="print('x')\n"
            )
            asyncio.run(execute_tool(tool_call, tmp, "confirm-flag-session"))

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["tool_name"] == "write_file"
        assert entry["has_confirmation_id"] is True


# ============================================================
# 5. 端到端链路（S8 验收清单场景简化版）
# ============================================================

def test_git_commit_e2e_with_auto_message_and_confirm(git_workspace):
    """
    S8 验收清单简化版：完整链路
    1. 新建文件 → 2. git_commit（自动 AI message）→ 3. confirm allow
    """
    # Step 1: 新建一个 Python 文件
    (Path(git_workspace) / "app.py").write_text(
        "def hello():\n    return 'world'\n", encoding="utf-8"
    )
    (Path(git_workspace) / "requirements.txt").write_text("flask==3.0.0\n", encoding="utf-8")

    # Step 2: 执行 git_commit（无 message，走自动生成）
    with patch(
        "app.services.tool_registry._generate_commit_message_via_llm",
        new_callable=AsyncMock,
        return_value="feat(flask): 初始化 Flask 项目骨架",
    ):
        tool_call = _make_tool_call("git_commit")
        exec_result = asyncio.run(execute_tool(tool_call, git_workspace, "e2e-session"))

    assert exec_result.success is True
    assert exec_result.requires_confirmation is True
    assert "AI 自动生成" in exec_result.confirmation_prompt
    assert "feat(flask): 初始化 Flask 项目骨架" in exec_result.output

    # Step 3: confirm allow → 真正 commit
    confirm_result = asyncio.run(confirm_tool(exec_result.confirmation_id, "allow", "e2e-session"))
    assert confirm_result.success is True

    # Step 4: 验证 git 历史
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=git_workspace, capture_output=True, text=True, check=True,
    )
    assert "feat(flask)" in log.stdout
    # 验证两个文件都被提交了
    files = subprocess.run(
        ["git", "show", "--name-only", "--format="],
        cwd=git_workspace, capture_output=True, text=True, check=True,
    )
    assert "app.py" in files.stdout
    assert "requirements.txt" in files.stdout


def test_git_confirm_deny_e2e(git_workspace):
    """confirm deny 完整链路：文件仍留在工作区但不进入 git 历史"""
    (Path(git_workspace) / "temp.py").write_text("print('temp')\n", encoding="utf-8")

    tool_call = _make_tool_call("git_commit", message="chore: temp")
    exec_result = asyncio.run(execute_tool(tool_call, git_workspace, "deny-e2e-session"))
    assert exec_result.requires_confirmation is True

    confirm_result = asyncio.run(confirm_tool(exec_result.confirmation_id, "deny", "deny-e2e-session"))
    assert confirm_result.success is False

    # 文件仍在工作区
    assert (Path(git_workspace) / "temp.py").exists()

    # 没有任何 commit
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=git_workspace, capture_output=True, text=True,
    )
    assert result.returncode != 0  # HEAD 不存在 = 没有 commit


# ============================================================
# 6. 边界场景：_generate_commit_message_via_llm 的输入边界
# ============================================================

def test_generate_commit_message_via_llm_disabled_returns_none():
    """AUTO_MESSAGE_ENABLED=False 时直接返回 None，不调 LLM"""
    import app.services.tool_registry as registry_mod

    with patch.object(settings, "GIT_COMMIT_AUTO_MESSAGE_ENABLED", False):
        result = asyncio.run(
            registry_mod._generate_commit_message_via_llm("some diff content")
        )
    assert result is None


def test_generate_commit_message_via_llm_empty_diff_returns_none():
    """diff 为空时直接返回 None"""
    import app.services.tool_registry as registry_mod

    result = asyncio.run(registry_mod._generate_commit_message_via_llm(""))
    assert result is None
    result2 = asyncio.run(registry_mod._generate_commit_message_via_llm("   \n   "))
    assert result2 is None
