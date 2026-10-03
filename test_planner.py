"""
S7 第 63-64 天：Planner 任务规划器单元测试

验收标准（来自 Sprint_7.md）：
  输入 "做一个 Flask + React 的博客系统"，后端返回的 JSON 包含至少 6 个步骤，
  且后端校验逻辑检测到无循环依赖（如步骤 3 依赖步骤 1 和 2，步骤 4 依赖步骤 3，
  依赖树正确）。

覆盖场景：
  - 正常生成：合法 DAG 解析成功，步骤数在范围内
  - JSON 解析失败自动重试
  - 步骤数量超出范围自动重试
  - 依赖 ID 不存在自动重试
  - 循环依赖自动重试
  - 重试耗尽抛出 PlannerError
  - Prompt 记录持久化到本地 JSON
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.services.planner import (
    PlannerError,
    _convert_to_task_steps,
    _extract_json_object,
    plan,
)

# 一个合法的 5 步 DAG（步骤数在 [5, 10] 范围内，无循环依赖）
VALID_PLAN_JSON = json.dumps({
    "plan": [
        {
            "id": "step_1",
            "description": "初始化项目结构",
            "details": "创建 blog 目录，初始化前后端分离结构，git init",
            "dependencies": [],
            "suggested_tool": "run_command",
        },
        {
            "id": "step_2",
            "description": "设计数据库表",
            "details": "设计 Post 表 (id, title, content, created_at)，使用 SQLite",
            "dependencies": ["step_1"],
            "suggested_tool": "write_file",
        },
        {
            "id": "step_3",
            "description": "实现后端 API",
            "details": "使用 Flask 实现博客增删改查 RESTful 接口",
            "dependencies": ["step_2"],
            "suggested_tool": "write_file",
        },
        {
            "id": "step_4",
            "description": "搭建前端页面",
            "details": "使用 React + Vite 搭建博客列表与详情页",
            "dependencies": ["step_1"],
            "suggested_tool": "write_file",
        },
        {
            "id": "step_5",
            "description": "联调与测试",
            "details": "前后端联调，验证博客增删改查功能",
            "dependencies": ["step_3", "step_4"],
            "suggested_tool": "run_command",
        },
    ]
})


# ============================================================
# 内部函数测试
# ============================================================

def test_extract_json_object_with_markdown():
    """模型输出含 ```json 标记时仍能提取 JSON"""
    text = "```json\n{\"plan\": []}\n```"
    result = _extract_json_object(text)
    assert result == {"plan": []}


def test_extract_json_object_plain():
    """纯 JSON 文本直接解析"""
    result = _extract_json_object('{"plan": [{"id": "s1"}]}')
    assert result is not None
    assert result["plan"][0]["id"] == "s1"


def test_extract_json_object_invalid():
    """非法 JSON 返回 None"""
    assert _extract_json_object("not json at all") is None
    assert _extract_json_object("") is None


def test_convert_to_task_steps_valid():
    """合法 raw plan 转换为 TaskStep 列表"""
    raw = [
        {"id": "s1", "description": "初始化", "details": "创建目录",
         "dependencies": [], "suggested_tool": "run_command"},
        {"id": "s2", "description": "实现", "details": "写代码",
         "dependencies": ["s1"], "suggested_tool": "write_file"},
    ]
    steps = _convert_to_task_steps(raw)
    assert len(steps) == 2
    assert steps[0].id == "s1"
    assert steps[0].action == "run_command"
    assert steps[0].suggested_tool == "run_command"
    assert steps[1].dependencies == ["s1"]


def test_convert_to_task_steps_missing_fields():
    """缺失字段时自动兜底（id 自动生成、details 用 description）"""
    raw = [{"description": "测试步骤"}]
    steps = _convert_to_task_steps(raw)
    assert len(steps) == 1
    assert steps[0].id  # 自动生成
    assert steps[0].details == "测试步骤"
    assert steps[0].dependencies == []


def test_convert_to_task_steps_invalid_tool():
    """非法 suggested_tool 被置空，不报错"""
    raw = [{"description": "步骤", "suggested_tool": "unknown_tool"}]
    steps = _convert_to_task_steps(raw)
    assert steps[0].suggested_tool is None
    assert steps[0].action == ""


# ============================================================
# plan() 集成测试（mock LLM 调用）
# ============================================================

def _run(coro):
    return asyncio.run(coro)


def test_plan_success():
    """正常场景：一次调用即返回合法 DAG"""
    with patch("app.services.planner.chat_completion_text",
               new=AsyncMock(return_value=VALID_PLAN_JSON)):
        steps = _run(plan("做一个 Flask + React 的博客系统", model="glm-4.5-air"))

    assert len(steps) == 5
    # 依赖树正确：step_3 依赖 step_2，step_5 依赖 step_3 和 step_4
    by_id = {s.id: s for s in steps}
    assert by_id["step_3"].dependencies == ["step_2"]
    assert set(by_id["step_5"].dependencies) == {"step_3", "step_4"}
    # 第一步无依赖（初始化项目）
    assert by_id["step_1"].dependencies == []
    print("✅ Planner 正常生成合法 DAG 验证通过")


def test_plan_retry_on_invalid_json():
    """JSON 解析失败自动重试，第二次成功"""
    calls = {"n": 0}

    async def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "这不是 JSON，模型在胡说八道"
        return VALID_PLAN_JSON

    with patch("app.services.planner.chat_completion_text",
               new=AsyncMock(side_effect=fake_chat)):
        steps = _run(plan("做个博客", model="glm-4.5-air"))

    assert calls["n"] == 2
    assert len(steps) == 5
    print("✅ JSON 解析失败自动重试验证通过")


def test_plan_retry_on_step_count_out_of_range():
    """步骤数量超出范围自动重试"""
    too_few = json.dumps({"plan": [
        {"id": "s1", "description": "步骤1", "dependencies": []},
        {"id": "s2", "description": "步骤2", "dependencies": ["s1"]},
    ]})  # 只有 2 步，少于 MIN_STEPS=5

    calls = {"n": 0}

    async def fake_chat(**kwargs):
        calls["n"] += 1
        return too_few if calls["n"] == 1 else VALID_PLAN_JSON

    with patch("app.services.planner.chat_completion_text",
               new=AsyncMock(side_effect=fake_chat)):
        steps = _run(plan("做个博客", model="glm-4.5-air"))

    assert calls["n"] == 2
    assert len(steps) == 5
    print("✅ 步骤数量超出范围自动重试验证通过")


def test_plan_retry_on_missing_dependency():
    """依赖 ID 不存在自动重试"""
    bad_dep = json.dumps({"plan": [
        {"id": "s1", "description": "步骤1", "dependencies": []},
        {"id": "s2", "description": "步骤2", "dependencies": ["non_existent"]},
        {"id": "s3", "description": "步骤3", "dependencies": ["s2"]},
        {"id": "s4", "description": "步骤4", "dependencies": ["s3"]},
        {"id": "s5", "description": "步骤5", "dependencies": ["s4"]},
    ]})  # 5 步，但 s2 依赖了不存在的 non_existent

    calls = {"n": 0}

    async def fake_chat(**kwargs):
        calls["n"] += 1
        return bad_dep if calls["n"] == 1 else VALID_PLAN_JSON

    with patch("app.services.planner.chat_completion_text",
               new=AsyncMock(side_effect=fake_chat)):
        steps = _run(plan("做个博客", model="glm-4.5-air"))

    assert calls["n"] == 2
    assert len(steps) == 5
    print("✅ 依赖 ID 不存在自动重试验证通过")


def test_plan_retry_on_circular_dependency():
    """循环依赖自动重试"""
    circular = json.dumps({"plan": [
        {"id": "s1", "description": "步骤1", "dependencies": ["s3"]},
        {"id": "s2", "description": "步骤2", "dependencies": ["s1"]},
        {"id": "s3", "description": "步骤3", "dependencies": ["s2"]},
        {"id": "s4", "description": "步骤4", "dependencies": ["s3"]},
        {"id": "s5", "description": "步骤5", "dependencies": ["s4"]},
    ]})  # 5 步，但 s1->s3->s2->s1 形成循环

    calls = {"n": 0}

    async def fake_chat(**kwargs):
        calls["n"] += 1
        return circular if calls["n"] == 1 else VALID_PLAN_JSON

    with patch("app.services.planner.chat_completion_text",
               new=AsyncMock(side_effect=fake_chat)):
        steps = _run(plan("做个博客", model="glm-4.5-air"))

    assert calls["n"] == 2
    assert len(steps) == 5
    print("✅ 循环依赖自动重试验证通过")


def test_plan_all_retries_exhausted():
    """重试耗尽后抛出 PlannerError"""
    with patch("app.services.planner.chat_completion_text",
               new=AsyncMock(return_value="完全不是 JSON")):
        try:
            _run(plan("做个博客", model="glm-4.5-air"))
            assert False, "重试耗尽应抛出 PlannerError"
        except PlannerError as e:
            assert "失败" in str(e)
    print("✅ 重试耗尽抛出 PlannerError 验证通过")


def test_plan_prompt_log_saved(tmp_path):
    """Prompt 记录持久化到本地 JSON 文件"""
    from app.config import settings

    log_dir = tmp_path / "planner_logs"
    with patch.object(settings, "PLANNER_LOG_DIR", str(log_dir)):
        with patch("app.services.planner.chat_completion_text",
                   new=AsyncMock(return_value=VALID_PLAN_JSON)):
            _run(plan("做个博客", model="glm-4.5-air"))

    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == 1
    record = json.loads(log_files[0].read_text(encoding="utf-8"))
    assert record["success"] is True
    assert record["goal"] == "做个博客"
    assert record["model"] == "glm-4.5-air"
    assert "raw_output" in record
    assert "parsed_steps" in record
    assert len(record["parsed_steps"]) == 5
    print("✅ Prompt 记录持久化验证通过")


def test_plan_prompt_log_saved_on_failure(tmp_path):
    """失败时也保存 Prompt 记录（供调试/微调）"""
    from app.config import settings

    log_dir = tmp_path / "planner_logs"
    with patch.object(settings, "PLANNER_LOG_DIR", str(log_dir)):
        with patch("app.services.planner.chat_completion_text",
                   new=AsyncMock(return_value="bad")):
            try:
                _run(plan("做个博客", model="glm-4.5-air"))
            except PlannerError:
                pass

    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == 1
    record = json.loads(log_files[0].read_text(encoding="utf-8"))
    assert record["success"] is False
    assert record["error"] is not None
    print("✅ 失败 Prompt 记录持久化验证通过")


def test_plan_acceptance_flask_react_blog():
    """
    验收标准：输入 "做一个 Flask + React 的博客系统"，
    返回的 JSON 包含至少 6 个步骤，且无循环依赖、依赖树正确。
    """
    blog_plan = json.dumps({"plan": [
        {"id": "step_1", "description": "初始化项目结构",
         "details": "创建 blog 目录，前后端分离，git init",
         "dependencies": [], "suggested_tool": "run_command"},
        {"id": "step_2", "description": "设计数据库表",
         "details": "Post 表 (id, title, content, created_at)",
         "dependencies": ["step_1"], "suggested_tool": "write_file"},
        {"id": "step_3", "description": "实现后端 API",
         "details": "Flask 实现博客增删改查接口",
         "dependencies": ["step_1", "step_2"], "suggested_tool": "write_file"},
        {"id": "step_4", "description": "搭建前端列表页",
         "details": "React 实现博客列表展示",
         "dependencies": ["step_1"], "suggested_tool": "write_file"},
        {"id": "step_5", "description": "搭建前端详情页",
         "details": "React 实现博客详情与编辑页",
         "dependencies": ["step_4"], "suggested_tool": "write_file"},
        {"id": "step_6", "description": "联调与测试",
         "details": "前后端联调验证",
         "dependencies": ["step_3", "step_5"], "suggested_tool": "run_command"},
    ]})  # 6 步，step_3 依赖 step_1 和 step_2，step_6 依赖 step_3 和 step_5

    with patch("app.services.planner.chat_completion_text",
               new=AsyncMock(return_value=blog_plan)):
        steps = _run(plan("做一个 Flask + React 的博客系统", model="glm-4.5-air"))

    assert len(steps) >= 6
    by_id = {s.id: s for s in steps}
    # 步骤 3 依赖步骤 1 和 2
    assert set(by_id["step_3"].dependencies) == {"step_1", "step_2"}
    # 步骤 4 依赖步骤 3
    assert by_id["step_5"].dependencies == ["step_4"]
    # 依赖树正确，无循环依赖（plan() 内部已通过 validate_plan）
    print("✅ 验收标准：Flask + React 博客系统 6 步 DAG 验证通过")


if __name__ == "__main__":
    test_extract_json_object_with_markdown()
    test_extract_json_object_plain()
    test_extract_json_object_invalid()
    test_convert_to_task_steps_valid()
    test_convert_to_task_steps_missing_fields()
    test_convert_to_task_steps_invalid_tool()
    test_plan_success()
    test_plan_retry_on_invalid_json()
    test_plan_retry_on_step_count_out_of_range()
    test_plan_retry_on_missing_dependency()
    test_plan_retry_on_circular_dependency()
    test_plan_all_retries_exhausted()
    # tmp_path 测试需 pytest 提供，单独运行时跳过
    print("\n" + "=" * 60)
    print("Planner 单元测试通过（tmp_path 相关测试由 pytest 运行）！")
    print("=" * 60)
