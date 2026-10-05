"""
S9 第 85-86 天：测试沙箱集成 —— 用"自动化测试"驱动修复

核心能力：
1. test_runner.py：检测项目中的测试文件与框架（pytest / unittest / jest），
   运行测试套件并返回结构化结果（passed / failed / errors / skipped）。
2. run_tests 工具：Agent 可主动调用该工具；self_repair 循环在代码修改后
   自动触发 run_tests，将测试失败也作为"错误"来源，驱动新一轮修复。
3. 自动依赖安装：检测 node_modules / venv / .venv 是否存在，不存在则先执行
   npm install / pip install -r requirements.txt。
4. 超时保护：默认 30s，超时后主动中止，防止 Agent 卡死在测试上。

风险预警防护（对应 S9 关键技术预研）：
  - 测试套件依赖复杂 → AUTO_INSTALL_DEPS 自动安装
  - 大模型"幻觉修复" → 测试作为唯一可靠验收裁判，有测试的项目强制跑通
  - 超时卡死 → TEST_RUNNER_DEFAULT_TIMEOUT + asyncio.wait_for 强制超时
  - node_modules / .venv 扫描过慢 → 扫描时跳过这些目录 + 文件数上限

设计要点：
  - 纯异步：所有外部调用（命令执行 / LLM）都是 async
  - 复用 SandboxOrchestrator：命令执行统一走沙箱编排器，Docker 模式下
    测试在容器内执行（安全 + 环境一致）
  - 结果结构化：返回 TestRunResult（Pydantic 模型），便于 SSE 推送
  - 降级策略：pytest 不可用 → 降级 unittest；jest 不可用 → 降级 node --test
"""

import asyncio
import json
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

TestFramework = Literal["pytest", "unittest", "jest", "npm_test", "unknown"]


class TestFailure(BaseModel):
    """单条测试失败详情（供 Builder 面板渲染失败列表）"""
    test_name: str = Field(..., description="测试用例全名，如 'tests/test_add.py::TestMath::test_add'")
    error: str = Field(default="", description="错误摘要，如 'assert 3 == 4'")
    file: Optional[str] = Field(default=None, description="测试文件路径（相对 workspace_root）")
    line: Optional[int] = Field(default=None, description="失败行号（1-based）")
    stack_trace: Optional[str] = Field(default=None, description="完整堆栈（可能很长，按需截断）")


class TestRunResult(BaseModel):
    """
    测试执行结果（S9 关键接口变更）。

    对应 Sprint_9.md 新增的 run_tests 工具返回结构：
    {
      "success": true,
      "output": "Ran 5 tests: 4 passed, 1 failed",
      "failures": [{"test_name": "test_add", "error": "assert 3 == 4"}]
    }
    """
    success: bool = Field(..., description="整体是否通过（全部 passed 时为 True）")
    framework: TestFramework = Field(..., description="实际检测到的测试框架")
    passed: int = Field(default=0, description="通过的测试数")
    failed: int = Field(default=0, description="失败的测试数")
    errors: int = Field(default=0, description="报错（非断言失败）的测试数")
    skipped: int = Field(default=0, description="跳过的测试数")
    total: int = Field(default=0, description="总测试数")
    duration_ms: float = Field(default=0.0, description="测试执行耗时（毫秒）")
    failures: List[TestFailure] = Field(default_factory=list, description="失败用例详情列表")
    stdout: str = Field(default="", description="原始 stdout 输出（截断后）")
    stderr: str = Field(default="", description="原始 stderr 输出（截断后）")
    timed_out: bool = Field(default=False, description="是否因超时被强制中止")
    error_message: Optional[str] = Field(default=None, description="框架运行失败时的错误描述")


# ============================================================
# 自动检测：测试文件扫描
# ============================================================

# 要跳过的大型依赖目录（扫描时排除，提升速度）
_SKIP_DIR_NAMES = {
    "node_modules", ".venv", "venv", "__pycache__", ".git", ".ai_index", ".ai_cache",
    "dist", "build", ".next", ".nuxt", ".svelte-kit",
}

# Python 测试文件匹配模式
_PYTHON_TEST_PATTERNS = [
    re.compile(r"^test_.*\.py$"),          # test_xxx.py
    re.compile(r".*_test\.py$"),            # xxx_test.py
]

# JavaScript 测试文件匹配模式
_JS_TEST_PATTERNS = [
    re.compile(r".*\.test\.(js|ts|jsx|tsx)$"),    # xxx.test.js
    re.compile(r".*\.spec\.(js|ts|jsx|tsx)$"),    # xxx.spec.js
    re.compile(r"^test_.*\.(js|ts)$"),             # test_xxx.js
]


def _should_skip_dir(dirname: str) -> bool:
    """判断目录名是否在跳过列表中"""
    return dirname in _SKIP_DIR_NAMES or dirname.startswith(".")


def _match_patterns(filename: str, patterns: List[re.Pattern]) -> bool:
    """文件名是否匹配任一模式"""
    for p in patterns:
        if p.match(filename):
            return True
    return False


async def _scan_test_files(workspace_root: str) -> Dict[str, List[str]]:
    """
    扫描工作区中的测试文件，按分类返回。

    Returns:
        {"python": [相对路径列表], "javascript": [相对路径列表]}
        未找到时对应列表为空。
    """
    root = Path(workspace_root)
    if not root.exists():
        return {"python": [], "javascript": []}

    python_files: List[str] = []
    js_files: List[str] = []
    scan_count = 0
    max_scan = settings.TEST_RUNNER_MAX_SCAN_FILES

    # 使用 asyncio.to_thread 避免阻塞事件循环
    def _do_scan():
        nonlocal scan_count
        for dirpath, dirnames, filenames in os.walk(root):
            # 原地修改 dirnames 来跳过大型目录
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
            for fname in filenames:
                scan_count += 1
                if scan_count > max_scan:
                    return
                if _match_patterns(fname, _PYTHON_TEST_PATTERNS):
                    rel = os.path.relpath(os.path.join(dirpath, fname), root).replace("\\", "/")
                    python_files.append(rel)
                elif _match_patterns(fname, _JS_TEST_PATTERNS):
                    rel = os.path.relpath(os.path.join(dirpath, fname), root).replace("\\", "/")
                    js_files.append(rel)

    if settings.TEST_RUNNER_SEARCH_RECURSIVE:
        await asyncio.to_thread(_do_scan)
    else:
        # 仅扫描根目录
        for fname in os.listdir(root):
            scan_count += 1
            full = root / fname
            if full.is_file() and _match_patterns(fname, _PYTHON_TEST_PATTERNS):
                python_files.append(fname)
            elif full.is_file() and _match_patterns(fname, _JS_TEST_PATTERNS):
                js_files.append(fname)

    logger.info(
        f"[TestRunner] 测试文件扫描完成: python={len(python_files)}, "
        f"javascript={len(js_files)}, scanned={scan_count}"
    )
    return {"python": python_files, "javascript": js_files}


# ============================================================
# 框架自动检测
# ============================================================

async def detect_test_framework(workspace_root: str, explicit_framework: Optional[str] = None) -> TestFramework:
    """
    自动检测项目使用的测试框架。

    检测逻辑（优先级从高到低）：
      1. 显式指定 → 直接返回
      2. package.json + npm test → npm_test / jest
      3. requirements.txt / pyproject.toml + test_*.py → pytest / unittest
      4. 测试文件存在但无依赖 → 按文件类型默认 pytest / npm test
      5. 无测试文件 → unknown

    Returns:
        检测到的 TestFramework。unknown 表示无法检测或无测试。
    """
    if explicit_framework and explicit_framework != "auto":
        fw = explicit_framework.lower()
        if fw in ("pytest", "unittest", "jest", "npm_test", "unknown"):
            return fw  # type: ignore[return-value]
        logger.warning(f"[TestRunner] 指定的框架未知: {explicit_framework}，将自动检测")

    test_files = await _scan_test_files(workspace_root)
    python_files = test_files["python"]
    js_files = test_files["javascript"]

    if not python_files and not js_files:
        logger.info("[TestRunner] 未找到任何测试文件")
        return "unknown"

    # 检查 Python 项目配置
    root = Path(workspace_root)
    has_requirements = (root / "requirements.txt").exists()
    has_pyproject = (root / "pyproject.toml").exists()

    # 检查 package.json + jest
    has_package_json = (root / "package.json").exists()
    uses_jest = False
    uses_npm_test = False
    try:
        if has_package_json:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            scripts = pkg.get("scripts", {})
            uses_jest = "jest" in deps
            uses_npm_test = "test" in scripts and scripts["test"]  # 非空
    except (json.JSONDecodeError, OSError):
        pass

    # 决策：优先选择有配置的
    if uses_jest:
        return "jest"
    elif uses_npm_test and js_files:
        return "npm_test"
    elif js_files:
        # 有 JS 测试文件但无 package.json → 尝试 node --test
        return "jest"  # 默认假设 jest

    if has_pyproject or has_requirements or python_files:
        # Python 项目：优先 pytest（S9 验收场景用的 pytest）
        return "pytest"

    return "unknown"


# ============================================================
# 依赖安装（风险预警应对）
# ============================================================

async def _check_python_deps_ready(workspace_root: str) -> bool:
    """
    检查 Python 项目依赖是否就绪。

    判定条件（满足任一即视为就绪）：
      - 工作区下有 .venv / venv 目录
      - python -c "import pytest" 不报错（pytest 已在当前环境可用）
      - requirements.txt 不存在（简单项目）
    """
    root = Path(workspace_root)
    if (root / ".venv").exists() or (root / "venv").exists():
        return True
    if not (root / "requirements.txt").exists() and not (root / "pyproject.toml").exists():
        return True  # 无依赖文件 → 不需要安装

    # 检查 pytest 是否可导入（用沙箱执行 python -c）
    try:
        orch = await _get_sandbox_orchestrator()
        exit_code, _, _ = await orch.execute_command(
            cmd='python -c "import pytest"',
            workspace_root=workspace_root,
            session_id="test-runner-deps-check",
            timeout=5.0,
        )
        if exit_code == 0:
            return True
    except Exception:
        pass
    return False


async def _check_node_deps_ready(workspace_root: str) -> bool:
    """检查 Node.js 项目依赖是否就绪（node_modules 存在）"""
    root = Path(workspace_root)
    return (root / "node_modules").exists()


async def _install_python_deps(workspace_root: str) -> Tuple[bool, str]:
    """
    安装 Python 项目依赖（pip install -r requirements.txt 或 pip install pytest）。

    Returns:
        (success, message)
    """
    root = Path(workspace_root)
    orch = await _get_sandbox_orchestrator()

    if (root / "requirements.txt").exists():
        cmd = "pip install -r requirements.txt"
    elif (root / "pyproject.toml").exists():
        cmd = "pip install pytest"
    else:
        cmd = "pip install pytest"

    try:
        exit_code, stdout, stderr = await orch.execute_command(
            cmd=cmd,
            workspace_root=workspace_root,
            session_id="test-runner-deps-install",
            timeout=settings.TEST_RUNNER_DEPS_INSTALL_TIMEOUT,
        )
        if exit_code == 0:
            return True, "依赖安装成功"
        err = (stderr or stdout or "").strip().splitlines()
        last = err[-1] if err else "未知错误"
        return False, f"依赖安装失败: {last}"
    except Exception as e:
        return False, f"依赖安装异常: {e}"


async def _install_node_deps(workspace_root: str) -> Tuple[bool, str]:
    """安装 Node.js 项目依赖（npm install）"""
    orch = await _get_sandbox_orchestrator()
    try:
        exit_code, stdout, stderr = await orch.execute_command(
            cmd="npm install",
            workspace_root=workspace_root,
            session_id="test-runner-deps-install",
            timeout=settings.TEST_RUNNER_DEPS_INSTALL_TIMEOUT,
        )
        if exit_code == 0:
            return True, "npm install 成功"
        err = (stderr or stdout or "").strip().splitlines()
        last = err[-1] if err else "未知错误"
        return False, f"npm install 失败: {last}"
    except Exception as e:
        return False, f"npm install 异常: {e}"


# ============================================================
# 测试结果解析
# ============================================================

# pytest 输出解析
_PYTEST_SUMMARY_RE = re.compile(
    r"(\d+)\s*(?:passed|failed|error|skipped|warnings?)\b"
)
_PYTEST_FAILURE_LINE_RE = re.compile(
    r"^FAILED\s+(?:\[.*?\]\s+)?(.+?)(?:\s*-\s*(.+))?$"
)
_PYTEST_ERROR_LINE_RE = re.compile(
    r"^ERROR\s+(.+?)(?:\s*-\s*(.+))?$"
)
_PYTEST_LOC_RE = re.compile(
    r"(\w[\w/.-]+\.py):(\d+)(?: in (\w+))?"
)


def _parse_pytest_output(stdout: str, stderr: str, exit_code: int) -> TestRunResult:
    """
    解析 pytest 输出为结构化 TestRunResult。

    pytest 退出码含义：
      0: 全部通过
      1: 有测试失败
      2: 用法错误 / 导入错误
      5: 无测试用例收集到
    """
    full = (stdout or "") + "\n" + (stderr or "")

    # 从尾部 Summary 行提取数字。pytest 可能输出：
    #   "1 passed in 0.10s"                    — 全部通过
    #   "1 failed, 1 passed in 0.09s"          — 有失败
    #   "2 passed, 1 skipped in 0.10s"         — 有跳过
    #   "1 failed, 1 error, 1 passed, 2 skipped in 0.15s"  — 全有
    # 数字出现顺序不固定，需逐个匹配
    passed = failed = errors = skipped = 0

    passed_m = re.search(r"(\d+)\s+passed\b", full)
    failed_m = re.search(r"(\d+)\s+failed\b", full)
    error_m = re.search(r"(\d+)\s+error[s]?\b", full)
    skipped_m = re.search(r"(\d+)\s+skipped\b", full)

    if passed_m:
        passed = int(passed_m.group(1))
    if failed_m:
        failed = int(failed_m.group(1))
    if error_m:
        errors = int(error_m.group(1))
    if skipped_m:
        skipped = int(skipped_m.group(1))
    total = passed + failed + errors + skipped

    # 从详细输出中提取失败用例
    failures: List[TestFailure] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        fm = _PYTEST_FAILURE_LINE_RE.match(line)
        if fm:
            test_full = fm.group(1).strip()
            err_msg = (fm.group(2) or "").strip()
            # 尝试从 test_full 中提取文件和行号
            file_path = None
            line_no = None
            loc_m = _PYTEST_LOC_RE.search(test_full)
            if loc_m:
                file_path = loc_m.group(1)
                line_no = int(loc_m.group(2))

            failures.append(TestFailure(
                test_name=test_full,
                error=err_msg or "assertion failed",
                file=file_path,
                line=line_no,
            ))
            continue

        em = _PYTEST_ERROR_LINE_RE.match(line)
        if em:
            test_full = em.group(1).strip()
            err_msg = (em.group(2) or "").strip()
            failures.append(TestFailure(
                test_name=test_full,
                error=err_msg or "test error",
                file=None,
                line=None,
            ))

    # 如果 FAILED 行没匹配到，但 errors > 0 且 exit_code != 0，
    # 尝试用更宽松的正则补充
    if not failures and (failed > 0 or errors > 0):
        for line in (stdout or "").splitlines():
            line = line.strip()
            # 匹配 "FAILED test_main.py::test_add_fail" 格式
            m = re.match(r"FAILED\s+(\S+)", line)
            if m:
                test_full = m.group(1).strip()
                failures.append(TestFailure(
                    test_name=test_full,
                    error="assertion failed",
                    file=None,
                    line=None,
                ))

    success = (failed == 0 and errors == 0 and exit_code == 0)

    return TestRunResult(
        success=success,
        framework="pytest",
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total=total,
        failures=failures[:20],  # 最多保留 20 条失败（太多会撑爆 SSE）
        stdout=(stdout or "")[:5000],
        stderr=(stderr or "")[:2000],
        timed_out=False,
    )


# Jest 输出解析
_JEST_SUMMARY_RE = re.compile(
    r"Tests:\s*(\d+)\s*(?:passed|failed)?,\s*(\d+)\s*(?:passed|failed)?.*?"
    r"(\d+)\s*(?:passed|failed)?.*?"
    r"(?:(\d+)\s*skipped)?"
)
_JEST_FAIL_RE = re.compile(r"●\s+(.+)")


def _parse_jest_output(stdout: str, stderr: str, exit_code: int) -> TestRunResult:
    """解析 Jest / npm test 输出"""
    full = (stdout or "") + "\n" + (stderr or "")

    passed = failed = skipped = total = 0

    # Jest 可能输出多种格式：
    #   "Tests: 5 passed, 2 failed, 1 skipped"
    #   "Test Suites: 1 failed, 3 passed, 4 total"
    #   "Tests:       2 failed, 5 passed, 7 total"
    m = re.search(
        r"Tests:\s+(\d+)\s+failed,\s*(\d+)\s+passed[,\s]*(\d+)\s+skipped",
        full
    )
    if m:
        failed = int(m.group(1))
        passed = int(m.group(2))
        skipped = int(m.group(3))
        total = passed + failed + skipped
    else:
        m2 = re.search(r"Tests:\s+(\d+)\s+passed", full)
        if m2:
            passed = int(m2.group(1))
            total = passed
        # 从退出码推断 failed
        if exit_code != 0 and total == 0:
            failed = 1
            total = 1

    # 提取失败详情
    failures: List[TestFailure] = []
    for line in (stdout or "").splitlines():
        fm = _JEST_FAIL_RE.match(line.strip())
        if fm:
            test_full = fm.group(1).strip()
            failures.append(TestFailure(
                test_name=test_full,
                error="test failed",
                file=None,
                line=None,
            ))

    success = (failed == 0 and exit_code == 0)

    return TestRunResult(
        success=success,
        framework="jest" if full else "npm_test",
        passed=passed,
        failed=failed,
        skipped=skipped,
        total=total,
        failures=failures[:20],
        stdout=(stdout or "")[:5000],
        stderr=(stderr or "")[:2000],
        timed_out=False,
    )


def _parse_timeout_result(framework: TestFramework, stdout: str, stderr: str) -> TestRunResult:
    """超时情况下生成的 TestRunResult"""
    return TestRunResult(
        success=False,
        framework=framework,
        passed=0,
        failed=0,
        errors=1,
        skipped=0,
        total=0,
        failures=[],
        stdout=(stdout or "")[:2000],
        stderr=(stderr or "")[:2000],
        timed_out=True,
        error_message=f"测试执行超时（{settings.TEST_RUNNER_DEFAULT_TIMEOUT}s），已主动中止。"
                       "请手动在终端检查测试。",
    )


# ============================================================
# 命令构建（按框架生成测试命令）
# ============================================================

def _build_test_command(framework: TestFramework, test_path: Optional[str] = None) -> str:
    """
    根据检测到的框架构建测试执行命令。

    降级策略（对应 S9 风险预警）：
      - pytest 不可执行 → 降级为 python -m pytest
      - 仍不可用 → 提示用户

    Args:
        framework: 检测到的测试框架。
        test_path: 可选，指定要运行的测试文件/目录。None 时运行全部。

    Returns:
        shell 命令字符串。
    """
    if framework == "pytest":
        # 优先尝试 pytest 命令，降级为 python -m pytest
        try:
            import shutil as _shutil
            if _shutil.which("pytest"):
                base = "pytest --tb=short -q"
            else:
                base = "python -m pytest --tb=short -q"
        except Exception:
            base = "python -m pytest --tb=short -q"

        if test_path:
            base += f" {shlex.quote(test_path)}"
        return base
    elif framework == "unittest":
        base = "python -m unittest discover -v"
        if test_path:
            base = f"python -m unittest {shlex.quote(test_path)}"
        return base
    elif framework in ("jest", "npm_test"):
        base = "npm test -- --no-coverage"
        if test_path and framework == "jest":
            base = f"npx jest {shlex.quote(test_path)} --no-coverage"
        return base
    else:
        return "python -m pytest --tb=short -q"  # 兜底


# ============================================================
# 核心执行入口：run_tests
# ============================================================

async def _get_sandbox_orchestrator():
    """Lazy 获取 SandboxOrchestrator（复用 tool_registry 的模式）"""
    from app.services.sandbox_orchestrator import get_sandbox_manager
    manager = get_sandbox_manager()
    return await manager.orchestrator()


async def run_tests(
    workspace_root: str,
    framework: Optional[str] = None,
    test_path: Optional[str] = None,
    timeout: Optional[float] = None,
    session_id: str = "",
) -> TestRunResult:
    """
    运行项目测试套件（S9 核心函数）。

    流程：
      1. 检测测试框架（pytest / unittest / jest / npm_test）。
      2. 自动安装依赖（若需且启用）。
      3. 执行测试命令（通过 SandboxOrchestrator）。
      4. 解析输出为结构化 TestRunResult。
      5. 超时保护：asyncio.wait_for(timeout)。

    风险预警防护：
      - 超时强制中止（asyncio.wait_for + 进程组 SIGTERM）
      - 依赖缺失自动安装（TEST_RUNNER_AUTO_INSTALL_DEPS）
      - 无测试框架 → 返回 error_message 提示

    Args:
        workspace_root: 项目根目录（cwd）。
        framework:      显式指定框架（pytest / unittest / jest / npm_test / auto）。
                        None 或 "auto" 时自动检测。
        test_path:      可选，只运行指定的测试文件/目录。
        timeout:        测试执行超时（秒）。None 时用默认值。
        session_id:     Agent 会话 ID（用于审计日志）。

    Returns:
        TestRunResult（结构化测试结果）。
    """
    if not settings.TEST_RUNNER_ENABLED:
        return TestRunResult(
            success=False,
            framework="unknown",
            error_message="测试运行功能已全局禁用（TEST_RUNNER_ENABLED=False）",
        )

    # 1. 检测框架
    fw = await detect_test_framework(workspace_root, framework)
    if fw == "unknown":
        return TestRunResult(
            success=False,
            framework="unknown",
            error_message="未检测到测试框架或测试文件。"
                           "请确认项目中存在 pytest / unittest / jest 测试文件。",
        )
    logger.info(f"[TestRunner] 检测到测试框架: {fw}")

    # 2. 依赖检查与安装（风险预警：测试套件依赖复杂）
    if settings.TEST_RUNNER_AUTO_INSTALL_DEPS:
        deps_ok = False
        if fw in ("pytest", "unittest"):
            deps_ok = await _check_python_deps_ready(workspace_root)
            if not deps_ok:
                logger.info("[TestRunner] Python 依赖未就绪，正在安装...")
                success, msg = await _install_python_deps(workspace_root)
                logger.info(f"[TestRunner] 依赖安装结果: {success}, {msg}")
        elif fw in ("jest", "npm_test"):
            deps_ok = await _check_node_deps_ready(workspace_root)
            if not deps_ok:
                logger.info("[TestRunner] Node.js 依赖未就绪，正在 npm install...")
                success, msg = await _install_node_deps(workspace_root)
                logger.info(f"[TestRunner] npm install 结果: {success}, {msg}")
        # 依赖安装失败不阻断——测试命令本身可能已经能运行

    # 3. 构建命令 + 执行
    cmd = _build_test_command(fw, test_path)
    effective_timeout = timeout or settings.TEST_RUNNER_DEFAULT_TIMEOUT
    effective_timeout = min(effective_timeout, settings.TEST_RUNNER_MAX_TIMEOUT)

    orch = await _get_sandbox_orchestrator()
    start_time = asyncio.get_event_loop().time()

    try:
        exit_code, stdout_text, stderr_text = await asyncio.wait_for(
            orch.execute_command(
                cmd=cmd,
                workspace_root=workspace_root,
                session_id=session_id or "test-runner",
                timeout=effective_timeout,
            ),
            timeout=effective_timeout + 5,  # 额外 5s 让沙箱清理
        )
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
    except asyncio.TimeoutError:
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.warning(
            f"[TestRunner] 测试执行超时（{effective_timeout}s）: "
            f"cmd={cmd}, session={session_id}"
        )
        return _parse_timeout_result(fw, "", "")
    except Exception as e:
        logger.error(f"[TestRunner] 测试执行异常: {e}", exc_info=True)
        return TestRunResult(
            success=False,
            framework=fw,
            error_message=f"测试执行异常: {type(e).__name__}: {e}",
        )

    # 4. 解析输出
    if fw in ("pytest", "unittest"):
        result = _parse_pytest_output(stdout_text, stderr_text, exit_code)
    else:
        result = _parse_jest_output(stdout_text, stderr_text, exit_code)

    result.duration_ms = round(duration_ms, 1)

    logger.info(
        f"[TestRunner] 测试执行完成: framework={fw}, "
        f"passed={result.passed}, failed={result.failed}, "
        f"errors={result.errors}, total={result.total}, "
        f"duration={result.duration_ms:.0f}ms, timed_out={result.timed_out}"
    )
    return result


# ============================================================
# 工具函数：快速检测（供 self_repair 判断是否需要自动跑测试）
# ============================================================

async def has_tests_in_project(workspace_root: str) -> bool:
    """
    快速判断工作区是否存在任何测试文件（供 self_repair 决策）。

    比完整的 run_tests 轻量得多——只做文件扫描，不执行任何命令。
    """
    test_files = await _scan_test_files(workspace_root)
    return bool(test_files["python"] or test_files["javascript"])


def result_to_summary(result: TestRunResult) -> str:
    """将 TestRunResult 转为一句话摘要（供日志 / SSE 推送）"""
    if result.timed_out:
        return f"测试超时中止（{result.error_message}）"
    if result.success:
        return f"测试全部通过 ✅ ({result.passed} passed)"
    if result.error_message:
        return f"测试框架不可用: {result.error_message}"
    return (
        f"测试存在失败 ❌ ({result.passed} passed, {result.failed} failed, "
        f"{result.errors} errors, {result.skipped} skipped)"
    )
