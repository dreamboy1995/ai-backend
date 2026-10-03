"""
Diff 生成器（S6 第 53-54 天：多文件 JSON Mode 配套）

当模型以 JSON Mode 返回多文件修改结果后，后端需要：
1. 读取每个文件的原始内容（从工作区磁盘）。
2. 使用 difflib 生成原始内容与模型返回新内容之间的 Unified Diff。
3. 将 diff 元数据随 SSE 流下发给前端，供 DiffPreviewPanel 渲染。

关键设计（对应 S6 风险预警）：
- 路径安全：模型返回的 path 必须落在 workspace_root 内，防止
  ../../../etc/passwd 之类的路径逃逸读取任意文件。
- 大文件上下文：difflib.unified_diff 的 n 参数控制上下文行数（默认 3），
  超大文件只保留变更行附近的 3 行上下文，避免 Diff 数据过大撑爆 SSE。
- 容错：原文件不存在（新增文件）时 old_content 为空；读取失败时降级为空，
  不阻断整体流程。
"""

import logging
import os
import difflib
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# S6 第 55-56 天：超大文件只生成变更行附近的 Diff 上下文（上下文行数 = 3）
DEFAULT_DIFF_CONTEXT_LINES = 3

# 单文件 Diff 字符数上限（防止极端大文件撑爆单个 SSE 包）。
# 超过时截断 diff 文本并标记 [diff truncated]，前端仍可凭 old/new_content 自行渲染。
MAX_DIFF_CHARS_PER_FILE = 200_000


def _is_path_within(root: str, target: str) -> bool:
    """
    判断 target 解析后的绝对路径是否位于 root 目录内（含 root 本身）。

    用于防止模型返回的相对路径通过 .. 逃逸出工作区，读取任意文件。
    """
    try:
        root_resolved = Path(root).resolve()
        target_resolved = Path(target).resolve()
        # commonpath 若等于 root_resolved，说明 target 在 root 内
        common = os.path.commonpath([str(root_resolved), str(target_resolved)])
        return Path(common) == root_resolved
    except (ValueError, OSError):
        return False


def read_original_content(file_path: str, workspace_root: Optional[str]) -> str:
    """
    读取工作区内某文件的原始内容。

    Args:
        file_path: 模型返回的相对路径（相对于 workspace_root）。
        workspace_root: 工作区根目录绝对路径；为 None 时无法定位文件，返回空串。

    Returns:
        文件原始内容；文件不存在 / 路径逃逸 / 读取失败时返回空串。
    """
    if not workspace_root or not file_path:
        return ""

    # 统一反斜杠为正斜杠，避免 Windows 路径拼接问题
    rel = file_path.replace("\\", "/")
    full_path = os.path.join(workspace_root, rel)

    # 安全校验：解析后的绝对路径必须仍在 workspace_root 内
    if not _is_path_within(workspace_root, full_path):
        logger.warning(
            f"[DiffGenerator] 路径逃逸拦截: file_path={file_path}, "
            f"workspace_root={workspace_root}"
        )
        return ""

    try:
        if not os.path.isfile(full_path):
            # 新增文件场景：原文件不存在，old_content 为空
            logger.info(
                f"[DiffGenerator] 原文件不存在（视为新增）: {file_path}"
            )
            return ""
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        logger.warning(
            f"[DiffGenerator] 读取原文件失败（降级为空）: {file_path}, err={e}"
        )
        return ""


def generate_unified_diff(
    old_content: str,
    new_content: str,
    file_path: str,
    context_lines: int = DEFAULT_DIFF_CONTEXT_LINES,
) -> str:
    """
    生成 old_content -> new_content 的 Unified Diff 文本。

    Args:
        old_content: 原始文件内容。
        new_content: 模型返回的新文件完整内容。
        file_path: 文件相对路径，用于 Diff 头部的 a/path b/path 标记。
        context_lines: 变更行前后保留的上下文行数（S6 大文件优化，默认 3）。

    Returns:
        Unified Diff 字符串（含 --- / +++ / @@ 头）。若新旧内容相同返回空串。
    """
    if old_content == new_content:
        return ""

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # difflib 要求行末带换行符；若最后一行无换行符，unified_diff 会加 \ No newline 标记
    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=context_lines,
    )
    diff_text = "".join(diff_lines)

    # 极端大文件保护：Diff 文本过长时截断，避免单个 SSE 包过大
    if len(diff_text) > MAX_DIFF_CHARS_PER_FILE:
        diff_text = diff_text[:MAX_DIFF_CHARS_PER_FILE] + "\n... [diff truncated]\n"
        logger.warning(
            f"[DiffGenerator] Diff 过长已截断: file={file_path}, "
            f"原始长度={len(diff_text)}"
        )

    return diff_text


def build_diff_files(
    files_json: List[Dict],
    workspace_root: Optional[str],
    context_lines: int = DEFAULT_DIFF_CONTEXT_LINES,
) -> List[Dict]:
    """
    将模型返回的 files JSON 数组转换为含 old_content / new_content / diff 的结构。

    Args:
        files_json: 模型返回的 files 数组，每个元素含 path 和 content。
        workspace_root: 工作区根目录，用于读取原文件。
        context_lines: Diff 上下文行数。

    Returns:
        列表，每个元素为：
        {
            "path": "...",
            "old_content": "...",
            "new_content": "...",
            "diff": "..."
        }
        无效条目（缺 path/content）会被跳过。
    """
    result: List[Dict] = []
    for item in files_json:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        new_content = item.get("content")
        if not path or not isinstance(new_content, str):
            logger.warning(
                f"[DiffGenerator] 跳过无效文件条目（缺 path 或 content）: {item}"
            )
            continue

        old_content = read_original_content(path, workspace_root)
        diff_text = generate_unified_diff(
            old_content, new_content, path, context_lines=context_lines
        )

        result.append({
            "path": path,
            "old_content": old_content,
            "new_content": new_content,
            "diff": diff_text,
        })
        logger.info(
            f"[DiffGenerator] 生成 Diff: file={path}, "
            f"old_chars={len(old_content)}, new_chars={len(new_content)}, "
            f"diff_chars={len(diff_text)}"
        )

    return result


def get_workspace_root() -> Optional[str]:
    """
    获取当前工作区根目录。

    优先复用索引服务缓存的 workspace_root（首次全量索引时传入）；
    若索引尚未执行（无缓存），则回退到进程当前工作目录。

    Returns:
        工作区根目录绝对路径；无法获取时返回 None。
    """
    try:
        from app.services.code_index.index_service import get_index_service
        ws = get_index_service().workspace_root
        if ws:
            return ws
    except Exception as e:
        logger.debug(f"[DiffGenerator] 从索引服务获取 workspace_root 失败: {e}")

    # 回退：进程当前工作目录
    cwd = os.getcwd()
    logger.info(
        f"[DiffGenerator] 索引服务无 workspace_root 缓存，回退到 cwd: {cwd}"
    )
    return cwd or None
