"""
S8 第 79-80 天：沙箱状态接口

对应 Sprint_8.md「关键接口/数据结构变更（S8 新增）」中插件端设置面板需要的
GlobalState.sandbox_mode / danger_commands_blocked_count 等字段，
提供沙箱运行时的只读状态查询。
"""

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.services.sandbox_orchestrator import _docker_is_available, get_sandbox_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class SandboxStatusResponse(BaseModel):
    """GET /v1/sandbox/status 响应体"""

    # 配置中的期望模式（可能是 "auto"）
    configured_mode: str
    # 实际生效的模式（auto 降级后可能与 configured_mode 不同）
    active_mode: str
    # Docker CLI + daemon 是否可用（active_mode="docker" 时应为 True）
    docker_available: bool
    # 熔断阈值
    fuse_limit: int
    # 沙箱安全信息摘要
    safety_info: dict
    # 降级提示（非空时表示当前处于降级状态）
    degradation_warning: Optional[str] = None


@router.get("/status", response_model=SandboxStatusResponse)
async def get_sandbox_status():
    """
    查询沙箱运行时状态。

    供插件端设置面板展示「沙箱模式」「熔断阈值」等只读信息。
    该接口不执行任何变更操作——S8 Day 79-80 后端不提供运行时切换模式的
    写接口（沙箱模式在服务启动时从 config/env 读取），前端设置面板只读展示即可。
    """
    manager = get_sandbox_manager()

    # 如果还没初始化（理论上 lifespan 已调用），这里做一次安全 fallback
    orchestrator = None
    try:
        orchestrator = await manager.orchestrator()
    except Exception:
        pass

    docker_ok = await _docker_is_available()

    degradation_warning = None
    configured = settings.SANDBOX_MODE.lower()
    active = manager.chosen_mode or "host"

    if configured == "docker" and active == "host":
        degradation_warning = (
            "配置了 SANDBOX_MODE=docker 但 Docker 不可用，已自动降级到 host 模式。"
            "请安装并启动 Docker Desktop，或在 .env 中设置 SANDBOX_MODE=host。"
        )

    safety_info = {
        "default_timeout_seconds": settings.TOOL_COMMAND_DEFAULT_TIMEOUT,
        "max_timeout_seconds": settings.TOOL_COMMAND_MAX_TIMEOUT,
        "danger_patterns_count": len(settings.TOOL_DANGER_COMMAND_PATTERNS),
        "network_disabled": settings.SANDBOX_DOCKER_NETWORK_DISABLED,
        "memory_limit_mb": settings.SANDBOX_DOCKER_MEMORY_LIMIT_MB,
        "cpu_limit": settings.SANDBOX_DOCKER_CPU_LIMIT,
        "allow_host_gateway": settings.SANDBOX_DOCKER_ALLOW_HOST_GATEWAY,
    }

    return SandboxStatusResponse(
        configured_mode=configured,
        active_mode=active,
        docker_available=docker_ok,
        fuse_limit=settings.TOOL_FAIL_FUSE_LIMIT,
        safety_info=safety_info,
        degradation_warning=degradation_warning,
    )
