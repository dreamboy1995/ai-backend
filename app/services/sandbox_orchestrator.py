"""
S8 第 79-80 天：Docker 沙箱编排器 + 宿主机降级实现

对应 Sprint_8.md 第 79-80 天后端任务：
  1. 启动 Docker 容器，将用户工作区挂载到容器 /workspace。
     - 所有 read_file / write_file 都操作容器内 /workspace/... 路径。
     - 所有 run_command 都在容器内执行（使用 docker exec）。
     - 资源限制：CPU 配额、内存、默认禁止网络访问。
  2. 降级策略：若用户机器未安装 Docker，自动降级为宿主机子进程执行，
     但强制开启所有危险命令确认弹窗（多一道心理防线）。
  3. S8 风险预警覆盖：
     - Docker Windows/Mac 文件挂载 IO 慢 → :delegated 可选
     - 容器内访问宿主机 localhost → host-gateway 开关，默认关闭
     - 命令注入防护 → Docker exec / sh -c 列表参数

架构设计：
  ┌────────────────────────────────────────────────────────────┐
  │  SandboxOrchestrator（抽象基类）                           │
  │    - resolve_path(host_path) -> sandbox_path              │
  │    - execute_command(cmd, workspace_root, session_id)      │
  │    - is_available() -> bool                                │
  │    - get_mode() -> Literal["docker", "host"]               │
  └────────────────────────────────────────────────────────────┘
           ▲                  ▲
           │                  │
  ┌────────┴────────┐  ┌──────┴──────┐
  │   HostSandbox   │  │DockerSandbox│
  │  (降级实现)      │  │(真实容器)    │
  └─────────────────┘  └─────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │  SandboxManager（工厂 + 单例）                             │
  │    根据 SANDBOX_MODE 选择实现；Docker 不可用时自动降级。   │
  └────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 基础工具
# ============================================================

def _now_iso() -> str:
    """ISO 8601 UTC 时间戳"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


async def _docker_is_available() -> bool:
    """
    检测 Docker 是否可用：
      1. docker CLI 是否存在（shutil.which）
      2. docker info 是否能正常返回（后台 daemon 运行中）

    检测失败（CLI 缺失 / daemon 未启动）都视为 Docker 不可用，
    上层 SandboxManager 会自动降级到 HostSandbox。
    """
    if shutil.which("docker") is None:
        logger.warning("[Sandbox] docker CLI 未找到")
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.wait(), timeout=5.0)
        if proc.returncode == 0:
            return True
        # docker daemon 没运行：CLI 存在但命令报错
        logger.warning(
            f"[Sandbox] docker info 返回非零退出码 {proc.returncode}"
        )
        return False
    except asyncio.TimeoutError:
        logger.warning("[Sandbox] docker info 超时（daemon 可能未启动）")
        return False
    except Exception as e:
        logger.warning(f"[Sandbox] Docker 可用性检测异常: {e}")
        return False


# ============================================================
# 抽象基类
# ============================================================

class SandboxOrchestrator(ABC):
    """沙箱编排器抽象接口。HostSandbox / DockerSandbox 必须实现。"""

    @abstractmethod
    async def resolve_path(self, host_path: str) -> str:
        """
        将宿主路径转换为沙箱内路径。

        HostSandbox 原样返回；DockerSandbox 将宿主 workspace_root/xxx 
        转换为容器内 /workspace/xxx。
        """
        ...

    @abstractmethod
    async def execute_command(
        self,
        cmd: str,
        workspace_root: str,
        session_id: str,
        timeout: float = 60.0,
    ) -> Tuple[int, str, str]:
        """
        在沙箱内执行命令。

        Returns:
            (exit_code, stdout_text, stderr_text)
            沙箱不可用时返回 (-1, "", error_message)。
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """当前沙箱是否可用（DockerSandbox 需确认容器存在且运行）。"""
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """沙箱模式：'docker' 或 'host'。"""
        ...


# ============================================================
# HostSandbox（降级实现）
# ============================================================

class HostSandbox(SandboxOrchestrator):
    """
    宿主机沙箱实现（降级路径）。

    - resolve_path：原样返回宿主路径（不做转换）。
    - execute_command：委托给 CommandExecutor（宿主子进程执行）。
    - is_available：永远 True（宿主永远可用）。

    S8 风险预警："若用户机器未安装 Docker，自动降级为在宿主机子进程中执行，
    但强制开启所有危险命令的确认弹窗（让用户多一道心理防线）。"
    — 强制确认逻辑在 tool_registry / command_executor 内完成，
      HostSandbox 只是提供能力层。
    """

    @property
    def mode(self) -> str:
        return "host"

    async def resolve_path(self, host_path: str) -> str:
        # 宿主模式不做路径转换
        return host_path

    async def is_available(self) -> bool:
        return True

    async def execute_command(
        self,
        cmd: str,
        workspace_root: str,
        session_id: str,
        timeout: float = 60.0,
    ) -> Tuple[int, str, str]:
        from app.services.command_executor import get_command_executor
        executor = get_command_executor()
        return await executor.execute(
            cmd=cmd,
            workspace_root=workspace_root,
            session_id=session_id,
            timeout=timeout,
        )


# ============================================================
# DockerSandbox（真实容器实现）
# ============================================================

class DockerSandbox(SandboxOrchestrator):
    """
    Docker 沙箱实现。

    生命周期策略（S8 风险预警 + 资源控制）：
      - 每个 AgentSession 一个专属容器（name=ai-agent-{session_id} 截断）。
      - 懒创建：首次 execute_command / resolve_path 调用时才 `docker create/start`。
      - 空闲回收：idle 时间超过 SANDBOX_DOCKER_IDLE_TIMEOUT_SECONDS 则自动 stop。
      - 关闭清理：lifespan finally 时 `docker rm -f` 所有托管容器。

    资源限制（S8 Day 79-80 验收标准）：
      - --cpus  : CPU 配额（默认 0.5 核）
      - --memory: 内存上限（默认 512MB）
      - --network none : 默认禁止外网访问（用户手动开启时改为 bridge）
      - --add-host host.docker.internal:host-gateway : host-gateway 开关

    Windows/Mac IO 慢（S8 风险预警）：
      - `-v host:/workspace:cached` 是默认；
      - 用户可通过 SANDBOX_DOCKER_MOUNT_MODE=delegated 切换，
        牺牲一点一致性换挂载 IO 性能。
    """

    def __init__(self):
        # session_id -> 容器名 的映射
        self._containers: Dict[str, str] = {}
        # 最近使用时间戳（用于 idle cleanup）
        self._last_used: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _container_name(session_id: str) -> str:
        """生成容器名（Docker 限制 63 字符）"""
        short = session_id.replace("-", "")[:12]
        return f"ai-agent-{short}"

    async def _is_container_running(self, container_name: str) -> bool:
        """检查容器是否存在且处于 running 状态"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect",
                "-f", "{{.State.Running}}",
                container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            stdout = await proc.stdout.read() if proc.stdout else b""
            return proc.returncode == 0 and b"true" in stdout.lower()
        except Exception:
            return False

    async def _start_container(self, session_id: str, workspace_root: str) -> Optional[str]:
        """
        创建并启动容器。返回容器名；失败返回 None。

        使用 docker run 的等价命令拆成 create + start（更可控，便于 inspect 错误）：
          docker create --name NAME --cpus 0.5 --memory 512m \
            --network none \
            [--add-host host.docker.internal:host-gateway] \
            -v host_path:/workspace:cached \
            python:3.11-slim sleep infinity
          docker start NAME
        """
        container_name = self._container_name(session_id)

        # 1. 先看容器是否已存在（可能上次异常退出但没清理）
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}",
                container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            if proc.returncode == 0:
                stdout = await proc.stdout.read() if proc.stdout else b""
                if b"true" in stdout.lower():
                    # 已在运行
                    logger.info(f"[Sandbox] 容器已在运行: {container_name}")
                    self._containers[session_id] = container_name
                    self._last_used[session_id] = time.time()
                    return container_name
                else:
                    # 存在但已停止 → 直接 start
                    start = await asyncio.create_subprocess_exec(
                        "docker", "start", container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(start.wait(), timeout=10.0)
                    if start.returncode == 0:
                        logger.info(f"[Sandbox] 容器已启动（之前停止）: {container_name}")
                        self._containers[session_id] = container_name
                        self._last_used[session_id] = time.time()
                        return container_name
        except Exception as e:
            logger.debug(f"[Sandbox] inspect/start 现有容器失败（可能不存在）: {e}")

        # 2. 构建 docker create 参数
        args: List[str] = [
            "docker", "create",
            "--name", container_name,
            "--cpus", str(settings.SANDBOX_DOCKER_CPU_LIMIT),
            "--memory", f"{settings.SANDBOX_DOCKER_MEMORY_LIMIT_MB}m",
        ]

        # 网络策略（默认禁止外网）
        if settings.SANDBOX_DOCKER_NETWORK_DISABLED:
            args += ["--network", "none"]
        else:
            args += ["--network", "bridge"]

        # host-gateway（S8 风险预警：默认关闭，用户手动开启）
        if settings.SANDBOX_DOCKER_ALLOW_HOST_GATEWAY:
            args += ["--add-host", "host.docker.internal:host-gateway"]

        # 工作区挂载（:cached / :delegated / :consistent）
        mount_mode = settings.SANDBOX_DOCKER_MOUNT_MODE
        mount_spec = f"{workspace_root}:/workspace:{mount_mode}"
        args += ["-v", mount_spec]

        # 容器进程：sleep infinity 让容器保持运行
        args += [settings.SANDBOX_DOCKER_IMAGE, "sleep", "infinity"]

        logger.info(f"[Sandbox] 创建容器: container={container_name}, workspace={workspace_root}")

        try:
            create = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(create.wait(), timeout=30.0)
            if create.returncode != 0:
                stderr = (await create.stderr.read()).decode("utf-8", errors="replace") if create.stderr else ""
                logger.error(f"[Sandbox] docker create 失败: {stderr.strip()}")
                return None

            # start
            start = await asyncio.create_subprocess_exec(
                "docker", "start", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(start.wait(), timeout=10.0)
            if start.returncode != 0:
                stderr = (await start.stderr.read()).decode("utf-8", errors="replace") if start.stderr else ""
                logger.error(f"[Sandbox] docker start 失败: {stderr.strip()}")
                # 清理残留容器
                await self._remove_container(container_name)
                return None

            self._containers[session_id] = container_name
            self._last_used[session_id] = time.time()
            logger.info(f"[Sandbox] 容器启动成功: {container_name}")
            return container_name

        except asyncio.TimeoutError:
            logger.error(f"[Sandbox] 创建容器超时: {container_name}")
            return None
        except Exception as e:
            logger.error(f"[Sandbox] 创建容器异常: {e}", exc_info=True)
            return None

    async def _remove_container(self, container_name: str) -> None:
        """强制删除容器（不管 running 与否）"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
            if proc.returncode == 0:
                logger.info(f"[Sandbox] 容器已清理: {container_name}")
        except Exception as e:
            logger.debug(f"[Sandbox] 容器清理失败: {container_name}, err={e}")

    async def _stop_idle_containers(self) -> None:
        """
        定时任务：清理超过 SANDBOX_DOCKER_IDLE_TIMEOUT_SECONDS 未使用的容器。

        避免用户长时间不用但容器一直在后台跑（浪费 CPU/内存）。
        """
        now = time.time()
        idle_threshold = settings.SANDBOX_DOCKER_IDLE_TIMEOUT_SECONDS
        to_remove: List[str] = []
        async with self._lock:
            for session_id, container_name in list(self._containers.items()):
                last = self._last_used.get(session_id, 0)
                if now - last > idle_threshold:
                    logger.info(
                        f"[Sandbox] 容器空闲超时，停止: session={session_id}, "
                        f"container={container_name}, idle={int(now - last)}s"
                    )
                    to_remove.append(container_name)
                    self._containers.pop(session_id, None)
                    self._last_used.pop(session_id, None)

        # 锁外执行 stop（避免持锁时跑外部命令）
        for cname in to_remove:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker", "stop", cname,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.wait(), timeout=15.0)
            except Exception as e:
                logger.debug(f"[Sandbox] docker stop 失败 {cname}: {e}")

    async def _idle_cleanup_loop(self) -> None:
        """后台任务：每 5 分钟扫描一次空闲容器"""
        interval = min(300, settings.SANDBOX_DOCKER_IDLE_TIMEOUT_SECONDS // 2)
        while True:
            try:
                await asyncio.sleep(interval)
                await self._stop_idle_containers()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[Sandbox] 空闲容器清理循环异常: {e}")

    async def start_cleanup_loop(self) -> None:
        """服务启动时调用，启动后台清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._idle_cleanup_loop())
            logger.info("[Sandbox] Docker 空闲容器清理任务已启动")

    async def cleanup_all(self) -> None:
        """服务关闭时调用，强制删除所有托管容器"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if settings.SANDBOX_DOCKER_AUTO_CLEANUP:
            async with self._lock:
                all_containers = list(self._containers.values())
                self._containers.clear()
                self._last_used.clear()
            for cname in all_containers:
                await self._remove_container(cname)
            logger.info(f"[Sandbox] 服务关闭，已清理 {len(all_containers)} 个容器")
        else:
            logger.info("[Sandbox] 服务关闭，保留容器（SANDBOX_DOCKER_AUTO_CLEANUP=False）")

    # ------------------------------------------------------------------
    # SandboxOrchestrator 接口实现
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "docker"

    async def resolve_path(self, host_path: str) -> str:
        """
        宿主路径 → 容器内 /workspace 路径。

        规则：
          - 已在 /workspace 下：原样返回
          - 绝对路径：假设是挂载进来的，取 basename 或保留
          - 相对路径：直接拼到 /workspace 下
        """
        if host_path.startswith("/workspace/"):
            return host_path
        # Windows 盘符路径（D:\xxx\yyy）也统一处理
        normalized = host_path.replace("\\", "/")
        # 如果以盘符开头（如 D:/project/abc.py），取 / 之后的部分
        if len(normalized) > 2 and normalized[1] == ":":
            # Windows 盘符：D:/path/to/file → path/to/file
            normalized = normalized[2:].lstrip("/")
        if normalized.startswith("/"):
            # 绝对 Unix 路径：假设是宿主根目录某文件，无法安全映射，直接返回
            return normalized
        return f"/workspace/{normalized}"

    async def is_available(self) -> bool:
        """
        Docker 本身是否可用（不关心具体容器）。
        容器级可用性由 execute_command 里的懒启动机制处理。
        """
        return await _docker_is_available()

    async def execute_command(
        self,
        cmd: str,
        workspace_root: str,
        session_id: str,
        timeout: float = 60.0,
    ) -> Tuple[int, str, str]:
        """
        在容器内执行命令：
          docker exec -w /workspace <container> sh -c '<cmd>'

        关键点：
          - 用 sh -c 包装：支持 &&、管道、重定向等 shell 特性。
          - -w /workspace：cwd 固定在容器内 /workspace（宿主 workspace_root 已挂载）。
          - stdout/stderr 分别 PIPE，最后合并返回。
        """
        container_name = self._containers.get(session_id)
        if container_name is None:
            async with self._lock:
                # 双重检查（避免并发创建）
                container_name = self._containers.get(session_id)
                if container_name is None:
                    container_name = await self._start_container(session_id, workspace_root)
                    if container_name is None:
                        return -1, "", (
                            "Docker 容器启动失败，沙箱不可用。"
                            "请检查 Docker daemon 是否运行，或降级到 host 模式。"
                        )

        # 更新最近使用时间
        async with self._lock:
            self._last_used[session_id] = time.time()

        # docker exec 命令
        exec_args = [
            "docker", "exec",
            "-w", "/workspace",
            container_name,
            "sh", "-c", cmd,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *exec_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                # 超时：强杀 docker exec 进程（容器进程组还在，需要 kill）
                try:
                    proc.kill()
                except Exception:
                    pass
                return -1, "", f"命令超时（{timeout}s），Docker exec 已终止"

            stdout = (await proc.stdout.read()).decode("utf-8", errors="replace") if proc.stdout else ""
            stderr = (await proc.stderr.read()).decode("utf-8", errors="replace") if proc.stderr else ""
            return proc.returncode or 0, stdout, stderr

        except FileNotFoundError:
            return -1, "", "docker CLI 未找到"
        except Exception as e:
            logger.error(f"[Sandbox] docker exec 异常: {e}")
            return -1, "", f"Docker exec 异常: {e}"


# ============================================================
# SandboxManager（工厂 + 单例）
# ============================================================

class SandboxManager:
    """
    沙箱编排器工厂（单例）。

    根据 SANDBOX_MODE 选择具体实现：
      - "docker"  → DockerSandbox（Docker 不可用时自动降级 HostSandbox）
      - "host"    → HostSandbox
      - "auto"    → 优先检测 Docker，可用则 DockerSandbox，否则 HostSandbox

    降级检测启动时做一次，后续不会再自动切回去（避免运行中切换导致路径错乱）。
    """

    def __init__(self):
        self._orchestrator: Optional[SandboxOrchestrator] = None
        self._chosen_mode: Optional[str] = None

    async def initialize(self) -> SandboxOrchestrator:
        """
        初始化并返回当前编排器。

        S8 Day 79-80 降级策略在这里集中处理：
          - SANDBOX_MODE="docker" + Docker 不可用 → HostSandbox + warning 日志
          - SANDBOX_MODE="auto"    + Docker 不可用 → HostSandbox
        """
        if self._orchestrator is not None:
            return self._orchestrator

        mode = settings.SANDBOX_MODE.lower()
        docker_available = await _docker_is_available()

        if mode == "host":
            self._orchestrator = HostSandbox()
            self._chosen_mode = "host"
            logger.info("[Sandbox] 已配置为 host 模式（SANDBOX_MODE=host）")

        elif mode in ("docker", "auto"):
            if docker_available:
                ds = DockerSandbox()
                await ds.start_cleanup_loop()
                self._orchestrator = ds
                self._chosen_mode = "docker"
                logger.info("[Sandbox] Docker 沙箱已就绪")
            else:
                self._orchestrator = HostSandbox()
                self._chosen_mode = "host"
                if mode == "docker":
                    # 用户明确要求 docker 但不可用 → 降级 + warning
                    logger.warning(
                        "[Sandbox] SANDBOX_MODE=docker 但 Docker 不可用，"
                        "自动降级到 host 模式。请安装并启动 Docker Desktop，"
                        "或手动设置 SANDBOX_MODE=host 消除警告。"
                    )
                else:
                    logger.info("[Sandbox] Docker 不可用，auto 模式降级到 host")

        else:
            # 非法模式 → 降级 host
            logger.warning(
                f"[Sandbox] 未知 SANDBOX_MODE={mode}，降级到 host"
            )
            self._orchestrator = HostSandbox()
            self._chosen_mode = "host"

        return self._orchestrator

    @property
    def chosen_mode(self) -> str:
        """实际生效的模式（可能因降级与 SANDBOX_MODE 不同）"""
        return self._chosen_mode or "host"

    async def orchestrator(self) -> SandboxOrchestrator:
        if self._orchestrator is None:
            return await self.initialize()
        return self._orchestrator

    async def shutdown(self) -> None:
        """服务关闭时清理资源"""
        orchestrator = self._orchestrator
        if orchestrator is not None and isinstance(orchestrator, DockerSandbox):
            await orchestrator.cleanup_all()


# 全局单例
_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager
