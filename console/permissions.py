"""
Simple Permission System

简易权限系统，仿 Minecraft 原版权限等级体系（0-4，OP 即 4 级）：

    等级  vanilla 名称      含义
    ----  ----------------  --------------------------------
    0     player            所有玩家（默认）
    1     moderator         绕过出生点保护
    2     gamemaster        游戏管理命令（/gamemode 等）、命令方块
    3     admin             玩家管理命令（/kick、/ban 等）
    4     owner             服务器所有者 —— 控制台（CONSOLE）恒为此级

命令用 ``command.level`` 声明所需的最低等级，发送者等级不足即拒绝
（见 ``console.command.CommandManager.execute_command``）。
发送者等级的来源：

- ``ConsoleSender`` 恒为 4：控制台就是服务器的最高权限。
- 玩家来自 config.toml 的 ``[permissions]`` 段::

      [permissions]
      default-level = 0                  # 未列出的玩家的等级
      operators = { "Akarin" = 4 }       # 玩家名 -> 权限等级

每次检查都实时读取 config（而不做缓存），因此控制台 ``/reload`` 之后
等级变更立即生效，无需重启。
"""

from __future__ import annotations

from loguru import logger

#: 权限等级上限（原版权限体系最高为 4，即控制台/所有者）。
MAX_LEVEL = 4

#: 各等级的 vanilla 名称，用于帮助输出与错误信息。
PERMISSION_NAMES: dict[int, str] = {
    0: "player",
    1: "moderator",
    2: "gamemaster",
    3: "admin",
    4: "owner",
}


def level_name(level: int) -> str:
    """等级的可读名称（未知等级按未知处理，不猜）。"""
    return PERMISSION_NAMES.get(level, "unknown")


def _config():
    """config 单例（懒加载）。

    不在模块顶部导入：``from core.config import config`` 会执行
    ``core/__init__``，连带 core.threads -> psutil。保持 console.command
    不拖起整个 core 包（console/commands.py 里懒导入 core.server 的
    理由相同）。命令执行期 core 早已加载完毕，缓存导入零开销。
    """
    from core.config import config

    return config


def clamp_level(value, default: int = 0) -> int:
    """把配置里的等级收敛到 0..MAX_LEVEL。

    配置写错（字符串、负数、超过 4）不应让命令系统瘫痪：收敛到最近的
    合法值并警告，而不是抛异常。
    """
    try:
        level = int(value)
    except (TypeError, ValueError):
        logger.warning(f"Permission level {value!r} is not a number; using {default}")
        return default
    if level < 0:
        logger.warning(f"Permission level {level} < 0; clamped to 0")
        return 0
    if level > MAX_LEVEL:
        logger.warning(f"Permission level {level} > {MAX_LEVEL}; clamped to {MAX_LEVEL}")
        return MAX_LEVEL
    return level


class PermissionManager:
    """简易权限系统。

    只回答一个问题：这个发送者能不能执行需要 X 级的命令。等级来源见
    模块 docstring；OP 名单在 config 里，不引入独立的 ops.json。
    """

    def get_player_level(self, name: str) -> int:
        """一个玩家的权限等级（按玩家名查 ops 表）。"""
        config = _config()
        operators = config.get("permissions.operators", {}) or {}
        if name in operators:
            return clamp_level(operators[name])
        return clamp_level(config.get("permissions.default-level", 0))

    def sender_level(self, sender) -> int:
        """发送者的权限等级：控制台恒为最高级，玩家按 ops 表。"""
        if getattr(sender, "is_console", False):
            return MAX_LEVEL
        return self.get_player_level(getattr(sender, "name", "?"))

    def check(self, sender, required_level: int) -> bool:
        """发送者等级是否达到命令要求的最低等级。"""
        return self.sender_level(sender) >= clamp_level(required_level)


#: 全局单例：CommandManager 的派发检查与 PlayerCommandSender 的等级上报
#: 都从这里读，保证同一份事实。
permission_manager = PermissionManager()
