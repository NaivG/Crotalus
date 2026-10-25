"""
In-game Command Sender (客户端命令发送者与注册)

客户端命令与控制台命令是同一套系统：都注册进 console/command.py 的
CommandManager（统一注册接口 register_command），都投递到统一命令线程
（console/dispatcher.py）按提交顺序调度执行。

- 暴露面由 command.side 决定：CLIENT 仅客户端、BOTH 两端、SERVER 仅
  控制台（玩家端视为未知命令）。
- 权限由 command.level 与 console/permissions.py 的 ops 表共同决定：
  玩家等级来自 config.toml [permissions].operators（名字 -> 等级），
  未列出的玩家为 default-level（默认 0）；控制台恒为 4（owner）。

本模块保留的职责只有两件：

- PlayerCommandSender：聊天命令发送者。回复经 System Chat 包送回玩家，
  线程归属在这里处理 —— 命令跑在命令线程的 loop 上，而连接的 writer
  归网络线程所有，所以发包用 run_coroutine_threadsafe 投回网络循环。
- dispatch_from_client：Chat Command 包（network/client_handler.py）的
  统一入口，把命令行投递到命令线程。
"""

from __future__ import annotations

import asyncio
from typing import Optional, Union

from loguru import logger

from console.command import (
    Command,
    CommandContext,
    CommandSender,
    CommandSide,
    Component,
    error,
    register_command,
)
from console.permissions import permission_manager


class PlayerCommandSender(CommandSender):
    """聊天命令发送者：包一个 ClientHandler。

    连接（writer）归网络线程所有。命令在命令线程上执行，因此一切
    碰 writer 的动作（回复、set_gamemode 这类发包方法）都要投回网络
    循环 —— reply() 用 run_coroutine_threadsafe 即发即忘（对同一网络
    循环按提交顺序排队，回复顺序与命令里调用 reply 的顺序一致）。
    """

    def __init__(self, handler):
        self.handler = handler
        try:
            # ClientHandler 在网络循环上构造（core/server._handle_client）
            self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # -- CommandSender 接口 --------------------------------------------------

    @property
    def name(self) -> str:
        player = getattr(self.handler, "player", None)
        name = getattr(player, "name", None)
        if name:
            return name
        return str(getattr(self.handler, "remote_address", "?"))

    @property
    def is_console(self) -> bool:
        return False

    @property
    def permission_level(self) -> int:
        return permission_manager.get_player_level(self.name)

    def reply(self, message: Component) -> None:
        """把一条回复送回玩家（System Chat，跨线程投递到网络循环）。"""
        from network.packets import CSystemChatMessage

        self._fire_and_forget(
            self.handler.send_packet(CSystemChatMessage(message)),
            what="reply",
        )

    # -- 跨线程工具 -----------------------------------------------------------

    def _fire_and_forget(self, coro, *, what: str) -> None:
        """把一个协程投到网络循环上执行，不等待完成。"""
        future = self._submit_future(coro, what=what)
        if future is not None:
            future.add_done_callback(self._log_send_error)

    async def submit(self, coro):
        """在发送者所属的网络循环上执行一个协程，并等待其完成。

        命令里需要"发包并确认成功"时用这个（如 /gamemode 的
        set_gamemode）；纯回复用 reply() 即可。
        """
        future = self._submit_future(coro, what="submit")
        if future is None:
            raise RuntimeError("network loop unavailable for command sender")
        return await asyncio.wrap_future(future)

    def _submit_future(self, coro, *, what: str):
        loop = self._loop
        if loop is None or loop.is_closed():
            coro.close()
            logger.debug(f"Cannot {what} for {self.name}: network loop is gone")
            return None
        return asyncio.run_coroutine_threadsafe(coro, loop)

    @staticmethod
    def _log_send_error(future: asyncio.Future) -> None:
        try:
            future.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # 玩家可能在命令执行途中退出：回复失败只记日志，不能炸掉命令线程
            logger.debug(f"Command reply failed: {e}")


def dispatch_from_client(handler, command_line: str) -> bool:
    """客户端命令入口：把一行命令投递到统一命令线程。

    由 network/client_handler.py 在收到 Chat Command 包时调用（网络
    线程）。命令随后在命令线程上执行，反馈经 PlayerCommandSender
    回给玩家。返回是否成功交给了调度方。
    """
    from console.dispatcher import command_dispatcher

    sender = PlayerCommandSender(handler)
    if command_dispatcher.post(command_line, sender):
        return True

    # 命令线程未运行（工具/测试环境）：退化为在当前（网络）循环上直接
    # 执行，保证命令系统在任何环境都可独立工作。
    from console.command import cmd_manager

    logger.debug("Command dispatcher not running; executing client command inline")
    task = asyncio.ensure_future(cmd_manager.execute_command(command_line, sender))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return True


#: 持有回退路径的后台任务引用，防止执行中被垃圾回收
_BACKGROUND_TASKS: set = set()


# ---------------- built-in client commands ----------------


# Vanilla brigadier accepts these spellings for /gamemode's mode argument;
# the digits are the wire values and the letters are the abbreviations.
GAMEMODE_ALIASES: dict[str, int] = {
    "survival": 0, "s": 0, "0": 0,
    "creative": 1, "c": 1, "1": 1,
    "adventure": 2, "a": 2, "2": 2,
    "spectator": 3, "sp": 3, "3": 3,
}
GAMEMODE_NAMES = {0: "survival", 1: "creative", 2: "adventure", 3: "spectator"}


class GamemodeCommand(Command):
    """Switch the sender's own gamemode.

    side=CLIENT：这条命令只对"有自己游戏模式"的发送者有意义，控制台
    没有玩家身份，不暴露。level=2：原版 /gamemode 需要 gamemaster 级。
    """

    name = "gamemode"
    description = "Change your game mode"
    usage = "/gamemode <survival|creative|adventure|spectator>"
    aliases = ["gm"]
    side = CommandSide.CLIENT
    level = 2

    async def execute(self, ctx: CommandContext) -> bool:
        if not ctx.args:
            ctx.reply(error(f"Usage: {self.usage}"))
            return False

        mode = GAMEMODE_ALIASES.get(ctx.args[0].lower())
        if mode is None:
            ctx.reply(error(f"Unknown game mode: {ctx.args[0]}"))
            return False

        handler = ctx.sender.handler
        if handler.player.game_mode.value == mode:
            # Vanilla no-ops a same-mode switch silently; inventing feedback
            # here would diverge from what players expect the command to do.
            return True

        # set_gamemode 会向 writer 发三个包：从命令线程投回网络循环执行
        await ctx.sender.submit(handler.set_gamemode(mode))
        ctx.reply({
            "translate": "commands.gamemode.success.self",
            "with": [{"translate": f"gameMode.{GAMEMODE_NAMES[mode]}"}],
        })
        return True


def register_player_commands() -> None:
    """把客户端命令注册进统一命令系统（side/level 决定暴露面）。"""
    register_command(GamemodeCommand())
    logger.debug("Player commands registered into the unified command manager")
