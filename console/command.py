"""
Unified Command System

控制台命令与客户端命令的统一模型。此前两端各有一套镜像实现
（console/command.py 与 player/commands.py），现在合并为一套：

- 命令用 ``side`` 声明执行边，``side`` 决定这条命令向哪一端暴露：
    SERVER  仅控制台（服务端）
    BOTH    控制台与客户端都可用
    CLIENT  仅客户端（聊天栏）
- 命令用 ``level`` 声明所需权限等级（仿原版 OP 体系，0-4），等级不足即
  拒绝；等级来源见 console/permissions.py。
- 所有命令注册到唯一的 ``CommandManager``，由统一命令线程
  （console/dispatcher.py）按提交顺序调度执行 —— 控制台与客户端的输入
  都只是向那个线程投递一行文本。
- 命令的输出通过 ``ctx.reply()`` 回到发送者：控制台是 print，玩家是
  System Chat 包。SERVER 命令只会被控制台执行，因此那些命令直接
  ``print()`` 也是回到控制台；BOTH/CLIENT 命令必须走 ``ctx.reply()``。
"""

from __future__ import annotations

import enum
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from loguru import logger

from console.permissions import level_name, permission_manager

# 一条回复：纯文本，或直接按 NBT 写出的组件字典（translate 等交给客户端本地化）。
Component = Union[str, dict]

ERROR_COLOR = "red"


def error(component: Component) -> Component:
    """把一条回复标记为错误（原版渲染为红色）。"""
    if isinstance(component, dict):
        return {**component, "color": ERROR_COLOR}
    return {"text": str(component), "color": ERROR_COLOR}


class CommandSide(enum.Enum):
    """命令执行边：决定命令向哪一端暴露。

    执行边负责分类客户端、服务端（即 CONSOLE）各自暴露的命令：

    - ``SERVER``：仅控制台可用。玩家端表现为"未知命令"（不泄露存在性，
      与原版一致）。
    - ``BOTH``：两端都注册、都可用（如 /help、/list）。
    - ``CLIENT``：仅客户端可用。控制台执行时给出明确提示（操作者需要
      知道为什么命令打不了，而不是收到一句"未知命令"）。
    """

    SERVER = "server"
    BOTH = "both"
    CLIENT = "client"

    @property
    def console_visible(self) -> bool:
        """控制台（CONSOLE）能否看到/执行此命令。"""
        return self in (CommandSide.SERVER, CommandSide.BOTH)

    @property
    def client_visible(self) -> bool:
        """客户端（玩家）能否看到/执行此命令。"""
        return self in (CommandSide.CLIENT, CommandSide.BOTH)

    @property
    def label(self) -> str:
        """可读名称，用于 /help 的详细输出。"""
        return {
            CommandSide.SERVER: "SERVER (仅控制台)",
            CommandSide.BOTH: "BOTH (控制台+客户端)",
            CommandSide.CLIENT: "CLIENT (仅客户端)",
        }[self]


class CommandSender(ABC):
    """命令发送者抽象。

    命令不关心对面是谁，只通过 reply() 把输出发回去，通过
    permission_level 让权限系统做检查。控制台与玩家各有一个实现。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """发送者名称（用于日志与权限查询）。"""

    @property
    @abstractmethod
    def permission_level(self) -> int:
        """发送者的权限等级（0-4）。"""

    @property
    def is_console(self) -> bool:
        """是否为控制台发送者（SERVER 命令只对它开放）。"""
        return False

    @abstractmethod
    def reply(self, message: Component) -> None:
        """把一条回复送回发送者。控制台: print；玩家: System Chat。"""


class ConsoleSender(CommandSender):
    """控制台发送者（SERVER 端，即 CONSOLE）。

    控制台是服务器的所有者：权限恒为最高级 4，输出直接 print 到
    服务器 stdout。
    """

    @property
    def name(self) -> str:
        return "Server"

    @property
    def permission_level(self) -> int:
        return 4

    @property
    def is_console(self) -> bool:
        return True

    def reply(self, message: Component) -> None:
        # 组件字典（translate/带色）只在玩家回复里出现；真落到控制台时
        # 打印其文本内容，而不是 Python repr。
        if isinstance(message, dict):
            text = message.get("text")
            if text is None and "translate" in message:
                text = f"<{message['translate']}>"
            print(text if text is not None else str(message))
        else:
            print(message)


#: 全局控制台发送者。命令线程派发时，未指明发送者的输入都来自它。
CONSOLE_SENDER = ConsoleSender()


@dataclass
class CommandContext:
    """命令执行上下文。

    sender 为 None 时的 reply() 退回到控制台 —— 工具脚本直接构造
    ``CommandContext(args=[...])`` 执行命令时输出仍然有地方去。
    """

    args: list[str] = field(default_factory=list)
    raw_args: str = ""
    sender: Optional[CommandSender] = None

    def reply(self, message: Component) -> None:
        """把一行输出发回给发送者（控制台: print；玩家: System Chat）。"""
        (self.sender or CONSOLE_SENDER).reply(message)


class Command(ABC):
    """命令基类。

    子类/装饰器需要设置的预设属性：

    - name        命令名（小写）
    - description 命令描述（/help 展示）
    - usage       用法（/help <命令> 展示）
    - aliases     别名列表
    - side        执行边（SERVER/BOTH/CLIENT），决定向哪端暴露
    - level       所需权限等级（仿原版 OP 体系，0-4），等级不足即拒绝
    """

    name: str = ""
    description: str = ""
    usage: str = ""
    aliases: list[str] = []
    side: CommandSide = CommandSide.SERVER
    level: int = 0

    def __init__(self):
        self._subcommands: dict[str, Command] = {}

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> bool:
        """
        执行命令

        Args:
            ctx: 命令上下文

        Returns:
            True表示命令执行成功，False表示失败
        """
        pass

    def add_subcommand(self, subcommand: Command) -> None:
        """添加子命令"""
        self._subcommands[subcommand.name] = subcommand
        for alias in subcommand.aliases:
            self._subcommands[alias] = subcommand

    def get_subcommand(self, name: str) -> Optional[Command]:
        """获取子命令"""
        return self._subcommands.get(name)

    def tab_complete(self, args: list[str]) -> list[str]:
        """参数补全（默认无参数建议；子类可覆盖）。"""
        return []

    def visible_to(self, sender: CommandSender) -> bool:
        """这条命令对发送者是否可见（执行边 + 权限等级都满足）。"""
        if sender.is_console:
            return self.side.console_visible
        return (
            self.side.client_visible
            and permission_manager.sender_level(sender) >= self.level
        )


class CommandManager:
    """命令管理器：唯一注册表 + 派发器。

    控制台命令与客户端命令都注册在这里，按执行边分类暴露；执行统一由
    命令线程（console/dispatcher.py）调起 ``execute_command``。
    """

    _instance: Optional[CommandManager] = None

    def __init__(self):
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}  # alias -> command name

    @classmethod
    def get_instance(cls) -> CommandManager:
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = CommandManager()
        return cls._instance

    def register(self, command: Command) -> None:
        """注册命令（统一注册接口的底层；一般用模块级 register_command）。"""
        if not command.name:
            raise ValueError("Command must have a name")

        if command.side not in CommandSide:
            raise ValueError(f"Command '{command.name}' has invalid side: {command.side!r}")
        if not isinstance(command.level, int) or not 0 <= command.level <= 4:
            logger.warning(
                f"Command '{command.name}' declares level {command.level!r}; "
                f"clamping to 0..4"
            )
            command.level = max(0, min(int(command.level), 4))

        if command.name in self._commands:
            logger.warning(f"Command '{command.name}' is already registered, overwriting")

        self._commands[command.name] = command
        self._aliases[command.name] = command.name

        # 注册别名
        for alias in command.aliases:
            if alias in self._aliases:
                logger.warning(f"Alias '{alias}' for '{command.name}' conflicts with existing command")
            self._aliases[alias] = command.name

        logger.debug(f"Registered command: {command.name} (side={command.side.value}, level={command.level})")

    def unregister(self, name: str) -> bool:
        """取消注册命令"""
        if name not in self._commands:
            return False

        command = self._commands[name]
        del self._commands[name]
        del self._aliases[name]

        for alias in command.aliases:
            self._aliases.pop(alias, None)

        return True

    def get_command(self, name: str) -> Optional[Command]:
        """获取命令"""
        real_name = self._aliases.get(name)
        if real_name:
            return self._commands.get(real_name)
        return None

    def get_all_commands(self) -> dict[str, Command]:
        """获取所有命令"""
        return self._commands.copy()

    def has_command(self, name: str) -> bool:
        """检查命令是否存在"""
        return name in self._aliases

    def get_command_names(self, console_visible: bool = True) -> list[str]:
        """获取所有命令名称（包括别名）。

        console_visible=True 时只返回控制台可见（SERVER/BOTH）执行边的
        命令 —— 补全器按它取词，客户端命令名不漏进控制台补全。
        """
        names = []
        for alias, real_name in self._aliases.items():
            command = self._commands.get(real_name)
            if command is None:
                continue
            if console_visible and not command.side.console_visible:
                continue
            names.append(alias)
        return names

    async def execute_command(
        self,
        input_str: str,
        sender: Optional[CommandSender] = None
    ) -> bool:
        """
        执行一行命令（由统一命令线程调用）。

        派发顺序：解析 -> 查命令 -> 执行边检查 -> 权限等级检查 -> 执行。

        Args:
            input_str: 命令字符串
            sender: 发送者（None 视为控制台）

        Returns:
            True表示命令执行成功，False表示失败
        """
        sender = sender or CONSOLE_SENDER

        if not input_str or not input_str.strip():
            return False

        # 原版惯例：控制台与聊天栏都接受带或不带前导斜杠的输入。
        # 统一入口在这里剥掉，两端调用方都不必再各自处理。
        line = input_str.strip().lstrip("/")
        if not line:
            return False

        try:
            parts = shlex.split(line)
        except ValueError as e:
            sender.reply(error(f"Invalid command syntax: {e}"))
            return False

        if not parts:
            return False

        command_name = parts[0].lower()
        command = self.get_command(command_name)

        if command is None:
            logger.debug(f"Unknown command from {sender.name}: {command_name}")
            sender.reply(error({"translate": "command.unknown.command"}))
            return False

        # 执行边检查：side 决定命令向哪端暴露
        if command.side is CommandSide.SERVER and not sender.is_console:
            # 玩家敲了服务端命令：按原版惯例当作不存在，不泄露命令列表
            logger.debug(f"Player {sender.name} tried server-side command /{command_name}")
            sender.reply(error({"translate": "command.unknown.command"}))
            return False
        if command.side is CommandSide.CLIENT and sender.is_console:
            # 控制台敲了客户端命令：操作者需要明确原因，不装作未知
            sender.reply(error(
                f"/{command_name} 仅可在客户端执行 (client-only command)"
            ))
            return False

        # 权限等级检查（简易权限系统，仿原版 OP 等级）
        if not permission_manager.check(sender, command.level):
            logger.debug(
                f"Permission denied: {sender.name} (level "
                f"{permission_manager.sender_level(sender)}) tried /{command_name} "
                f"(needs level {command.level})"
            )
            sender.reply(error(
                f"权限不足: /{command_name} 需要等级 {command.level} "
                f"({level_name(command.level)})，你的等级为 "
                f"{permission_manager.sender_level(sender)}"
            ))
            return False

        ctx = CommandContext(
            args=parts[1:] if len(parts) > 1 else [],
            raw_args=" ".join(parts[1:]) if len(parts) > 1 else "",
            sender=sender,
        )

        try:
            return await command.execute(ctx)
        except Exception as e:
            logger.exception(f"Error executing command '{command_name}': {e}")
            sender.reply(error(f"An error occurred while executing /{command_name}"))
            return False

    def tab_complete(self, input_str: str, sender: Optional[CommandSender] = None) -> list[str]:
        """
        命令补全（控制台用；只补全对发送者可见的命令）。

        Args:
            input_str: 当前输入
            sender: 发送者（None 视为控制台）

        Returns:
            可能的补全选项列表
        """
        sender = sender or CONSOLE_SENDER

        if not input_str or not input_str.strip():
            return [name for name in self.get_command_names()
                    if (cmd := self.get_command(name)) and cmd.visible_to(sender)]

        parts = input_str.rstrip().split()
        if not parts:
            return [name for name in self.get_command_names()
                    if (cmd := self.get_command(name)) and cmd.visible_to(sender)]

        if len(parts) == 1 and not input_str.endswith(" "):
            # 补全命令名
            prefix = parts[0].lower()
            return [name for name in self.get_command_names()
                    if name.startswith(prefix)
                    and (cmd := self.get_command(name)) and cmd.visible_to(sender)]

        # 补全命令参数
        command_name = parts[0].lower()
        command = self.get_command(command_name)
        if command and command.visible_to(sender):
            return command.tab_complete(parts[1:])
        return []


def command(
    name: str,
    description: str = "",
    usage: str = "",
    aliases: Optional[list[str]] = None,
    side: CommandSide = CommandSide.SERVER,
    level: int = 0,
) -> Callable:
    """
    命令装饰器

    用法:
        @command("hello", description="Say hello", side=CommandSide.BOTH, level=2)
        async def hello_cmd(ctx: CommandContext) -> bool:
            ctx.reply("Hello, world!")
            return True
    """
    def decorator(func: Callable) -> type[Command]:
        class DecoratedCommand(Command):
            name = name
            description = description
            usage = usage or f"/{name}"
            aliases = aliases or []
            side = side
            level = level

            async def execute(self, ctx: CommandContext) -> bool:
                return await func(ctx)

        return DecoratedCommand()

    return decorator


def register_command(command_obj: Command) -> Command:
    """统一命令注册接口。

    控制台命令（SERVER/BOTH）与客户端命令（CLIENT/BOTH）都从这里注册进
    唯一的 CommandManager；注册后由命令线程统一调度。返回传入的命令，
    方便在注册处就地 ``return register_command(SomeCommand())``。
    """
    cmd_manager.register(command_obj)
    return command_obj


def register_commands(*commands: Command) -> None:
    """批量注册命令。"""
    for command_obj in commands:
        register_command(command_obj)


# 全局命令管理器实例
cmd_manager = CommandManager.get_instance()
