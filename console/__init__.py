"""
Crotalus Console Module

提供统一命令系统：命令注册（SERVER/BOTH/CLIENT 执行边 + 0-4 权限等级）、
简易权限系统、统一命令线程（控制台与客户端命令都投递到这里调度执行）、
控制台界面与自动补全。
"""

from console.command import (
    CONSOLE_SENDER,
    Command,
    CommandContext,
    CommandManager,
    CommandSender,
    CommandSide,
    command,
    cmd_manager,
    register_command,
    register_commands,
)
from console.permissions import PermissionManager, permission_manager
from console.dispatcher import CommandDispatcher, command_dispatcher
from console.completer import ConsoleCompleter
from console.interface import ConsoleInterface, SimpleConsoleInterface
from console.commands import register_builtin_commands

__all__ = [
    "CONSOLE_SENDER",
    "Command",
    "CommandContext",
    "CommandManager",
    "CommandSender",
    "CommandSide",
    "CommandDispatcher",
    "PermissionManager",
    "command",
    "cmd_manager",
    "command_dispatcher",
    "permission_manager",
    "register_command",
    "register_commands",
    "ConsoleCompleter",
    "ConsoleInterface",
    "SimpleConsoleInterface",
    "register_builtin_commands",
]
