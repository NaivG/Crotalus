"""
Console Completer

使用 prompt_toolkit 实现命令自动补全。
"""

from __future__ import annotations

from typing import Optional
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from console.command import CommandManager, CommandSide


class ConsoleCompleter(Completer):
    """控制台命令补全器

    每次补全都实时从 CommandManager 取词（命令可能在控制台启动后才注册，
    例如客户端命令），并且只补全控制台可见执行边（SERVER/BOTH）的命令 ——
    CLIENT 命令不属于服务端（CONSOLE）暴露面，不能出现在补全里。
    """

    def __init__(self, command_manager: Optional[CommandManager] = None):
        self.command_manager = command_manager or CommandManager.get_instance()

    def _visible_commands(self):
        """当前已注册、控制台可见的 (名称, 命令) 列表（含别名）。"""
        pairs = []
        for name in self.command_manager.get_command_names(console_visible=True):
            command = self.command_manager.get_command(name)
            if command is not None:
                pairs.append((name, command))
        return sorted(pairs)

    def get_completions(
        self,
        document,
        complete_event
    ):
        """获取补全项"""
        text = document.text
        visible = self._visible_commands()

        if not text or not text.strip():
            # 无输入时，列出所有可见命令
            for cmd_name, command in visible:
                display = f"{cmd_name}"
                if command.aliases:
                    display += f" ({', '.join(command.aliases)})"
                yield Completion(
                    cmd_name,
                    start_position=0,
                    display=display,
                    display_meta=command.description or "Command"
                )
            return

        # 对正在输入的最后一个词做前缀补全（忽略大小写）
        last_token = text.rsplit(" ", 1)[-1]
        prefix = last_token.lower()
        for cmd_name, command in visible:
            if cmd_name.lower().startswith(prefix) and cmd_name != last_token:
                display = cmd_name
                if command.aliases:
                    display += f" ({', '.join(command.aliases)})"
                yield Completion(
                    cmd_name,
                    start_position=-len(last_token),
                    display=display,
                    display_meta=command.description or "Command"
                )


class ConsoleKeyBindings:
    """控制台按键绑定"""

    def __init__(self):
        self._kb = KeyBindings()

        # Ctrl+C 中断当前输入
        @self._kb.add(Keys.ControlC, filter=~False)
        def _(event: KeyPressEvent):
            event.current_buffer.text = ""
            event.current_buffer.cursor_position = 0

        # Tab 补全
        @self._kb.add(Keys.Tab)
        def _(event: KeyPressEvent):
            b = event.current_buffer
            if b.text:
                # 触发补全
                b.start_completion(select_first=False)

    @property
    def bindings(self) -> KeyBindings:
        return self._kb


def create_console_style() -> Style:
    """创建控制台样式"""
    return Style.from_dict({
        # 命令提示符
        "prompt": "ansigreen bold",

        # 命令名称
        "command": "ansigreen",

        # 参数
        "arg": "ansicyan",

        # 错误消息
        "error": "ansired bold",

        # 成功消息
        "success": "ansigreen",

        # 警告消息
        "warning": "ansiyellow",

        # 信息消息
        "info": "ansiblue",

        # 补全菜单
        "completion-menu": "bg:#334155 #e2e8f0",
        "completion-menu.completion.current": "bg:#475569 #ffffff",
        "completion-menu.completion": "bg:#334155 #e2e8f0",

        # 补全候选
        "completion-toolbar": "bg:#1e293b #e2e8f0",
        "completion-toolbar.completion.current": "reverse",

        # 工具栏
        "toolbar": "bg:#1e293b #e2e8f0",

        # 选择条
        "selection": "reverse",
    })
