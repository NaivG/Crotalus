"""
Console Interface

交互式控制台界面，使用 prompt_toolkit 实现。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional, Callable
from queue import Queue, Empty
from prompt_toolkit import Application, PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.history import InMemoryHistory
from loguru import logger

from console.command import CommandManager
from console.completer import ConsoleCompleter, create_console_style


class ConsoleInterface:
    """
    交互式控制台界面

    提供命令输入、历史记录、自动补全等功能。
    """

    def __init__(
        self,
        command_manager: Optional[CommandManager] = None,
        completer: Optional[Completer] = None,
        title: str = "Crotalus Console",
        prompt: str = "> ",
    ):
        self.command_manager = command_manager or CommandManager.get_instance()
        self.completer = completer or ConsoleCompleter(self.command_manager)
        self.title = title
        self.prompt = prompt

        self._running = False
        self._app: Optional[Application] = None
        self._session: Optional[PromptSession] = None
        self._input_queue: Queue = Queue()
        self._output_history: list[str] = []
        self._on_output: Optional[Callable[[str], None]] = None

        self._history = InMemoryHistory()
        self._style = create_console_style()
        self._kb = self._create_key_bindings()

    def _create_key_bindings(self) -> KeyBindings:
        """创建按键绑定"""
        kb = KeyBindings()

        # Ctrl+C 中断当前输入
        @kb.add(Keys.ControlC)
        def _(event: KeyPressEvent):
            event.current_buffer.text = ""
            event.current_buffer.cursor_position = 0

        # Ctrl+L 清屏
        @kb.add(Keys.ControlL)
        def _(event: KeyPressEvent):
            self._output_history.clear()
            self._refresh_display()

        return kb

    def _create_layout(self) -> Layout:
        """创建布局"""
        # 输出区域
        output_control = FormattedTextControl(
            text=self._get_output_text,
            focusable=False,
        )
        output_window = Window(
            content=output_control,
            height=10,
            style="class:output",
        )

        # 输入区域
        input_buffer = Buffer(
            history=self._history,
            multiline=False,
            completer=self.completer,
        )

        # 使用自定义布局
        from prompt_toolkit.layout.containers import VSplit, HSplit

        input_window = Window(
            content=FormattedTextControl(
                [(("class:prompt", self.prompt))]
            ),
            height=1,
        )

        main_container = HSplit([
            output_window,
            Window(height=1, char="─", style="class:separator"),
            Window(
                content=FormattedTextControl(
                    [(("class:prompt", self.prompt + " "))]
                ),
                height=1,
            ),
        ])

        return Layout(main_container)

    def _get_output_text(self) -> list[tuple[str, str]]:
        """获取输出文本"""
        lines = []
        for msg in self._output_history[-50:]:  # 只显示最后50行
            lines.append(("", msg))
        return lines

    def _refresh_display(self):
        """刷新显示"""
        if self._app and self._app.layout:
            self._app.layout.refresh()

    def print(self, message: str, end: str = "\n") -> None:
        """
        打印消息到控制台

        Args:
            message: 消息内容
            end: 结束符
        """
        self._output_history.append(message + end.rstrip("\n"))
        self._refresh_display()
        if self._on_output:
            self._on_output(message)

    def print_info(self, message: str) -> None:
        """打印信息消息"""
        self.print(f"[INFO] {message}")

    def print_success(self, message: str) -> None:
        """打印成功消息"""
        self.print(f"[OK] {message}")

    def print_warning(self, message: str) -> None:
        """打印警告消息"""
        self.print(f"[WARN] {message}")

    def print_error(self, message: str) -> None:
        """打印错误消息"""
        self.print(f"[ERROR] {message}")

    def run_sync(self) -> None:
        """
        同步运行控制台（阻塞）

        创建一个独立的线程来运行 asyncio 事件循环。
        """
        if self._running:
            logger.warning("Console is already running")
            return

        self._running = True

        def run_loop():
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.get_event_loop().run_until_complete(self.run_async())

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        thread.join()

    async def run_async(self) -> None:
        """异步运行控制台"""
        if self._running:
            logger.warning("Console is already running")
            return

        self._running = True
        logger.info("Console started")

        self._session = PromptSession(
            message=self.prompt,
            completer=self.completer,
            key_bindings=self._kb,
            style=self._style,
            history=self._history,
            multiline=False,
            rprompt=None,
        )

        while self._running:
            try:
                # 读取用户输入
                text = await self._session.prompt_async()
                if text.strip():
                    # 添加到历史
                    self._history.append_string(text)
                    # 执行命令
                    await self._execute_input(text)
            except KeyboardInterrupt:
                continue
            except Exception as e:
                logger.exception(f"Console error: {e}")
                self.print_error(f"Error: {e}")

        logger.debug("Console stopped")

    async def _execute_input(self, text: str) -> None:
        """执行用户输入：投递到统一命令线程。"""
        from console.command import CONSOLE_SENDER
        from console.dispatcher import command_dispatcher

        self.print(f"{self.prompt}{text}")

        # 控制台只负责投递；命令在命令线程上执行，输出经发送者回程。
        if not command_dispatcher.post(text, CONSOLE_SENDER):
            # 命令线程未运行（工具/测试环境）：就地执行兜底
            await self.command_manager.execute_command(text, CONSOLE_SENDER)

    def stop(self) -> None:
        """停止控制台"""
        self._running = False
        if self._app:
            self._app.exit()

    @property
    def is_running(self) -> bool:
        """检查控制台是否正在运行"""
        return self._running

    def set_on_output(self, callback: Callable[[str], None]) -> None:
        """
        设置输出回调

        Args:
            callback: 回调函数，接收消息字符串
        """
        self._on_output = callback


class SimpleConsoleInterface:
    """
    简化的控制台界面

    使用简单的同步方式运行，不需要完整的事件循环。
    适合在服务器主线程中使用。
    """

    def __init__(
        self,
        command_manager: Optional[CommandManager] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        title: str = "Crotalus Console",
        prompt: str = "> ",
    ):
        self.command_manager = command_manager or CommandManager.get_instance()
        self.event_loop = event_loop
        self.title = title
        self.prompt = prompt
        self._running = False
        self._history: list[str] = []
        self._output_buffer: list[str] = []

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """设置事件循环（用于在线程中执行协程）"""
        self.event_loop = loop

    def print(self, message: str) -> None:
        """打印消息"""
        print(message)
        self._output_buffer.append(message)

    def print_info(self, message: str) -> None:
        """打印信息消息"""
        self.print(f"[INFO] {message}")

    def print_success(self, message: str) -> None:
        """打印成功消息"""
        self.print(f"[OK] {message}")

    def print_warning(self, message: str) -> None:
        """打印警告消息"""
        self.print(f"[WARN] {message}")

    def print_error(self, message: str) -> None:
        """打印错误消息"""
        self.print(f"[ERROR] {message}")

    def _execute_command_sync(self, text: str) -> None:
        """把输入行投递到统一命令线程。

        控制台只负责投递，命令由命令线程统一调度执行（可能排队、可能
        耗时数秒），控制台线程立即回到输入提示符。输出经发送者回程：
        ConsoleSender.reply 即 print，命令线程打印的结果照常出现在这里。
        """
        from console.command import CONSOLE_SENDER
        from console.dispatcher import command_dispatcher

        if not command_dispatcher.post(text, CONSOLE_SENDER):
            # 命令线程未运行（工具/测试环境）：就地执行兜底，保持可用
            try:
                asyncio.run(self.command_manager.execute_command(text, CONSOLE_SENDER))
            except Exception as e:
                logger.exception(f"Command execution error: {e}")

    def run(self) -> None:
        """
        运行控制台（阻塞）

        主线程中使用简单的 input() 循环。
        """
        if self._running:
            logger.warning("Console is already running")
            return

        self._running = True
        logger.debug("Console started")

        while self._running:
            try:
                text = input(self.prompt)
                if text.strip():
                    self._history.append(text)
                    self._execute_command_sync(text)
            except KeyboardInterrupt:
                self.print("")
                continue
            except EOFError:
                break
            except Exception as e:
                logger.exception(f"Console error: {e}")
                self.print_error(f"Error: {e}")

        logger.debug("Console stopped")

    def stop(self) -> None:
        """停止控制台"""
        self._running = False

    @property
    def is_running(self) -> bool:
        """检查控制台是否正在运行"""
        return self._running
