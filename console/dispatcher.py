"""
Unified Command Dispatcher Thread (统一命令线程)

控制台与客户端的命令行不再各自就地执行，而是统一投递到本线程排队：

    控制台输入 ──┐
                 ├──> queue.Queue ──> CommandThread（常驻 asyncio loop）
    客户端命令 ──┘                          │ 单消费者，按提交顺序串行执行
                                            v
                                    CommandManager.execute_command

设计要点：

- **投递即返回**。``post()`` 只做一次队列 put，控制台线程立即回到输入
  提示符，网络线程立即继续读包 —— 命令执行（可能阻塞数秒，如
  /light relight 的 inline 排空）不再拖住任何输入端。
- **统一调度**。命令线程上只有一个消费者协程：取一条、执行完、再取下
  一条。命令之间天然互斥，命令里不再需要考虑与另一条命令并发的问题。
- **常驻事件循环**。命令在自己的 loop 上 await（actor 回复的
  ``msg.done.wait``、跨线程发包），且 ``asyncio.create_task`` 在命令里
  真正可用 —— 此前控制台每条命令用一个一次性 loop（``asyncio.run``），
  /stop 里 create_task 的停服任务会在 loop 关闭时被取消，从未执行过。
- **回复经发送者回程**。命令输出通过 ``CommandSender.reply()`` 回到
  各自的输出端（控制台 print / 玩家 System Chat），线程归属由发送者
  自己负责（玩家回复用 run_coroutine_threadsafe 投回网络循环）。

生命周期由 core/server.py 管理：start() 在监听端口之前（客户端的第一条
命令也要有归宿），stop() 在关停最开始（不再执行新的命令）。
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Optional

from loguru import logger

from console.command import CommandManager, CommandSender, cmd_manager

#: 队列深度上限。命令极轻（一行文本），满了只可能是消费者卡死 ——
#: 拒绝投递并让调用方给出反馈，而不是无声堆积。
MAX_PENDING_COMMANDS = 256


class CommandDispatcher:
    """统一命令线程。

    控制台与客户端共享同一个实例（模块级 ``command_dispatcher``），
    命令行经 ``post()`` 投递，由命令线程按提交顺序统一执行。
    """

    def __init__(self, manager: Optional[CommandManager] = None):
        self._manager = manager or cmd_manager
        self._queue: "queue.Queue[Optional[tuple[str, CommandSender]]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """启动命令线程（幂等）。"""
        if self._running:
            logger.warning("Command dispatcher is already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="CommandThread", daemon=True
        )
        self._thread.start()
        logger.info("Command dispatcher thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """停止命令线程。

        从命令线程自身调用时（/stop 命令 -> server.stop() ->
        dispatcher.stop()）不能 stop 自己的 loop：停服协程正跑在这个
        loop 上，中途停掉会把存档关停砍在半路。此时直接返回，loop 与
        守护线程随进程退出。
        """
        if not self._running:
            return
        self._running = False

        if threading.current_thread() is self._thread:
            logger.debug("Command dispatcher.stop() called from itself; leaving the loop to the daemon exit")
            return

        # 唤醒消费者（哨兵让它退出 get），再停 loop。
        self._queue.put(None)
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.debug("Command dispatcher stopped")

    def _run(self) -> None:
        """线程主体：常驻事件循环 + 单消费者协程。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        consumer = self._loop.create_task(self._consume())
        try:
            self._loop.run_forever()
        finally:
            consumer.cancel()
            self._loop.run_until_complete(asyncio.gather(consumer, return_exceptions=True))
            self._loop.close()
            logger.debug("Command thread stopped")

    async def _consume(self) -> None:
        """单消费者：取一条命令、执行完、再取下一条（统一调度的核心）。"""
        loop = asyncio.get_running_loop()
        while self._running:
            # run_in_executor 包住阻塞的 queue.get：投递即刻唤醒，无需轮询；
            # 阻塞的是默认执行器的一个线程，代价可忽略。
            item = await loop.run_in_executor(None, self._queue.get)
            if item is None or not self._running:
                break
            line, sender = item
            try:
                await self._manager.execute_command(line, sender)
            except Exception as e:  # noqa: BLE001 - 命令线程绝不能死
                logger.exception(f"Command dispatcher error executing '{line}': {e}")

    # -- posting ------------------------------------------------------------

    def post(self, line: str, sender: CommandSender) -> bool:
        """投递一行命令到命令线程（非阻塞）。

        Returns:
            True 已入队，将由命令线程统一执行；
            False 命令线程未运行或队列已满，调用方可就地回退。
        """
        if not self._running or self._thread is None:
            return False
        try:
            self._queue.put_nowait((line, sender))
        except queue.Full:
            logger.warning(
                f"Command queue full ({MAX_PENDING_COMMANDS}); dropping: {line!r}"
            )
            return False
        return True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pending(self) -> int:
        """当前排队未执行的命令数（状态/调试用）。"""
        return self._queue.qsize()


#: 全局命令线程实例。core/server.py 启动它，控制台与客户端向它投递。
command_dispatcher = CommandDispatcher()
