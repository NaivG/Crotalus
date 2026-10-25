"""End-to-end check for the unified command system.

Covers the pieces the other command checkers do not touch:

1. 统一命令线程：控制台与客户端命令都投递进 CommandDispatcher，按提交
   顺序串行执行（FIFO），前一条命令阻塞时后一条排队等待。
2. 执行边 side：SERVER 命令对玩家表现为未知命令；CLIENT 命令对控制台
   给出明确提示；BOTH 两端都可用。
3. 权限等级 level：0-4（仿原版 OP 体系）。ops 表来自 config
   [permissions].operators，改表立即生效。
4. 统一注册接口 register_command / register_commands。
5. 回复回程：ConsoleSender -> stdout；PlayerCommandSender -> System
   Chat 包（跨线程投回"网络循环"）。

Run: python tools/check_command_system.py
"""

import asyncio
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- psutil stub（仅当环境里的 psutil 不可导入时；console.commands 顶层要它）---
try:
    import psutil  # noqa: F401
except Exception:  # noqa: BLE001 - 本工具只关心命令系统，不关心真实内存
    _fake = types.ModuleType("psutil")

    class _FakeProcess:
        def memory_info(self):
            return types.SimpleNamespace(rss=0, vms=0)

    _fake.Process = _FakeProcess
    _fake.virtual_memory = lambda: types.SimpleNamespace(
        total=0, used=0, available=0, percent=0
    )
    sys.modules["psutil"] = _fake

from console.command import (  # noqa: E402
    CONSOLE_SENDER,
    Command,
    CommandContext,
    CommandSide,
    cmd_manager,
    register_command,
)
from console.commands import register_builtin_commands  # noqa: E402
from console.dispatcher import command_dispatcher  # noqa: E402
from console.permissions import permission_manager  # noqa: E402
from console.completer import ConsoleCompleter  # noqa: E402
from player.commands import PlayerCommandSender, register_player_commands  # noqa: E402

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def wait_for(predicate, timeout: float = 8.0) -> bool:
    """轮询等待条件成立（命令线程跨线程回复有一次性导入开销，不定时）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


# ---------------------------------------------------------------------------
# 0. Fake network loop + fake client handler (PlayerCommandSender's world)
# ---------------------------------------------------------------------------

class FakePlayer:
    def __init__(self, name: str):
        self.name = name
        self.game_mode = types.SimpleNamespace(value=0)


class FakeHandler:
    """足够 PlayerCommandSender / gamemode 命令使用的 ClientHandler 替身。"""

    def __init__(self, name: str, loop: asyncio.AbstractEventLoop):
        self.player = FakePlayer(name)
        self.remote_address = ("127.0.0.1", 12345)
        self._loop = loop
        self.sent_packets: list = []
        self.gamemode_calls: list[int] = []

    async def send_packet(self, packet) -> None:
        assert asyncio.get_running_loop() is self._loop, "packet sent off the network loop"
        self.sent_packets.append(packet)

    async def set_gamemode(self, mode: int) -> None:
        assert asyncio.get_running_loop() is self._loop, "set_gamemode ran off the network loop"
        self.gamemode_calls.append(mode)
        self.player.game_mode.value = mode


# ---------------------------------------------------------------------------
# 1. Test commands exercising side/level/serialization
# ---------------------------------------------------------------------------

ORDER: list[str] = []


class OrderCommand(Command):
    name = "order"
    description = "records execution order; 'sleep' blocks a moment"
    usage = "/order <sleep|now>"
    side = CommandSide.BOTH

    async def execute(self, ctx: CommandContext) -> bool:
        ORDER.append(f"start:{ctx.args[0]}")
        if ctx.args and ctx.args[0] == "sleep":
            await asyncio.sleep(0.3)
        ORDER.append(f"end:{ctx.args[0]}")
        return True


class OpOnlyCommand(Command):
    name = "oponly"
    description = "level 3 command"
    usage = "/oponly"
    side = CommandSide.BOTH
    level = 3

    async def execute(self, ctx: CommandContext) -> bool:
        ctx.reply("oponly ran")
        return True


def main() -> None:
    print("== unified command system check ==\n")

    register_builtin_commands()
    register_player_commands()
    register_command(OrderCommand())
    register_command(OpOnlyCommand())

    # 统一注册接口确认两边都在同一个注册表里
    check("registry holds console+client commands",
          cmd_manager.has_command("stop") and cmd_manager.has_command("gamemode"))

    # 补全只暴露控制台可见执行边
    completer = ConsoleCompleter()
    console_names = completer.command_manager.get_command_names(console_visible=True)
    check("completer hides CLIENT-side commands from console",
          "gamemode" not in console_names and "help" in console_names)

    # 命令线程
    command_dispatcher.start()
    try:
        # -- FIFO / 串行调度 --
        command_dispatcher.post("/order now", CONSOLE_SENDER)
        command_dispatcher.post("/order sleep", CONSOLE_SENDER)
        command_dispatcher.post("order now2", CONSOLE_SENDER)  # 无斜杠输入同样有效
        deadline = time.monotonic() + 5
        while len([o for o in ORDER if o.startswith("end")]) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        check("commands run FIFO on the command thread",
              ORDER[:2] == ["start:now", "end:now"]
              and ORDER[2:4] == ["start:sleep", "end:sleep"]
              and ORDER[4:] == ["start:now2", "end:now2"],
              f"order={ORDER}")

        # -- 控制台 side/level --
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            command_dispatcher.post("/gamemode creative", CONSOLE_SENDER)
            time.sleep(0.3)
        check("console rejects CLIENT-side command with a clear reason",
              "仅可在客户端执行" in buf.getvalue(), f"got: {buf.getvalue()!r}")

        buf = io.StringIO()
        with redirect_stdout(buf):
            command_dispatcher.post("/help", CONSOLE_SENDER)
            time.sleep(0.3)
        out = buf.getvalue()
        check("console help shows SERVER commands", "stop" in out and "status" in out)
        check("console help hides CLIENT commands", "gamemode" not in out)

        buf = io.StringIO()
        with redirect_stdout(buf):
            command_dispatcher.post("/help gamemode", CONSOLE_SENDER)
            command_dispatcher.post("/help stop", CONSOLE_SENDER)
            time.sleep(0.4)
        check("detailed help shows side and level",
              "CLIENT (仅客户端)" in buf.getvalue() and "gamemaster" in buf.getvalue(),
              f"got: {buf.getvalue()!r}")

        # -- 玩家侧：网络循环 + PlayerCommandSender --
        net_loop = asyncio.new_event_loop()
        net_thread = threading.Thread(target=net_loop.run_forever, daemon=True)
        net_thread.start()

        def make_sender(name: str) -> tuple[PlayerCommandSender, FakeHandler]:
            fut = asyncio.run_coroutine_threadsafe(_mk(name, net_loop), net_loop)
            return fut.result(2)

        async def _mk(name, loop):
            handler = FakeHandler(name, loop)
            return PlayerCommandSender(handler), handler

        def packet_texts(handler) -> list[str]:
            return [str(getattr(p, "message", "")) for p in handler.sent_packets]

        # 无 OP 玩家：level 0（首次回复要在命令线程上导入 network.packets，
        # 等待用轮询而非固定 sleep）
        sender, handler = make_sender("NoOp")
        command_dispatcher.post("/gamemode creative", sender)
        check("player without op is denied level-2 command",
              wait_for(lambda: any("权限不足" in t for t in packet_texts(handler)))
              and handler.gamemode_calls == [],
              f"packets={packet_texts(handler)}")

        # SERVER 命令对玩家 = 未知命令（不泄露存在性）
        sender, handler = make_sender("NoOp2")
        command_dispatcher.post("/stop", sender)
        check("SERVER command is unknown to players",
              wait_for(lambda: any("command.unknown.command" in t for t in packet_texts(handler))),
              f"packets={packet_texts(handler)}")

        # BOTH + 低等级：玩家可用 /help /list
        sender, handler = make_sender("NoOp3")
        command_dispatcher.post("/help", sender)
        check("player help lists BOTH commands",
              wait_for(lambda: any("/list" in t for t in packet_texts(handler))))
        check("player help hides SERVER commands",
              "/stop" not in " ".join(packet_texts(handler)))

        # 设 op 后：等级生效，gamemode 走通，回复经网络循环回到玩家
        from console.permissions import _config
        _config().set("permissions.operators", {"OpPlayer": 4, "OpPlayer2": 4})
        check("ops table applies without restart", permission_manager.get_player_level("OpPlayer") == 4)

        sender, handler = make_sender("OpPlayer")
        command_dispatcher.post("/gamemode creative", sender)
        check("op player gamemode executes via network loop",
              wait_for(lambda: handler.gamemode_calls == [1]),
              f"gamemode_calls={handler.gamemode_calls}")
        check("op player gamemode success reply sent",
              wait_for(lambda: any("gamemode.success.self" in t for t in packet_texts(handler))),
              f"packets={packet_texts(handler)}")

        # level 3 命令对 level 4 的 op 可用
        sender, handler = make_sender("OpPlayer2")
        command_dispatcher.post("/oponly", sender)
        check("level-4 op runs level-3 command",
              wait_for(lambda: any("oponly ran" in t for t in packet_texts(handler))))

        # 玩家权限不足 level 3
        sender, handler = make_sender("NoOp4")
        command_dispatcher.post("/oponly", sender)
        check("level-0 player denied level-3 command",
              wait_for(lambda: any("需要等级 3" in t for t in packet_texts(handler))),
              f"packets={packet_texts(handler)}")

        net_loop.call_soon_threadsafe(net_loop.stop)
    finally:
        command_dispatcher.stop(timeout=5)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
