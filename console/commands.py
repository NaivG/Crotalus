"""
Console Built-in Commands

内置命令，提供基本的管理功能。

每条命令声明两个预设属性：

- side（执行边）：SERVER 仅控制台暴露，BOTH 控制台与客户端都暴露，
  CLIENT 仅客户端。服务端（CONSOLE）命令的输出直接 print 即回到控制台；
  BOTH 命令必须经 ctx.reply() 输出，玩家执行时才能收到回复。
- level（权限等级）：仿原版权限体系 0-4（见 console/permissions.py）。
  SERVER 命令只被控制台执行（控制台恒为 4 级），level 主要对 BOTH
  命令的客户端执行生效。

命令的执行由统一命令线程（console/dispatcher.py）调度，本文件不关心
命令从哪来。
"""

import asyncio
import psutil
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from console.command import (
    CONSOLE_SENDER,
    Command,
    CommandContext,
    CommandSide,
    command,
    cmd_manager,
    error,
)
from console.permissions import level_name
from loguru import logger

# The light query messages and their one producing function live in `world/`
# rather than here, because both sides of the round trip need them: the console
# posts and renders, the dimension actor answers. Keeping them in the console
# would mean `world/dimension_actor.py` importing `console`, which is the
# dependency direction the rest of the tree avoids.
from world.light_query import (
    LIGHT_QUERY_TIMEOUT,
    LightQueryMsg,
    LightRelightMsg,
    read_light,
)
from world.perf import PerfMsg


class HelpCommand(Command):
    """帮助命令

    执行边为 BOTH：控制台与玩家都要能看帮助。输出按发送者过滤 ——
    控制台看到 SERVER+BOTH 执行边的全部命令，玩家只看到 CLIENT+BOTH
    执行边且权限等级足够的命令。
    """

    name = "help"
    description = "显示帮助信息"
    usage = "/help [command]"
    side = CommandSide.BOTH

    async def execute(self, ctx: CommandContext) -> bool:
        if ctx.args:
            # 显示特定命令的帮助。玩家只看得到自己可用的命令（与原版一致，
            # 不可用的命令不存在）；控制台是操作者，允许查看任何已注册
            # 命令的详情 —— 执行边一栏会说明它属于哪一端。
            cmd_name = ctx.args[0].lower()
            cmd = cmd_manager.get_command(cmd_name)
            sender = ctx.sender or CONSOLE_SENDER
            allowed = cmd is not None and (sender.is_console or cmd.visible_to(sender))
            if not allowed:
                ctx.reply(error(f"Unknown command: {cmd_name}"))
                return True
            self._print_command_help(cmd, ctx)
            return True

        # 显示所有（对发送者可见的）命令
        self._print_all_commands(ctx)
        return True

    def _print_all_commands(self, ctx: CommandContext):
        """打印发送者可见的全部命令。"""
        sender = ctx.sender or CONSOLE_SENDER
        if sender.is_console:
            lines = [
                "",
                "=== Crotalus 控制台命令 ===",
                "",
                "可用命令 (执行边: SERVER/BOTH):",
                "-" * 40,
            ]
            for name, cmd in sorted(cmd_manager.get_all_commands().items()):
                if not cmd.side.console_visible:
                    continue
                aliases = f" (别名: {', '.join(cmd.aliases)})" if cmd.aliases else ""
                desc = cmd.description or "无描述"
                lines.append(f"  {name:<15} - {desc}{aliases}")
            lines += [
                "",
                "输入 /help <命令> 查看特定命令的详细帮助",
                "=" * 40,
                "",
            ]
        else:
            lines = ["可用命令:"]
            for name, cmd in sorted(cmd_manager.get_all_commands().items()):
                if not cmd.visible_to(sender):
                    continue
                aliases = f" (别名: /{', /'.join(cmd.aliases)})" if cmd.aliases else ""
                desc = cmd.description or "无描述"
                lines.append(f"/{name} - {desc}{aliases}")
        ctx.reply("\n".join(lines))

    def _print_command_help(self, cmd: Command, ctx: CommandContext):
        """打印单个命令的帮助（含执行边与权限等级）。"""
        lines = [f"=== {cmd.name} ===", ""]
        lines.append(f"描述: {cmd.description or '无描述'}")
        lines.append(f"用法: {cmd.usage}")
        if cmd.aliases:
            lines.append(f"别名: {', '.join(cmd.aliases)}")
        lines.append(f"执行边: {cmd.side.label}")
        lines.append(f"权限等级: {cmd.level} ({level_name(cmd.level)})")
        lines.append("")
        ctx.reply("\n".join(lines))


class StopCommand(Command):
    """停止服务器命令

    side=SERVER：只有控制台能停服。此前这个命令在控制台的一次性
    事件循环里 create_task，停服任务从未真正跑过；命令线程的常驻
    loop 让 create_task 的协程得以执行。
    """

    name = "stop"
    description = "停止服务器"
    usage = "/stop"
    side = CommandSide.SERVER
    level = 4

    def __init__(self, on_stop_callback=None):
        super().__init__()
        self._on_stop = on_stop_callback

    async def execute(self, ctx: CommandContext) -> bool:
        logger.info("Stopping server via console command...")

        # 触发停止回调
        if self._on_stop:
            asyncio.create_task(self._do_stop())
        else:
            logger.info("Stop command executed, but no stop callback registered")

        return True

    async def _do_stop(self):
        """执行停止"""
        if self._on_stop:
            await self._on_stop()


class StatusCommand(Command):
    """服务器状态命令"""

    name = "status"
    description = "显示服务器状态"
    usage = "/status"
    side = CommandSide.SERVER
    level = 0

    async def execute(self, ctx: CommandContext) -> bool:
        try:
            from core.server import get_server

            server = get_server()
            status = server.get_status()

            print("\n=== Crotalus 服务器状态 ===\n")

            # 基本信息
            print(f"运行状态: {'运行中' if status['running'] else '已停止'}")
            print(f"版本: {status['version']}")
            print(f"协议版本: {status['protocol']}")

            # 玩家信息
            players = status.get("players", {})
            print(f"在线玩家: {players.get('online', 0)} / {players.get('max', 0)}")

            # 线程状态
            threads = status.get("threads", {})
            if threads:
                print("\n线程状态:")
                for thread_name, thread_status in threads.items():
                    print(f"  {thread_name}: {thread_status}")

            # 世界信息
            world = status.get("world")
            if world:
                print("\n世界信息:")
                print(f"  名称: {world.get('name', 'Unknown')}")
                print(f"  种子: {world.get('seed', 'Unknown')}")

            print("\n" + "=" * 32 + "\n")

        except Exception as e:
            logger.exception(f"Failed to get status: {e}")
            print(f"Error getting status: {e}")

        return True


class ListCommand(Command):
    """列出在线玩家命令

    执行边 BOTH：原版 /list 对所有玩家开放。输出经 ctx.reply ——
    控制台执行时 print，玩家执行时以 System Chat 回给发送者本人。
    """

    name = "list"
    description = "列出在线玩家"
    usage = "/list"
    side = CommandSide.BOTH
    level = 0

    async def execute(self, ctx: CommandContext) -> bool:
        try:
            from core.server import get_server

            server = get_server()
            # 快照再遍历：连接字典被网络线程并发修改时迭代会炸
            connections = list(server._connections.items())

            lines = ["", "=== 在线玩家 ===", ""]
            if not connections:
                lines.append("当前没有玩家在线")
            else:
                lines.append(f"共 {len(connections)} 名玩家在线:")
                sender = ctx.sender or CONSOLE_SENDER
                for addr, conn in connections:
                    # A connection exists from the first byte, so the name is
                    # empty until the client logs in; show the address then.
                    player = getattr(conn, 'player', None)
                    player_name = (player.name if player else "") or addr
                    # 原版 /list 只给玩家看名字；地址是 log-ips 管的运维信息，
                    # 只对控制台（CONSOLE）显示。
                    if sender.is_console:
                        lines.append(f"  - {player_name} ({addr})")
                    else:
                        lines.append(f"  - {player_name}")
            lines.append("")
            ctx.reply("\n".join(lines))

        except Exception as e:
            logger.exception(f"Failed to list players: {e}")
            ctx.reply(error(f"Error listing players: {e}"))

        return True


class GCCommand(Command):
    """垃圾回收命令"""

    name = "gc"
    description = "手动触发垃圾回收"
    usage = "/gc"
    side = CommandSide.SERVER
    level = 3

    async def execute(self, ctx: CommandContext) -> bool:
        import gc

        before = gc.collect()
        print(f"垃圾回收完成，回收了 {before} 个对象")
        return True


class MemoryCommand(Command):
    """内存使用情况命令"""

    name = "memory"
    description = "显示内存使用情况"
    usage = "/memory"
    side = CommandSide.SERVER
    level = 2

    async def execute(self, ctx: CommandContext) -> bool:
        process = psutil.Process()

        mem_info = process.memory_info()
        print("\n=== 内存使用情况 ===\n")
        print(f"RSS (物理内存): {mem_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS (虚拟内存): {mem_info.vms / 1024 / 1024:.2f} MB")

        # 系统内存
        sys_mem = psutil.virtual_memory()
        print(f"\n系统内存:")
        print(f"  总计: {sys_mem.total / 1024 / 1024:.2f} MB")
        print(f"  已使用: {sys_mem.used / 1024 / 1024:.2f} MB")
        print(f"  可用: {sys_mem.available / 1024 / 1024:.2f} MB")
        print(f"  使用率: {sys_mem.percent}%")

        print()
        return True


class TPSCommand(Command):
    """TPS (每秒刻)命令

    执行边 BOTH、等级 2：原版 /tps 面向管理员。输出经 ctx.reply。
    """

    name = "tps"
    description = "显示服务器TPS"
    usage = "/tps"
    side = CommandSide.BOTH
    level = 2

    async def execute(self, ctx: CommandContext) -> bool:
        try:
            from core.server import get_server

            ticker = get_server().ticker
            lines = ["", "=== 服务器 TPS ===", ""]
            if ticker:
                tps = ticker.tps
                lines.append(f"当前TPS: {tps:.1f}")
                lines.append(f"目标TPS: 20.0")
                lines.append(f"Tick时间: {1000/tps:.1f}ms")

                if tps >= 19.0:
                    lines.append("状态: 良好")
                elif tps >= 15.0:
                    lines.append("状态: 警告 - 服务器负载较高")
                else:
                    lines.append("状态: 危险 - 服务器过载")
            else:
                lines.append("Tick线程未运行")

            lines.append("")
            ctx.reply("\n".join(lines))
        except Exception as e:
            logger.exception(f"Failed to get TPS: {e}")
            ctx.reply(error(f"Error getting TPS: {e}"))

        return True


class ClearCommand(Command):
    """清屏命令"""

    name = "clear"
    description = "清空控制台屏幕"
    usage = "/clear"
    aliases = ["cls"]
    side = CommandSide.SERVER
    level = 0

    async def execute(self, ctx: CommandContext) -> bool:
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        return True


class ReloadCommand(Command):
    """重新加载配置命令

    权限等级 4（owner）：重载会重置运行时状态；[permissions] 的
    operators 表是每次检查实时读取的，重载后立即生效。
    """

    name = "reload"
    description = "重新加载配置文件"
    usage = "/reload"
    side = CommandSide.SERVER
    level = 4

    async def execute(self, ctx: CommandContext) -> bool:
        try:
            from core.config import config

            config.reload()
            print("配置文件已重新加载")
        except Exception as e:
            logger.exception(f"Failed to reload config: {e}")
            print(f"Error reloading config: {e}")

        return True


class RegistryCommand(Command):
    """Inspect the loaded game registry.

    Read-only on purpose: the registry freezes before the world loads, so there
    is nothing a console command could change without handing out ids that
    saved chunks and connected clients already disagree with.
    """

    name = "registry"
    description = "查看游戏注册表 (blocks/items/tags/recipes)"
    usage = "/registry [blocks|items|tags|recipes|synced|plugins] [name]"
    aliases = ["reg"]
    side = CommandSide.SERVER
    level = 2

    async def execute(self, ctx: CommandContext) -> bool:
        try:
            from registry import get_game_registry, is_initialized
        except ImportError as error:
            print(f"Registry module unavailable: {error}")
            return True

        if not is_initialized():
            print("Registry has not been loaded yet.")
            return True

        registry = get_game_registry()
        topic = ctx.args[0].lower() if ctx.args else ""
        name = ctx.args[1] if len(ctx.args) > 1 else None
        extra = ctx.args[2] if len(ctx.args) > 2 else None

        try:
            if topic in ("", "info", "summary"):
                self._print_summary(registry)
            elif topic == "blocks":
                self._print_blocks(registry, name)
            elif topic == "items":
                self._print_items(registry, name)
            elif topic == "tags":
                self._print_tags(registry, name, extra)
            elif topic == "recipes":
                self._print_recipes(registry, name)
            elif topic == "synced":
                self._print_synced(registry, name)
            elif topic == "plugins":
                self._print_plugins(registry)
            else:
                print(f"Unknown topic '{topic}'. Usage: {self.usage}")
        except Exception as error:  # noqa: BLE001 - console must not crash the server
            logger.exception(f"registry command failed: {error}")
            print(f"Error: {error}")
        return True

    def _print_summary(self, registry) -> None:
        summary = registry.summary()
        print("\n=== Game registry ===\n")
        for key, value in summary.items():
            print(f"  {key:<18} {value}")
        print("\nSections: blocks, items, tags, recipes, synced, plugins")
        print()

    def _print_blocks(self, registry, name) -> None:
        if name is None:
            print(f"\n{len(registry.blocks.blocks)} blocks ({registry.blocks.total_states} states). "
                  f"Example: /registry blocks minecraft:oak_log")
            return
        block = registry.blocks.get_block(name)
        if block is None:
            print(f"No block {name}")
            return
        print(f"\n{block.name}  id={block.numeric_id}  states={block.state_count} "
              f"({block.first_state_id}..{block.last_state_id})")
        print(f"  default state: {block.default_state_id}  hardness={block.hardness} "
              f"item={block.item}")
        if block.properties:
            print(f"  properties: {', '.join(str(p) for p in block.properties)}")
        states = registry.blocks.states_of(block.name)
        for state in states[:8]:
            print(f"    {state.id:>6}  {state.properties or '-'}  light={state.luminance}")
        if len(states) > 8:
            print(f"    ... {len(states) - 8} more")
        print()

    def _print_items(self, registry, name) -> None:
        if name is None:
            print(f"\n{len(registry.items.items)} items. "
                  f"Example: /registry items minecraft:diamond_sword")
            return
        item = registry.items.get(name)
        if item is None:
            print(f"No item {name}")
            return
        print(f"\n{item.name}  id={item.numeric_id}  block={item.block}")
        for key, value in sorted(item.components.items()):
            text = str(value)
            print(f"    {key}: {text[:90]}{'...' if len(text) > 90 else ''}")
        print()

    def _print_tags(self, registry, registry_id, tag_name=None) -> None:
        if registry_id is None:
            print(f"\n{len(registry.tags.registries)} tag registries:")
            for candidate in registry.tags.registry_ids():
                print(f"  {candidate}: {len(registry.tags.tags_of(candidate))} tags")
            print("\nExample: /registry tags minecraft:block minecraft:planks")
            return

        tags = registry.tags.tags_of(registry_id)
        if not tags:
            print(f"No tags for registry {registry_id}")
            return

        if tag_name is None:
            # Listing every member of 329 block tags is not a useful answer to
            # "what tags are there".
            print(f"\n{registry_id}: {len(tags)} tags")
            for tag in tags:
                print(f"    {tag.name}: {len(tag)} entries")
            print(f"\nExample: /registry tags {registry_id} minecraft:planks")
            return

        matches = [tag for tag in tags if tag_name in tag.name]
        if not matches:
            print(f"No tag matching {tag_name!r} in {registry_id}")
            return
        for tag in matches:
            members = registry.tags.entries(registry_id, tag.name)
            print(f"\n{tag.name}: {len(tag)} entries ({len(members)} resolved)")
            for member in sorted(members)[:25]:
                print(f"    {member}")
            if len(members) > 25:
                print(f"    ... {len(members) - 25} more")
        print()

    def _print_recipes(self, registry, item) -> None:
        if item is None:
            print(f"\n{len(registry.recipes)} recipes. "
                  f"Example: /registry recipes minecraft:oak_planks")
            return
        recipes = registry.recipes.by_result(item)
        if not recipes:
            print(f"No recipe produces {item}")
            return
        print(f"\n{len(recipes)} recipe(s) for {item}:")
        for recipe in recipes[:20]:
            print(f"    {recipe.key}  [{recipe.type}]  {recipe.result}")
        print()

    def _print_synced(self, registry, registry_id) -> None:
        if registry_id is None:
            for candidate in registry.synced.registry_ids():
                print(f"  {candidate}: {len(registry.synced.entries(candidate))} entries")
            print()
            return
        keys = registry.synced.entry_keys(registry_id)
        if not keys:
            print(f"No synced registry {registry_id}")
            return
        print(f"\n{registry_id}: {len(keys)} entries")
        for index, key in enumerate(keys[:40]):
            print(f"    {index:>3}  {key}")
        if len(keys) > 40:
            print(f"    ... {len(keys) - 40} more")
        print()

    def _print_plugins(self, registry) -> None:
        records = registry.plugins.records
        if not records:
            print("No registry plugins registered.")
            return
        print(f"\n{len(records)} registry plugin(s):")
        for line in registry.plugins.summary():
            print(line)
        print()


class PluginsCommand(Command):
    """列出插件命令"""

    name = "plugins"
    description = "列出已加载的注册表插件"
    usage = "/plugins"
    side = CommandSide.SERVER
    level = 0

    async def execute(self, ctx: CommandContext) -> bool:
        try:
            from registry import get_game_registry, is_initialized
        except ImportError as error:
            print(f"Registry module unavailable: {error}")
            return True

        if not is_initialized():
            print("No registry plugins: the registry has not been loaded yet.")
            return True

        registry = get_game_registry()
        records = registry.plugins.records
        print(f"\n=== Registry plugins ({len(records)}) ===\n")
        if not records:
            print("None. Drop a module into the plugins directory to add one;")
            print("see registry/README.md for the interface.")
        for line in registry.plugins.summary():
            print(line)
        print()
        return True


class ChunksCommand(Command):
    """区块加载状态命令。

    Reports what the ticket system decided to keep in memory and why, which is
    the thing that used to be invisible: before the load-level work there was
    no unloading policy at all, so "how many chunks are loaded" only ever went
    up. `chunks forceload` adds the persisted FORCED ticket, `chunks unload`
    takes a chunk out now (through the save pipeline), `chunks save` drains the
    streaming saver and writes the world.
    """

    name = "chunks"
    description = "显示区块加载状态 / forceload / unload / save"
    usage = "/chunks [forceload|unload <x> <z> [dimension] | save]"
    side = CommandSide.SERVER
    level = 2

    def __init__(self, world_getter=None):
        super().__init__()
        self._world_getter = world_getter

    def _world(self):
        if self._world_getter is None:
            print("No world attached to this console.")
            return None
        return self._world_getter()

    async def execute(self, ctx: CommandContext) -> bool:
        world = self._world()
        if world is None:
            return False

        args = [a.lower() for a in ctx.args]
        if args:
            action = args[0]
            if action == "save":
                world.save()
                print("World saved (streaming save queue drained first).")
                return True
            if action == "forceload" and len(args) >= 3:
                return self._force_load(world, ctx, args)
            if action in ("unload", "evict") and len(args) >= 3:
                return self._unload(world, ctx, args)
            print(f"Usage: {self.usage}")
            return False

        return self._report(world)

    def _dimension(self, world, name: str | None):
        from world.dimension import DimensionType

        if not name:
            return DimensionType.OVERWORLD
        for candidate in DimensionType:
            if candidate.value == name or candidate.name.lower() == name:
                return candidate
        print(f"Unknown dimension: {name}")
        return None

    def _coords(self, args):
        try:
            return int(args[1]), int(args[2])
        except ValueError:
            print(f"Coordinates must be integers: {args[1]}, {args[2]}")
            return None

    def _force_load(self, world, ctx, args) -> bool:
        coords = self._coords(args)
        if coords is None:
            return False
        dim_type = self._dimension(world, args[3] if len(args) > 3 else None)
        if dim_type is None:
            return False
        loaded = not (len(args) > 4 and args[4] in ("remove", "off", "false"))
        queued = world.chunk_actor_inbox is not None
        ok = world.forceload(coords[0], coords[1], dim_type, loaded=loaded)
        state = "force-loaded" if loaded else "released"
        suffix = " (queued to the dimension thread)" if queued else ""
        print(
            f"Chunk {coords} in {dim_type.value}: {state}{suffix}"
            if ok
            else f"Chunk {coords} in {dim_type.value}: nothing changed"
        )
        return True

    def _unload(self, world, ctx, args) -> bool:
        coords = self._coords(args)
        if coords is None:
            return False
        dim_type = self._dimension(world, args[3] if len(args) > 3 else None)
        if dim_type is None:
            return False

        # Queued when an actor owns the dimension: it refuses chunks a ticket
        # still covers, and that decision belongs on the thread that owns the
        # ticket table.
        queued = world.chunk_actor_inbox is not None
        ok = world.unload_chunk_now(coords[0], coords[1], dim_type, save=True)
        if ok:
            suffix = " (queued to the dimension thread)" if queued else ""
            print(f"Chunk {coords} in {dim_type.value}: unloaded{suffix}")
        else:
            print(
                f"Chunk {coords} in {dim_type.value}: not unloaded "
                f"(a ticket still covers it, or it is not in memory)"
            )
        return True

    def _report(self, world) -> bool:
        stats = world.get_chunk_stats()
        saver = stats["saver"]
        print("\n=== Chunk loading ===\n")
        print(
            f"  saver: {saver['pending']} queued, {saver['in_flight']} in flight, "
            f"{saver['saved']} saved, {saver['failed']} failed, "
            f"{saver['workers']} worker(s)"
        )
        print()
        header = (
            f"  {'dimension':<20} {'mem':>6} {'want':>6} {'tickets':>8} "
            f"{'players':>8} {'loads':>7} {'drops':>7} {'saves':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, dim in stats["dimensions"].items():
            print(
                f"  {name:<20} {dim['in_memory']:>6} {dim['desired']:>6} "
                f"{dim['tickets']:>8} {dim['players']:>8} "
                f"{dim['requests']:>7} {dim['unloaded']:>7} "
                f"{dim['saved_on_unload']:>7}"
            )
        print()
        for name, dim in stats["dimensions"].items():
            types = ", ".join(
                f"{ticket_type} x{count}"
                for ticket_type, count in sorted(dim["ticket_types"].items())
            ) or "(none)"
            print(
                f"  {name}: ticks={dim['ticks']} untracked={dim['untracked']} "
                f"loading={dim['loading']} waiting-unload={dim['waiting_unload']}"
            )
            print(f"    ticket types: {types}")
        print()
        return True


# ---------------------------------------------------------------------------
# Light: the console side of `world/light_engine.py`
# ---------------------------------------------------------------------------

#: The largest `/light relight <cx> <cz> <radius>` this command accepts, in
#: chunks. 16 covers a 33x33 area (1089 chunks) -- more than a view distance,
#: and the size at which "queue the whole dimension" (`all`) is the honest way
#: to ask. A bigger radius is a typo, and queueing it would look like a hang.
MAX_RELIGHT_RADIUS = 16

#: How long the no-actor inline relight may spend draining its own queue, in
#: seconds. A patch is ~5 ms, so this is a few hundred chunks; past it the
#: command reports what is left queued rather than holding the console. With an
#: actor (the normal case) nothing waits at all -- the actor's light tick drains
#: the queue inside `light-budget-ms`.
INLINE_RELIGHT_SECONDS = 30.0


def _dimension_type(name: str):
    """A `DimensionType` from a console token, or None.

    Both spellings are accepted, `nether` and `minecraft:the_nether`, because
    `DimensionType` values are resource locations while everyone types the short
    name (`ChunksCommand._dimension` has the same rule).
    """
    from world.dimension import DimensionType

    for candidate in DimensionType:
        if candidate.value == name or candidate.name.lower() == name:
            return candidate
    return None


def _dimension_label(dim_type) -> str:
    """The short name of a dimension, for the lines a human reads.

    `DimensionType.value` is a resource location (`minecraft:the_nether`), which
    is right for a log line and wrong for "in ..." in a half-sentence; the enum
    name is what `--dim` documents and accepts.
    """
    return dim_type.name.lower()


def _parse_light_args(args: List[str]) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Split `--dim <name>` out of a `/light` argument list.

    Returns `(positional, dimension_name, error)`. `--dim=end` is accepted as
    well, and so is a bare trailing dimension name, which is the shape
    `/chunks forceload <x> <z> <dimension>` already uses -- a dimension name is
    never a coordinate, so the two cannot be confused.

    **Only `--` makes a token an option.** Coordinates are signed: `-13` is a
    chunk coordinate, and reading `token.startswith("-")` as "this is a flag" is
    what made `/light relight -13 0` answer "Unknown option: -13" -- which is
    every query west or north of the origin, i.e. most of the map.
    """
    positional: List[str] = []
    name: Optional[str] = None
    index = 0
    while index < len(args):
        token = args[index]
        lowered = token.lower()
        if lowered == "--dim":
            if index + 1 >= len(args):
                return [], None, "Missing value for --dim"
            name = args[index + 1]
            index += 2
            continue
        if lowered.startswith("--dim="):
            name = token.split("=", 1)[1]
            index += 1
            continue
        if token.startswith("--"):
            return [], None, f"Unknown option: {token}"
        positional.append(token)
        index += 1

    if name is None and len(positional) > 1 and _dimension_type(positional[-1].lower()):
        name = positional.pop()
    return positional, name, None


class LightCommand(Command):
    """Read one position's light, or force chunks to be lit again.

    Both halves of `LIGHT_PLAN.md` §6 stage 7:

        /light <x> <y> <z> [--dim <overworld|nether|end>]
        /light relight <cx> <cz> [<radius>] [--dim <dimension>]
        /light relight all [--dim <dimension>]

    The query answers the operator's "why is it dark here" with the two numbers
    the engine keeps -- sky light and block light -- plus their maximum, which is
    what `entity/spawn_placement.luminance_at` hands the spawner. It refuses to
    invent anything: an unloaded chunk says 区块未加载 rather than 0, and a chunk
    whose `light_populated` is False is labelled a fallback (the heightmap
    approximation) rather than printed as an engine result.

    The relight queues full 3x3 patches, which is the repair path for exactly
    those two cases. Three shapes, because "this chunk is wrong" and "this save
    is wrong" are different jobs: one chunk, a square around one (`<radius>` in
    chunks), or every generated chunk in the dimension (`all` -- the explicit
    full-map recompute `world/README.md` asks for, since light written before
    the engine existed is the all-15 default and nothing else would correct it).
    Queueing is cheap and asynchronous: the patches run inside the light tick's
    budget, right after this command returns.

    Neither half may read `Chunk` or `DimensionLight` from this thread: they
    belong to the dimension actor, where "single ownership is the concurrency
    control" (`world/dimension_actor.py`). Both go through
    `World._post_to_actor` and wait on the message's `done` event, the way
    `World.get_chunk_stats` uses `StatsMsg` -- with a 2 second timeout, after
    which the console says "actor 忙，稍后再试" instead of blocking forever.

    **The actor side is not wired up yet**: `LightQueryMsg` and `LightRelightMsg`
    are defined in this file precisely so the actor can stay untouched while this
    command lands, and until `DimensionActor._handle` learns them the reply never
    arrives and the timeout message is what an operator sees. When no actor owns
    the world at all (`world.chunk_actor_inbox is None`: tools and tests) there is
    nobody to race, so the read -- and, for `relight`, the patches -- happens on
    this thread instead.
    """

    name = "light"
    description = "查询某坐标的光照 / 强制重算一个区块、一片区域或全部已加载区块的光"
    usage = (
        "/light <x> <y> <z> [--dim <overworld|nether|end>] | "
        "/light relight <cx> <cz> [<radius>] [--dim <dimension>] | "
        "/light relight all [--dim <dimension>]"
    )
    side = CommandSide.SERVER
    level = 2

    def __init__(self, world_getter=None):
        super().__init__()
        self._world_getter = world_getter
        self.timeout = LIGHT_QUERY_TIMEOUT

    def _world(self):
        """The world behind this console, or None with a reason printed.

        A getter that answers None is the normal "the server has not loaded a
        world yet" case (`core/server` passes `lambda: self._world`), so it is
        reported the same way as no getter at all rather than failing silently.
        """
        world = self._world_getter() if self._world_getter is not None else None
        if world is None:
            print("World not available: no world is attached to this console "
                  "(or it has not been loaded yet).")
            return None
        return world

    async def execute(self, ctx: CommandContext) -> bool:
        world = self._world()
        if world is None:
            return False

        args, dim_name, error = _parse_light_args(list(ctx.args))
        if error is not None:
            print(error)
            print(f"Usage: {self.usage}")
            return False

        dim_type = _dimension_type((dim_name or "overworld").lower())
        if dim_type is None:
            print(f"Unknown dimension: {dim_name}")
            print(f"Usage: {self.usage}")
            return False

        dimensions = getattr(world, "dimensions", None) or {}
        dimension = dimensions.get(dim_type)
        if dimension is None:
            print(f"Dimension {_dimension_label(dim_type)} is not loaded in this world.")
            return False

        if args and args[0].lower() == "relight":
            return self._relight(world, dimension, dim_type, args[1:])
        return self._query(world, dimension, dim_type, args)

    def _actor_owns(self, world) -> bool:
        """Whether a dimension actor owns the world's state right now.

        `World.chunk_actor_inbox` is set by `core/server` while the actors run
        and cleared after they stop (`core/server.py:381`, `:481`). `getattr`
        rather than attribute access because a tool's stand-in for the world
        need not have it -- and a world without it has no owner, which is the
        "read it here" case.
        """
        return getattr(world, "chunk_actor_inbox", None) is not None

    # ---------------- /light <x> <y> <z> ----------------

    def _query(self, world, dimension, dim_type, args) -> bool:
        coords = self._block_coords(args)
        if coords is None:
            return False
        x, y, z = coords

        if self._actor_owns(world):
            # An actor owns the dimension: ask it, and never read the chunk from
            # here (see the class docstring).
            msg = LightQueryMsg(x=x, y=y, z=z)
            if not world._post_to_actor(dim_type, msg):
                self._no_reply(dim_type, "LightQueryMsg", posted=False)
                return True
            if not msg.done.wait(self.timeout):
                self._no_reply(dim_type, "LightQueryMsg", posted=True)
                return True
            answer = msg.box
        else:
            # No actor: this thread is the only one touching the dimension.
            try:
                answer = read_light(dimension, x, y, z)
            except Exception as error:  # noqa: BLE001 - console must not crash
                logger.exception(f"/light query failed: {error}")
                print(f"Light query failed: {error}")
                return False

        return self._print_query(_dimension_label(dim_type), x, y, z, answer)

    def _print_query(self, dim_name: str, x: int, y: int, z: int, answer) -> bool:
        print(f"\n=== 光照 ({x}, {y}, {z}) in {dim_name} ===\n")

        error = answer.get("error")
        if error:
            print(f"  查询失败 (light query failed): {error}")
            print()
            return False

        if "loaded" not in answer:
            # `done` was set without the contract being filled in: a partially
            # wired actor. Say so; do not read the missing key as "unloaded".
            print("  actor 回了空结果 (the reply carried none of the LightQueryMsg "
                  "keys): the actor's handler is incomplete")
            print()
            return False

        if not answer.get("loaded"):
            print(f"  区块未加载: chunk ({x >> 4}, {z >> 4}) 不在内存中")
            print("  未加载区块不报光值 —— 拒绝, 不是'亮'也不是'黑' "
                  "(an unloaded chunk refuses; it does not read as 0)")
            print()
            return True

        if not answer.get("in_range", True):
            print(f"  y={y} 超出该维度的高度范围 ({answer.get('min_y')}.."
                  f"{answer.get('max_y')}): 没有可读的光")
            print()
            return True

        if answer.get("fallback"):
            print("  区块已加载, 但 light_populated=False: 这个区块没有真光")
            print("  下面的数字是回退值 (heightmap 近似), 不是光照引擎的结果 "
                  "(fallback, not engine output)")
            top = answer.get("top")
            surface = f"y={top}" if top is not None else "(空列, 无参考面)"
            print(f"    参考面 top_motion_blocking_no_leaves: {surface}")
            print(f"    max(sky, block) 的替代值: {answer.get('level')}")
            print(f"  强制重算该区块: /light relight {x >> 4} {z >> 4} --dim {dim_name}")
            print()
            return True

        print(f"  天光 (sky light):     {answer.get('sky')}")
        print(f"  方块光 (block light): {answer.get('block')}")
        print(f"  max(sky, block):      {answer.get('level')}")
        if not answer.get("has_skylight", True):
            print("  (该维度没有天光: has_skylight=False, 天光恒为 0)")
        print()
        return True

    # ---------------- /light relight <cx> <cz> [<radius>] | all ----------------

    def _relight(self, world, dimension, dim_type, args) -> bool:
        target = self._relight_target(args)
        if target is None:
            return False
        all_loaded, chunk_x, chunk_z, radius = target

        if self._actor_owns(world):
            msg = LightRelightMsg(chunk_x=chunk_x, chunk_z=chunk_z,
                                  radius=radius, all_loaded=all_loaded)
            if not world._post_to_actor(dim_type, msg):
                self._no_reply(dim_type, "LightRelightMsg", posted=False)
                return True
            if not msg.done.wait(self.timeout):
                self._no_reply(dim_type, "LightRelightMsg", posted=True)
                return True
            answer = msg.box
        else:
            try:
                answer = self._relight_inline(dimension, chunk_x, chunk_z,
                                              radius=radius,
                                              all_loaded=all_loaded)
            except Exception as error:  # noqa: BLE001 - console must not crash
                logger.exception(f"/light relight failed: {error}")
                print(f"Relight failed: {error}")
                return False

        return self._print_relight(_dimension_label(dim_type), chunk_x, chunk_z,
                                   answer, radius=radius, all_loaded=all_loaded)

    def _relight_target(self, args) -> Optional[Tuple[bool, int, int, int]]:
        """Parse the shapes of `relight` into `(all, cx, cz, radius)`.

        `relight all` is the whole dimension; `relight <cx> <cz> [<radius>]` is
        a square of chunks around one (radius 0 = that chunk alone). A radius
        growing faster than the square (a million chunks) is a typo, not a
        request: refuse it rather than queueing it.
        """
        positional = [token for token in args]
        if not positional:
            print("Expected a chunk coordinate or 'all'. "
                  f"Usage: {self.usage}")
            return None
        if positional[0].lower() == "all":
            if len(positional) > 1:
                print(f"Unexpected argument after 'all': {' '.join(positional[1:])}")
                return None
            return True, 0, 0, 0

        if len(positional) not in (2, 3):
            print(f"Expected two chunk coordinates and an optional radius. "
                  f"Usage: {self.usage}")
            return None
        try:
            chunk_x, chunk_z = int(positional[0]), int(positional[1])
            radius = int(positional[2]) if len(positional) == 3 else 0
        except ValueError:
            print(f"Chunk coordinates and radius must be integers: "
                  f"{', '.join(positional)}")
            return None
        if radius < 0:
            print(f"Radius must not be negative: {radius}")
            return None
        if radius > MAX_RELIGHT_RADIUS:
            print(f"Radius {radius} covers {(2 * radius + 1) ** 2} chunks; "
                  f"the limit is {MAX_RELIGHT_RADIUS} "
                  f"({(2 * MAX_RELIGHT_RADIUS + 1) ** 2} chunks). "
                  f"Use 'all' for the whole dimension.")
            return None
        return False, chunk_x, chunk_z, radius

    def _relight_inline(self, dimension, chunk_x: int, chunk_z: int,
                        radius: int = 0, all_loaded: bool = False) -> Dict[str, object]:
        """No actor: queue, and run patches, on this thread.

        With no actor nobody else owns the dimension (`World._post_to_actor`),
        so doing here what the actor's light tick would have done is the same
        work on a different thread, not a second engine.

        An engine already attached to the dimension wins: a tool that built one
        owns the light field, and a second engine over the same chunks would
        compute the same numbers into the same arrays. Otherwise one is built
        with a budget that covers the request, and the queue is drained here
        rather than over the next few hundred ticks -- there is no actor ticking
        it, so "queued" would mean "queued forever". Its first call compiles the
        numba kernels (~5 s, `LIGHT_PLAN.md` §5.1) -- acceptable on a console
        path with no actor, and the reason the server warms them up at startup
        instead.
        """
        if all_loaded:
            chunks = getattr(getattr(dimension, "chunk_manager", None), "chunks", None) or {}
            targets = [(pos.x, pos.z) for pos in list(chunks.keys())]
        elif radius > 0:
            targets = [(chunk_x + dx, chunk_z + dz)
                       for dx in range(-radius, radius + 1)
                       for dz in range(-radius, radius + 1)]
        else:
            targets = [(chunk_x, chunk_z)]

        present = list(targets)

        def real(coord) -> bool:
            """Whether a generated chunk is behind this coordinate.

            `getattr` with a True default so a fixture chunk that says nothing
            about generation still counts -- but the None check has to come
            *first*: `getattr(None, ...)` answers the default, so the other order
            would count a missing chunk as generated.
            """
            chunk = dimension.get_chunk(*coord)
            return chunk is not None and getattr(chunk, "is_generated", True)

        answer: Dict[str, object] = {"chunks": len(targets), "skipped": 0}
        generated = [coord for coord in present if real(coord)]
        answer["skipped"] = len(present) - len(generated)
        answer["loaded"] = bool(generated)
        if not generated:
            return answer

        engine = getattr(dimension, "light", None)
        if engine is not None:
            for coord in generated:
                engine.queue_chunk(*coord)
            answer["queued"] = len(generated)
            answer["engine"] = "dimension.light"
            return answer

        try:
            from world.light_engine import DimensionLight

            engine = DimensionLight(dimension, chunks_per_tick=max(1, len(generated)))
        except Exception as error:  # noqa: BLE001 - report, do not crash
            answer["error"] = (
                f"no light engine available ({type(error).__name__}: {error})"
            )
            return answer

        for coord in generated:
            engine.queue_chunk(*coord)
        answer["queued"] = len(generated)
        answer["engine"] = "inline"
        # Drain it here: with no actor there is no `light.tick()` to come back
        # for the rest, and a command that left 700 patches queued would report
        # success and change nothing.
        deadline = time.monotonic() + INLINE_RELIGHT_SECONDS
        while engine.pending_chunks and time.monotonic() < deadline:
            engine.tick()
        answer["lit_now"] = engine.chunks_lit > 0
        answer["lit"] = engine.chunks_lit
        answer["deferred"] = engine.deferred_chunks
        answer["pending"] = engine.pending_chunks
        answer["dropped"] = engine.dropped_unloaded
        # `DimensionLight.tick` never raises: a kernel or a half-built chunk that
        # throws is counted and logged so the tick survives. Saying so here is
        # what keeps "queued, nothing happened" from looking like a quiet success.
        answer["engine_errors"] = engine.errors
        answer["engine_error"] = engine.last_error
        return answer

    def _print_relight(self, dim_name: str, chunk_x: int, chunk_z: int, answer,
                       radius: int = 0, all_loaded: bool = False) -> bool:
        if all_loaded:
            header = f"=== 重算光照 全部已加载区块 in {dim_name} ===\n"
        elif radius > 0:
            header = (f"=== 重算光照 chunk ({chunk_x}, {chunk_z}) "
                      f"半径 {radius} ({2 * radius + 1}x{2 * radius + 1}) in {dim_name} ===\n")
        else:
            header = f"=== 重算光照 chunk ({chunk_x}, {chunk_z}) in {dim_name} ===\n"
        print("\n" + header)

        error = answer.get("error")
        if error:
            print(f"  重算失败 (relight failed): {error}")
            print()
            return False

        if "chunks" not in answer and "loaded" not in answer and "queued" not in answer:
            print("  actor 回了空结果 (the reply carried none of the LightRelightMsg "
                  "keys): the actor's handler is incomplete")
            print()
            return False

        queued = answer.get("queued")
        # A single chunk answers `queued: True` (the old shape) or `queued: 1`.
        queued_count = int(queued) if isinstance(queued, (int, bool)) else 0
        chunks = answer.get("chunks")
        skipped = answer.get("skipped")

        if not answer.get("loaded") and not queued_count:
            if all_loaded:
                print("  这个维度里没有已生成的区块可重算 "
                      "(no generated chunk is loaded in this dimension)")
            else:
                print(f"  区块未加载: 没有可重算的光 "
                      f"(load it first, e.g. /chunks forceload {chunk_x} {chunk_z} {dim_name})")
            print()
            return True

        if queued_count:
            if chunks and chunks > 1:
                print(f"  已入队 {queued_count}/{chunks} 个区块: 每个都会做一次完整的 3x3 "
                      f"patch 重算 (queued for full patches)")
            else:
                print("  已入队: 该区块会做一次完整的 3x3 patch 重算 "
                      "(queued for a full patch)")
        else:
            print("  未入队: 引擎没有接受这次请求 (not queued)")

        if skipped:
            print(f"  跳过 {skipped} 个: 没有已生成的区块 (no generated chunk there)")

        neighbours = answer.get("neighbours")
        if neighbours is not None:
            note = "" if neighbours == 9 else \
                " —— 缺邻居时 patch 会推迟到邻居到齐 (LIGHT_PLAN.md §4.5)"
            print(f"  3x3 邻域: {neighbours}/9 个区块在内存中{note}")

        if answer.get("lit_now"):
            print("  已在本线程立即算出并写回 (the patches ran in this call)")
        if answer.get("lit"):
            print(f"  本次共点亮 {answer['lit']} 个区块 (patches written back)")
        if answer.get("engine_errors"):
            print(f"  patch 出错, 光没有写回: {answer.get('engine_error')} "
                  f"(the engine caught it; see the server log)")
        if answer.get("deferred"):
            print(f"  被推迟的 patch: {answer['deferred']} 次 (deferred, a neighbour is missing)")
        if answer.get("pending"):
            if answer.get("engine") == "inline":
                print(f"  还有 {answer['pending']} 项没跑完: 没有 actor 接手排空, 命令到 "
                      f"{INLINE_RELIGHT_SECONDS:.0f}s 上限就返回了 —— 再执行一次会接着跑 "
                      f"(no actor: the command stopped at its time limit)")
            else:
                print(f"  引擎队列里还有 {answer['pending']} 项 "
                      f"(the next light tick drains it)")
        print()
        return True

    # ---------------- shared plumbing ----------------

    def _no_reply(self, dim_type, message: str, *, posted: bool) -> None:
        """The actor never answered: say so, and say what that can mean."""
        label = _dimension_label(dim_type)
        if posted:
            print(f"  {label}: actor 忙，稍后再试 "
                  f"(no reply to {message} within {self.timeout:.1f}s)")
        else:
            print(f"  {label}: 没有 actor 接手这条消息，稍后再试 "
                  f"({message} was not posted)")
        print(f"  If this never succeeds, the actor side has no branch for "
              f"{message} yet (see the LightCommand docstring).")

    def _block_coords(self, args) -> Optional[Tuple[int, int, int]]:
        if len(args) != 3:
            print(f"Expected three world coordinates. Usage: {self.usage}")
            return None
        try:
            return int(args[0]), int(args[1]), int(args[2])
        except ValueError:
            print(f"Coordinates must be integers: {', '.join(args)}")
            return None


class PerfCommand(Command):
    """Tick 性能明细命令。

    Two sources, one report. The Ticker knows the per-actor lag bookkeeping
    (last tick duration, inbox backlog, ticks discarded while the actor was
    busy, how long an actor has been behind); each actor's `TickProfiler`
    knows where the time went *inside* its tick. The per-stage half is
    answered on the actor thread through `PerfMsg`, the same round trip
    `StatsMsg` uses -- the profiler is actor-owned state, and the console
    gets a dict of numbers, never a reference into it.

    The profiler is always on (~1 µs of `perf_counter` per tick), so the
    numbers are already in history when lag is noticed; there is no
    "enable profiling, reproduce, hope" step. `/perf reset` zeroes the
    accumulators (via the same message) for a clean window, e.g. right
    after reproducing a stall.
    """

    name = "perf"
    description = "显示 tick 性能明细 (TPS / 各维度 tick 内各阶段耗时)"
    usage = "/perf [reset] [--dim <overworld|nether|end>]"
    side = CommandSide.SERVER
    level = 2

    timeout = 2.0

    def __init__(self, world_getter=None):
        super().__init__()
        self._world_getter = world_getter

    async def execute(self, ctx: CommandContext) -> bool:
        from core.server import get_server

        args, dim_name, error = _parse_light_args(list(ctx.args))
        if error is not None:
            print(error)
            print(f"Usage: {self.usage}")
            return False
        bad = [a for a in args if a.lower() != "reset"]
        if bad:
            print(f"Unknown argument: {' '.join(bad)}")
            print(f"Usage: {self.usage}")
            return False
        reset = bool(args)

        print("\n=== Perf: tick 性能明细 ===\n")

        ticker = get_server().ticker
        if ticker is not None:
            self._print_ticker(ticker.snapshot())
        else:
            print("Tick 线程未运行 (the tick pipeline is stopped)")

        world = self._world_getter() if self._world_getter is not None else None
        if world is None:
            print("\n没有 world 挂到这个控制台: 跳过各维度的阶段明细")
            print()
            return True

        dimensions = getattr(world, "dimensions", None) or {}
        owned = getattr(world, "chunk_actor_inbox", None) is not None
        printed = False
        for dim_type in dimensions:
            if dim_name is not None and not self._dim_matches(dim_type, dim_name):
                continue
            printed = True
            self._print_dimension(world, dim_type, reset, owned)
        if dim_name is not None and not printed:
            print(f"\n维度 {dim_name} 未在这个 world 里加载")
        print()
        return True

    @staticmethod
    def _dim_matches(dim_type, name: str) -> bool:
        want = name.lower()
        return dim_type.value == want or dim_type.name.lower() == want

    @staticmethod
    def _print_ticker(snap: dict) -> None:
        tps = snap.get("tps", 0.0)
        rate = snap.get("tick_rate", 20)
        print(f"Ticker: TPS {tps:.1f} / {rate}   tick #{snap.get('tick_count', 0)}")
        actors = snap.get("actors") or {}
        if not actors:
            print("  (没有注册的维度 actor)")
            return
        print(f"  {'actor':<24} {'last ms':>8} {'inbox':>6} "
              f"{'discarded':>10} {'behind':>8}")
        for name, a in sorted(actors.items()):
            behind = a.get("behind_for")
            behind_s = f"{behind:.1f}s" if behind is not None else "-"
            print(f"  {name:<24} {a.get('last_duration_ms', 0.0):>8.2f} "
                  f"{a.get('queue_depth', 0):>6} {a.get('discarded', 0):>10} "
                  f"{behind_s:>8}")

    def _print_dimension(self, world, dim_type, reset: bool, owned: bool) -> None:
        label = dim_type.name.lower()
        if not owned:
            print(f"\n--- {label}: 没有 actor 在跑, 无 tick 明细 "
                  "(tools/tests path: nobody owns this dimension)")
            return

        msg = PerfMsg(reset=reset)
        if not world._post_to_actor(dim_type, msg):
            print(f"\n--- {label}: 没有 actor 接手 PerfMsg, 稍后再试")
            return
        if not msg.done.wait(self.timeout):
            print(f"\n--- {label}: actor 忙, {self.timeout:.0f}s 内未响应 (稍后再试)")
            return

        box = msg.box
        name = box.get("dimension", label)
        tick = box.get("tick") or {}
        tick_avg = tick.get("avg_ms") or 0.0
        tick_count = box.get("tick_count", 0) or 0
        print(f"\n--- {name}: tick avg {tick_avg:.2f}ms"
              f"  max {(tick.get('max_ms') or 0.0):.2f}ms"
              f"  last {(tick.get('last_ms') or 0.0):.2f}ms"
              f"   ({tick_count} ticks, {box.get('discarded', 0)} discarded,"
              f" inbox {box.get('queue_depth', 0)})")

        if reset:
            print("    (累计已重置 — 窗口从这条命令重新开始)")

        stages = box.get("stages") or {}
        if not stages:
            print("    (还没有 tick 记录)")
            return

        print(f"    {'stage':<16} {'avg ms':>8} {'share':>6} "
              f"{'max ms':>8} {'last ms':>8} {'calls':>8}")
        # share is computed from window totals, not per-call averages, so the
        # periodic stages (time_sync broadcasts every 20 ticks) are not
        # overstated by 20x: what matters is fraction of all tick time spent
        # in the stage, and both totals cover the same window.
        tick_total_ms = tick_avg * tick_count
        for stage_name, s in stages.items():
            stage_total = s.get("avg_ms", 0.0) * s.get("count", 0)
            share = (stage_total / tick_total_ms * 100.0) if tick_total_ms > 0 else 0.0
            print(f"    {stage_name:<16} {s.get('avg_ms', 0.0):>8.2f} {share:>5.0f}%"
                  f" {s.get('max_ms', 0.0):>8.2f} {s.get('last_ms', 0.0):>8.2f}"
                  f" {s.get('count', 0):>8}")


def register_builtin_commands(stop_callback=None, world_getter=None) -> None:
    """注册所有内置命令"""

    # 注册帮助命令
    cmd_manager.register(HelpCommand())

    # 注册停止命令
    stop_cmd = StopCommand(on_stop_callback=stop_callback)
    cmd_manager.register(stop_cmd)

    # 注册状态命令
    cmd_manager.register(StatusCommand())

    # 注册玩家列表命令
    cmd_manager.register(ListCommand())

    # 注册垃圾回收命令
    cmd_manager.register(GCCommand())

    # 注册内存命令
    cmd_manager.register(MemoryCommand())

    # 注册TPS命令
    cmd_manager.register(TPSCommand())

    # 注册tick性能明细命令
    cmd_manager.register(PerfCommand(world_getter=world_getter))

    # 注册清屏命令
    cmd_manager.register(ClearCommand())

    # 注册重载命令
    cmd_manager.register(ReloadCommand())

    # 注册插件列表命令
    cmd_manager.register(PluginsCommand())

    # 注册注册表查看命令
    cmd_manager.register(RegistryCommand())

    # 注册区块加载状态命令
    cmd_manager.register(ChunksCommand(world_getter=world_getter))

    # 注册光照查询/重算命令
    cmd_manager.register(LightCommand(world_getter=world_getter))

    logger.debug("Built-in commands registered")
