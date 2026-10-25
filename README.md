# Crotalus

> 用 Python + numba JIT 从零实现的 Minecraft Java 版 **1.21.11** 服务端（协议 **774**）

Crotalus 不包装 vanilla，也不基于 Paper。网络协议、游戏注册表、世界生成、区块生命周期、
光照、物理和生物 AI 都在同一个仓库里；纯 Python 保持可读，计算热点（地形、光照、碰撞、
噪声）交给 numba 编译。

**当前状态：Alpha（`0.0.1.dev0`）**

## 特性

- **完整连接链路**：handshake → status → login（离线 / Mojang 正版验证 / AES-CFB8 加密 /
  可选压缩）→ configuration（known packs、完整注册表、tags、keep-alive）→ play。
- **1.21.11 数据驱动注册表**：1,166 个方块 / 29,671 个状态、1,505 个物品、1,470 个配方、
  1,131 个标签、371 项 synced registry，启动时加载并在世界生成前冻结；支持插件与 datapack。
- **原版对齐的世界生成**：density function、噪声、生物群系、地表规则、洞穴、地貌特征和
  矿脉按 1.21.11 数据移植；主世界地形逐格一致率 97.4–99.6%，地表规则约 98.1%。
  另有 `legacy` 后端（自研 Perlin，约 20 ms/区块）用于快速生成。
- **可扩展的区块系统**：ticket 等级、自动加载/卸载、Anvil 就地扇区写入、后台进程池、
  视野内流式推送与增量光照包。
- **真实的天空光 + 方块光**：numba 泛洪引擎，3×3 patch 一次跨区块算对边界，增量队列受
  每 tick 预算约束，光照随区块落盘并在线路上更新。
- **服务端权威物理**：AABB 扫掠、重力、摩擦、台阶和睡眠体；玩家移动在服务端重算，
  穿墙或超速会被修正。
- **实体与生物**：实体追踪与增量包；鸡、牛、羊、猪、兔的自然刷怪、goal 选择器和
  A* 寻路。
- **可运维**：prompt_toolkit 控制台、OP 等级权限，以及 `perf`、`tps`、`chunks`、`light`、
  `registry` 等命令。

## 项目进度

| 领域 | 状态 | 说明 |
|---|---|---|
| 协议 / 登录 | ✅ | 1.21.11 / 774；离线与正版验证、配置阶段、play 阶段、聊天、玩家列表 |
| 注册表 / 数据包 | ✅ | 43 张辅助数据表；插件和 datapack 可扩展；启动约 1.2 s / 90 MB |
| 世界生成 | 🟡 | 主世界可用；下界/末地的 vanilla density 尚未复现，含水层与结构生成缺失 |
| 区块 / 存档 | ✅ | ticket、自动卸载、流式保存、forceload 跨重启恢复 |
| 光照 | ✅ | 完整的光照引擎：天光 + 方块光、增量更新、随区块持久化 |
| 物理 / 移动校验 | ✅ | 碰撞、重力、摩擦、台阶；水/岩浆仍是简化模型 |
| 实体 / 生物 | 🟡 | 追踪与五种被动动物（鸡/牛/羊/猪/兔）已通；无繁殖、掉落，生物不写盘 |
| 方块交互 / 物品 | ⛔ | 挖掘、放置、背包、容器、方块实体均未接入 |
| 战斗 / 生存 | 🟡 | 健康/伤害/死亡模型已有；攻击、伤害来源、摔落伤害和死亡/重生尚未接线 |
| 控制台 / 权限 | ✅ | 14 条控制台命令、4 条玩家命令、OP 0–4 |

## 快速开始

### 环境

Python **3.10+**。推荐 free-threaded **3.14**：世界生成、光照和物理都是 numba 内核，
无 GIL 时线程模式可以真正并行，且不必为每个 worker 付一份解释器内存。

普通构建同样能跑，但后台生成应使用 `multiprocess` 模式。

```bash
git clone https://github.com/NaivG/Crotalus.git
cd Crotalus
py -3.14t -m pip install -r requirements.txt
py -3.14t main.py
```

> `py -3.14t` 是 Windows Python Launcher 的写法；Linux/macOS 上使用对应的
> `python3.14t`（或普通 `python`）即可。

启动后用 **Minecraft 1.21.11** 客户端连接 `localhost:25565`。默认 `online-mode = false`，
任意用户名都能进；要正版验证就把 `config.toml` 的 `online-mode` 改成 `true`（需能访问
Mojang session server）。

> **第一次启动会比较久。** 服务端会先完成各模块编译，生成出生区（`spawn-radius = 4`，9×9 = 81 个区块）并
> 完成光照，`vanilla` 后端单核约 1–1.2 s/区块。磁盘上已有的区块不会重新生成。

### 命令行

```text
py -3.14t main.py [-p PORT] [-m MAX_PLAYERS] [--ip IP] [--version]
```

| 参数 | 作用 |
|---|---|
| `-p, --port` | 覆盖 `server.server-port` |
| `-m, --max-players` | 覆盖 `server.max-players` |
| `--ip` | 绑定地址 |
| `--version` | 打印版本与协议号后退出 |

### 配置

所有配置都在 `config.toml`，常用项：

| 键 | 默认 | 说明 |
|---|---|---|
| `server.server-port` | `25565` | 监听端口 |
| `server.online-mode` | `false` | 正版验证；离线时任何人可用任意名字 |
| `server.max-players` | `20` | 人数上限 |
| `server.view-distance` | `10` | 视野半径（区块） |
| `server.simulation-distance` | `10` | 实体/方块 tick 半径 |
| `server.level-name` / `server.level-seed` | `world` / 随机 | 存档目录与种子 |
| `[permissions] operators` | `{ "Atoc501" = 4 }` | 改成自己的 ID，否则无法用 `/gamemode`、`/tps` |
| `[world] generator` | `vanilla` | 对齐原版；`legacy` 快但地形不同 |
| `[world] chunk-generation-mode` | `thread` | **非 free-threaded 构建改成 `multiprocess`** |
| `[world] chunk-generation-workers` | `8` | 生成进程/线程数，`0` = 自动 |
| `[world] light-budget-ms` | `4.0` | 每 tick 光照预算 |
| `[advanced] path-nodes-per-tick` | `200` | 所有寻路搜索每 tick 的 A* 扩展上限 |
| `[advanced] server-authoritative-movement` | `true` | 服务端移动校验 |

修改配置后需要重启；权限表在 `reload` 后立即生效。世界存档位于 `data/<level-name>/`
（Anvil 区域文件），日志在 `logs/`（旧日志自动 gzip）。

### 控制台与命令

控制台支持自动补全：

```text
help  status  list  tps  perf  chunks  light  registry  plugins
memory  gc  reload  clear  stop
```

- `tps` / `perf`：TPS 与每个维度 tick 内各阶段耗时；
- `chunks`：加载状态，支持 `forceload` / `unload` / `save`；
- `light`：查询坐标光照；`light relight <cx> <cz> [radius]` 或 `light relight all` 强制重算；
- `registry`：查看方块/物品/标签/配方规模；
- `stop` 只能从控制台执行。

玩家侧命令：`/help`、`/list`、`/tps`、`/gamemode <survival|creative|adventure|spectator>`。
其中 `/tps` 和 `/gamemode` 需要 OP 等级 2，控制台恒为 4 级。

### 插件与数据包

把 `*.py` 放进 `plugins/`，datapack 放进 `datapacks/`；注册表会在启动时合并后冻结。
API 和示例见 [registry/README.md](registry/README.md)。

## 性能

测试环境：**Intel i7-8550U（4 核 8 线程）/ Windows 10 / CPython 3.14.8 free-threaded**。

| 场景 | 实测 | 备注 |
|---|---|---|
| 500 个物理实体，20% 在动 | **1.25 ms/tick** | 预算的 2.5%；`tools/bench_physics.py` |
| 单实体物理（静止 / 下落 / 撞墙） | 6.8 / 9.3 / 13.3 µs | 睡觉体几乎只剩调用开销 |
| 3×3 真实区块光照 patch | best **3.95** / p50 **5.81 ms** | `tools/bench_light.py`；4 ms 预算约点亮 0.7 区块/tick |
| 单区块天光 + 方块光 | ~0.25 ms | 9 个同构真实区块摊分 |
| 单次方块光照变更（完整引擎路径） | p50 **1.20 ms**，p99 4.55 ms | 含队列、region 与写回 |
| A* 寻路 12 格（平地 / 绕障）、40 格 | 3.2 / 3.5 / **10.4 ms** | `tools/bench_pathfinding.py` |
| `vanilla` 地形生成 | ~1.0–1.2 s/区块/核 | 单核纯 Python 的工作量 |
| `legacy` 地形生成 | ~20 ms/区块 | 比 `vanilla` 快一个数量级以上，地形形状是自研的 |
| 后台生成吞吐（4 进程） | **≈3.7×** 单核 | `tools/bench_chunk_parallelism.py` |
| 热启动 | ~7.2 s | 注册表 1.1 s + numba 预热 + 出生区加载 |

说明：

- 光照内核用 `cache=True`。**首次启动，或改过内核源码后的第一次启动**要额外花数十秒编译；
  命中缓存后约 1 s。`/perf` 能看到实际 tick 耗时。
- 3×3 patch 通常由 `light-budget-ms`（4 ms）限制，而不是 `light-chunks-per-tick`；
  想更快就调预算，代价是单 tick 更长。
- 1.25 ms 只含物理阶段；按 50 ms 折算约 2 万实体的理论容量，实际整 tick 还包含区块、
  光照、AI 和发包。
- 数字会随地形、区块内容与后台负载浮动；基准脚本打印目标值，不做断言。

### 调优旋钮

- 低配 / 快速试跑：`generator = "legacy"`、`server.view-distance = 6`、
  `chunk-generation-workers = 4`。
- 内存：`multiprocess` 的每个 worker 是完整解释器（约 200–400 MB）；内存紧张时减少 worker，
  或使用 free-threaded 构建 + `chunk-generation-mode = "thread"`。
- 卡顿：先看 `/perf`。寻路重的场景调低 `advanced.path-nodes-per-tick`；区块涌入太猛调低
  `light-budget-ms` 或 `advanced.physics-view-adds-per-tick`。
- 带宽：默认 `network-compression-threshold = -1`（关闭）；设成 `256` 可启用 zlib 压缩。

## 架构

```
Ticker（20 Hz，丢弃过期 tick，不排队追帧）
  └─ 每个维度一个 DimensionActor 线程
       dimension → lifecycle → physics_view.sync → physics → player_validator
                 → spawner → light → entity_tracker → streamer
```

| 目录 | 职责 |
|---|---|
| `core/` | 服务器生命周期、tick 调度、线程、配置、Mojang 认证 |
| `network/` | 协议 774 的包定义、编解码、连接与状态机 |
| `registry/` | 方块/物品/配方/标签/世界生成数据与插件接口 |
| `world/` | 维度、区块生命周期、生成后端、光照、Anvil 存档 |
| `physics/` | numba 碰撞/重力内核、世界视图、玩家移动校验 |
| `entity/` | 实体与追踪、刷怪、AI goal、A* 寻路 |
| `player/` | 玩家状态、持久化、客户端命令 |
| `console/` | 控制台、命令框架、OP 权限 |
| `datatypes/` | NBT、MiniMessage 等数据类型 |
| `tools/` | 53 个 `check_*.py` 验证脚本、9 个 `bench_*.py` 基准 |
| `assets/` | 1.21.11 数据（方块、注册表、世界生成、语言文件等） |

## 验证

```bash
# 在仓库根目录；推荐 free-threaded 解释器
py -3.14t tools/check_registry.py
py -3.14t tools/check_light_engine.py
py -3.14t tools/check_physics_solver.py
py -3.14t tools/check_pathfinding.py
py -3.14t tools/check_mob_ai.py
```

53 个 check 覆盖注册表、世界生成逐格对照、区块编解码与生命周期、光照与参考实现、
物理、实体包、刷怪、AI、命令系统和服务端管线；各计划文档记录了当轮结果与已知失败项。
基准脚本包括 `bench_physics.py`、`bench_light.py`、`bench_pathfinding.py`、
`bench_chunk_parallelism.py` 等。

## 已知限制

- **还不是可玩的生存服务器**：没有挖掘/放置、背包、容器、合成、战斗、掉落物。
- 只实现了五种被动动物（鸡、牛、羊、猪、兔）；没有繁殖、掉落，生物不写盘。
- `vanilla` 后端的下界接近空、末地全空气；地下水用简化模型代替 aquifer；无结构生成。
- 水/岩浆物理是近似；旧存档的光照不会自动重算，需要 `light relight all`。
- 世界边界、gamerule 引擎、红石、记分板等原版系统尚未实现。

## 文档

- 世界模块：[world/README.md](world/README.md)
- 原版世界生成细节：[world/vanilla/README.md](world/vanilla/README.md)
- 注册表与插件接口：[registry/README.md](registry/README.md)

## 许可

代码以 **GPL-3.0** 发布，见 [LICENSE](LICENSE)。

`assets/` 的数据主要来自 Pumpkin（Rust 服务端）在 1.21.11 提交上生成的报告。

Minecraft 是 Mojang Studios 的商标，本项目与 Mojang / Microsoft 无隶属关系。
