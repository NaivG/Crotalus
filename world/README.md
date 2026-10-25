# World Module

Minecraft世界管理系统，支持维度、生成、加载与保存等功能。

## 模块结构

```
world/
├── __init__.py      # 模块入口
├── generator.py     # 生成器工厂 + 两个后端共用的基类（不含算法）
├── feature/         # 装饰层：生物群系/地表/洞穴/特征，两个后端共用
├── legacy/          # 自研算法：Perlin 高度图 + 自研矿脉 + 自研隧道洞穴
├── vanilla/         # vanilla 移植：RNG/噪声/density function/生物群系/地表/洞穴/特征
├── chunk.py         # 区块数据结构 + ChunkManager（内存边界）
├── block_states.py  # 内部方块ID -> 1.21.11 全局状态ID（由注册表推导）
├── chunk_codec.py   # 区块 -> LevelChunkWithLight 网络载荷
├── chunk_ticket.py  # 加载等级：ticket / 等级传播 / 加载类型
├── chunk_lifecycle.py # 自动加载卸载：ticket -> 内存里的区块
├── chunk_streamer.py # 按视野给玩家增量推送区块包
├── chunk_saver.py   # 流式保存：离开内存的区块在该线程写盘
├── anvil.py         # Anvil 区域文件（就地扇区写入）
├── dimension.py     # 维度管理
├── world.py         # 主世界类
└── serializer.py    # 序列化和持久化
```

## 地形生成后端：`[world] generator`

地形有两种实现，`config.toml` 的 `[world] generator` 选一种：

| 值 | 实现 | 说明 |
| --- | --- | --- |
| `"vanilla"`（默认） | `world/vanilla/terrain.py` | 地形 = `assets/density_function.json` 里的 density function。逐格对上真原版 97.4–99.6% |
| `"legacy"` | `world/legacy/generator.py` | 地形 = 自研 numba Perlin 高度图。约 20ms/区块，比 vanilla 快约 30 倍，形状是它自己的 |

**两个后端的装饰层完全相同** —— 生物群系、地表规则、洞穴、地貌特征都走
`world/feature/`，底层换成 Perlin 还是 density 它们一行都不用改。legacy 意
味着"快但地形不是 Minecraft"，不是"简陋"。

代价：vanilla 后端单核约 **1 秒/区块**（density 填充 ~140ms，生物群系
200–450ms，地表规则 ~600ms，洞穴 35–55ms，特征 ~40ms）。`view-distance = 10`
时一个玩家的首屏要等几分钟。数字见 `world/vanilla/README.md` 的 "Cost"。

改这个值**不会重算磁盘上已有的区块**；没生成过的区域才会用新后端。

**已知缺口**：nether 和 end 两个 preset 的 density 场目前复现不了原版（末地
根本不走 noise router）。详见 `world/DESIGN-density-preset-gaps.md`。所以
`generator = "vanilla"` 时下界接近空、末地全是空气 —— 这是有意接受的。

## 与游戏注册表的关系

世界生成/加载**必须**在 `registry.init_registries()` 之后进行（`core/server.py` 在
`_init_world()` 前调用）。接线分两层：

**一、方块状态 ID（chunk → wire → 存档）**

- `chunk.py` 的每个格子存的是一个**全局方块状态 ID**（`uint16`），也就是客户端在
  区块 palette 里看到的那个数，和 `assets/blocks.json` 一一对应；区块自己的编号
  **就是**注册表的编号，没有第二套；
- `block_states.py` 是唯一的"方块名 → 状态ID"入口（`BLOCKS.STONE`、`state_id(...)`）。
  旧版本用一张 1.8 时代的内部ID表做翻译，那张表在启动时分配、被写进存档，导致
  存档的含义取决于当时加载了什么（插件换顺序 = 一个方块变成另一个方块），而且只有
  33 个ID，而 worldgen 数据里点名了 176 个方块。整条翻译层已删除；
- 存档（`anvil.py`）写的是 `{Name, Properties}` + 打包索引（和 vanilla 同形），
  所以存档**按名字**记住方块：换一份 `assets/blocks.json` 后，读不出来的方块变成
  空气并打一条 warning，而不是被误解成别的方块；
- 插件方块直接用注册表分配给它的状态ID，不存在"内部ID"可以撞车。

验证：`python tools/check_chunk_storage.py`（状态ID、存档格式、未知方块降级、
旧格式不被猜测）。

**二、世界生成参数（registry.worldgen → generator/dimension）**

`registry.worldgen` 把 `assets/` 里那 12 个世界生成文件变成可查询的类型化视图，
world 侧读的是它：

| 值 | 来源 | 消费者 |
| --- | --- | --- |
| 维度 min_y / height / logical_height / 环境属性 | `dimension.json` | `dimension.py::DimensionSettings.for_dimension` |
| 海平面、默认方块/流体、地形 shape | `chunk_gen_settings.json`（按 `preset`） | `generator.py::BaseGenerator._apply_registry` |
| 默认生物群系 ID | `biome.json` + `synced_registries.json` 顺序 | `chunk_codec.py::default_biome_id` |

所以：主世界的海平面现在是数据里的 **63**（以前手写 62）、主世界维度是 -64..319
（`DimensionSettings` 与区块编码器用同一个高度）、下界是 256 高（以前被写成了
128，等于漏掉一半区块），每个维度用自己的生成 preset（下界真的填 netherrack +
lava、末地填 end_stone + air——方块按名字解析，不再受"内部ID表只有 32 项"限制）。

区块**存储**也按维度开窗：`Chunk` 携带 `min_y / height`（主世界 24 个 section，
Y -64..319），`ChunkManager` 与 `RegionManager` 用 `DimensionSettings` 给每个区块
盖章，存档的 section `Y` 标签是绝对值（主世界写 -4..19）。所以 surface 规则的
`above_bottom` 基岩带（Y -64..-59）落在区块数组里、真的会被放置；基岩只由那条
`vertical_gradient` 规则产生，`_generate_terrain` 不再手放。

`registry.worldgen.validate()` 在每次启动时做全量交叉校验（约 60ms，结果进
`GameRegistry.problems`）：

- 每个 preset 的 15 个 noise router 槽位都能解析到 `density_function.json` 的对应
  节点（`final_density` ↔ `finalDensity`）；
- 每个 `noise` 引用都在 `noise_parameters.json` 里；
- 每个生物群系的 `carvers` / `features` 都能在 `carver.json` / `placed_feature.json`
  → `configured_features.json` 里找到，每个 spawner 的实体都存在于实体注册表；
- 每个 surface rule 的方块状态与 `biome_is` 都能解析；
- 每个维度都有对应的 preset，`infiniburn` tag 存在；
- `chunk.json` 的 98304 个值都是合法的全局状态ID（air/stone/water）。

校验：`python tools/check_worldgen_data.py`（worldgen 视图 + 世界接线）、
`python tools/check_boot_order.py`（启动顺序 + 生成后每个方块ID都能解析）、
`python tools/check_registry.py`（注册表数据与状态解码）、
`python tools/check_chunk_codec.py`（区块编码往返）、
`python tools/check_chunk_storage.py`（区块存储与存档格式）、
`python tools/check_terrain_shape.py`（density 地形 vs 真原版区块 + 后端选择）、
`python tools/check_chunk_tickets.py`（等级模型 vs 洪泛参考 + wiki 表 + 玩家方格 + 超时）、
`python tools/check_chunk_lifecycle.py`（自动加载卸载 + 卸载即保存 + LRU 不丢数据 +
读后写保护 + 强制加载跨重启）、
`python tools/check_region_streaming_save.py`（就地扇区写入：同区域其他区块不被破坏）、
`python tools/check_server_chunk_pipeline.py`（真 Ticker + actor 线程 + 后台生成池的
加入/断开/停服全流程）。

地形两个后端各自的验收：`check_terrain_shape.py`（vanilla，逐格对
`assets/vanilla_chunks/` 的 `before`）和 `check_terrain_equivalence.py`
（legacy，向量化填充 vs 朴素循环参考实现）。两者都把自己钉在对应后端上。

## 区块加载：等级、自动加载卸载与流式保存

一个区块**该不该在内存里**由 ticket 决定，不由"它最近被访问过"决定。完整设计记录见
`world/DESIGN-chunk-loading.md`，这里只放结论：

- **等级**（`chunk_ticket.py`）。ticket 把它所在方格的等级钉住，向外每格 +1，封顶 44。
  等级模型有**闭式解**（Chebyshev 度量 + 凸区域），所以没有洪泛、没有需要维护一致性的
  传播状态：`level(c) = min_t(t.level + max(0, chebyshev(c, t.pos) - t.radius))`。
  实测 11.8–21 µs 一次查询（6 张 ticket）。等级 ≤31 实体 tick、32 方块 tick、
  33 Border、≥34 不可达；**≤33 就必须在内存里**。
- **自动加载卸载**（`chunk_lifecycle.py`）。每个玩家两张 ticket：
  `player_loading`（等级 31，半径 = 视野距离）负责加载，
  `player_simulation`（等级 31，半径 = 模拟距离）负责 ticking 与维度保活。
  因为只有后者带 SIMULATION flag，视野距离再大也不会把 ticking 区域拉宽 ——
  实体 tick 是 `2s+1` 方格，方块 tick 是 `2s+3` 方格，加载区是 `2(r+2)+1` 方格。
- **流式保存**（`chunk_saver.py`）。卸载时先把区块的 ownership **交给**保存线程，
  成功了才从内存移除；actor 线程不做磁盘 I/O。已落盘且未被改动的区块不会重写
  （`Chunk.is_persisted`），所以"走一圈"不等于重写整片视野。读同一个坐标会先等
  在途保存落盘（`flush_key`），避免读到旧副本再把它写回去。
- **接口**。`World.load_chunk/unload_chunk/forceload/unload_chunk_now`、`Dimension.add_ticket/remove_ticket/
  get_load_level/get_load_type/ticks_entities/ticks_blocks/unload_chunk`、
  `ChunkManager.set_eviction_handler/set_pin_predicate`，控制台 `/chunks`。
  `PERSIST` 的 ticket（`/chunks forceload`）写进 `forced_chunks.json` 并在下次启动恢复。
  其它线程想改 ticket 必须发消息（`TicketMsg`，由 actor 线程执行）——
  ticket 表是单属主状态，控制台直接改它会在 tick 里炸出
  "dictionary changed size during iteration"（free-threaded 构建下真会发生）。

`ChunkManager` 的两条硬规则：离开内存前必须先问 `eviction_handler`（返回 False 就留下，
没保存的脏区块比晚卸载的区块糟），以及被票覆盖的区块永不被 LRU 驱逐
（`max_chunks` 只是兜底阀门 —— 驱逐一个正在被看的区块只会让下一 tick 再加载一次）。

## 光照

天光和方块光由一个泛洪引擎产生，落在每个 `ChunkSection` 自带的那两条 2048 字节
打包 nibble 数组上 —— 也就是 `anvil.py` 落盘、`chunk_codec.py` 发出去用的同一份
数据。没有第二份光场。

| 文件 | 职责 |
|---|---|
| `light_layout.py` | 只做格式：打包 nibble ↔ 4096 格、section/格索引换算。只有它知道"一个 section 是 2048 字节" |
| `light_region.py` | 3×3 patch 的组装与回写。九宫格缺一个就 `ready=False`，不做算法。另出 `air_sections`（哪些 section 是纯空气，内核据此跳过扫描） |
| `light_engine.py` | 三个 numba 内核（`relight_region` / `update_block_light` / `update_sky_column`）+ `DimensionLight` 调度（队列、每 tick 预算、脏坐标） |
| `light_query.py` | `/light` 的跨线程消息与唯一的生产者函数（控制台发、actor 答） |

规则只有三条：`进入邻居后的等级 = level - max(opacity, 1)`；天光特例
`level == 15 且朝下 且 opacity == 0 -> 15`；`opacity == 15` 该方向归 0。
不透明度取 `assets/blocks.json` 的 `opacity` 字段（只有 0/1/15 三档），
**不用 `state_flags`** —— 这份数据里遮挡位全为 0，理由写在
`registry/blocks.py` 的 `light_opacity_table()` docstring 里。

**线上格式的 mask 按"光 section"索引，不是按区块 section。** 客户端的光 section 窗口
比方块 section 窗口上下各多一格（`minSectionY - 1` 到 `maxSectionY + 1`），所以协议里
区块 section `s` 落在 mask 的 **bit `s + 1`**，bit 0 是世界中最低 section 下面那 16 层
（协议原话："the least significant bit is for blocks 16 blocks to 1 block below the min
world height"）。写错这一位不会报错、在平的 fixture 上也看不出来，但客户端的整片光场
会**下移 16 格**：地表显示的是上面那个 section 的光（地表以上全是 15，于是整片地表没有
任何阴影），火把的光池出现在火把下方 16 格。`tools/check_chunk_codec.py` 按协议把包
解析回"客户端最终拿到什么"，逐 section 与区块自己的光比对，并把旧的排法当反例断言。

四个 mask 里这个编解码器只用三个：天光数组（`empty sky` 一个 bit 都不写 —— 没有 bit
就是 15，全 0 的 section 发真实数组，因为这一位的语义两处资料互相矛盾），方块光数组，
以及 `empty block`（1 bit = 该 section 全 0）。

**3×3 patch。** 点亮一个区块要读九个邻居的方块数据、只写回中心。跨边界的光一次
算对，邻居之后点亮时会在自己的边界写出同样的值（同样的方块、同样的规则），所以
是幂等的。九宫格里缺任何一个，这次 patch 就不做、区块留在队列里 —— 拿空气当邻居
会点亮一片虚假的天光，而且没人会再去修它。

**接入点。** `ChunkLifecycle._request_light` 在"区块变可读"的两处入队
（`on_chunk_ready`，以及 untracked 里发现从磁盘读回的区块时）；
`DimensionActor._on_tick` 把 `self.light.tick()` 插在 `spawner.tick()` 之后、
`entity_tracker.pump()` 之前。`chunk_streamer.pump()` 每 tick 取一次
`DimensionLight.take_dirty()`，向看过该区块的玩家发 `CLightUpdate`。

> 这两处**必须都真的调用**，否则整条链路是装饰品：流式器只要 `is_generated` 就把
> 区块发出去，而 `ChunkSection` 的默认天光是全 15，编码器对
> `light_populated=False` 的区块会退回**高度图近似**（地表以上 15、以下 0、没有
> 方块光）。2026-10-07 那次联机就是 `on_chunk_ready` 少了那一行：存档里 631 个
> 区块有 576 个从未被 patch 碰过，客户端收到的 381 个区块里 377 个是那张高度图，
> 整局只有 2 个 `CLightUpdate`——玩家看到的是"凡是头顶有方块就全黑，不管那方块
> 透不透光"。现在由 `tools/check_server_chunk_pipeline.py` 钉住（视野内区块全部
> 点亮、点亮前发出的区块全部被 `CLightUpdate` 补上）。

**实测**（`py -3.14t`，free-threaded 3.14.8；`tools/bench_light.py`）：

| | |
|---|---|
| 一次真实 3×3 patch | best **3.48** / p50 **4.30 ms**（build 0.75–0.88 + 内核 1.33–1.59 + write_back 0.54–0.70） |
| 内核（同一 region，带 section 表） | best **1.330** / p50 **1.591 ms**（不带表 best 2.380 / p50 2.733） |
| 增量路径的 region build（含解包 432 个数组） | best **1.21–1.96** / p50 **1.5–2.2 ms**（改前 best 5.73 / p50 7.61） |
| 单列 16×16×384 天光+方块光 | **0.236 ms** |
| 单次方块变更 | 裸内核 best 0.134 / p50 0.26–0.36 ms；走 `DimensionLight` best 0.002 / p50 0.77 ms |
| 一次 `CLightUpdate`（真光区块） | **16,438 B**，zlib 后 **124 B**；8 条天光数组、0 条方块光数组 |
| 首次 JIT（`cache=True` 冷编译） | **25–34 s**，**在启动期**（`core/server.py._warm_up_kernels`）；命中缓存后 **0.51 s**，第二次 2.6 ms |

> 上表 best/p50 的区间是同一台机器上连跑两次的范围（`LightRegion.build` 的
> 方块拷贝那一项方差最大，p50 在 0.84–1.61 之间跳）。改动前后同口径对比：
> 内核 3.16/4.52 → **2.38–2.67/2.49–2.80 ms**（去掉每次调用 `np.empty` 的
> 3 × 3.5 MB，见 `world/light_engine.py` 的 `LightQueues`）；解包走 numba 后
> 432 次从 ~6.5 ms 降到 ~0.6 ms；**内核再 2.380 → 1.330 ms**（跳过全空气
> section，见下）。**一次 50 ms tick 能装下的方块编辑从 43 条升到 52–65 条。**

### 跳过全空气 section（收益最大的一项）

真实地形中心区块 24 个 section 里 **19 个是均匀的**（底下 4 个纯石头、上面
15 个纯空气），3×3 的 216 个 chunk-section 里 **138 个是纯空气**——占 884,736
格的 64%，而那两趟线性扫描（列顶/发光体扫描 + 每列天光填充）占内核的 81%，
在纯空气里什么也找不到。

`LightRegion.air_sections` 就是这两趟扫描缺的元数据，来自
`ChunkSection.non_air_count`（`chunk.py:152`，`set_block` 维护、
`recompute_non_air_count` 修复，编解码器本来就在信它）。内核拿它做两件事：

1. **纯空气 section 直接跳过扫描**——空气既没有 opacity 也不发光；
2. **天空光"预置 15"而不是逐格写**——但那只在"这个 section 确实在它所有列顶
   之上"时才成立，`is_empty` 本身说不出这一点（洞穴上方放一个方块，就能让一个
   全空气 section 在两 patch 之间变成逐格光）。所以这个判据由 pass 1 自己的
   `tops` 网格现算：每 section 256 次比较，216 个 section 共 **2 µs**；判据不成立
   就退回逐格填充。**这一条写错过一次**——盲信 `is_empty` 会让 16 格留在 15
   （正确答案 14），`tools/check_light_engine_incremental.py` 抓到的。

实测同 fixture、同机器，best of 40：整个 patch 内核 **2.29 → 1.33 ms（−42%）**，
两层光 **逐字节相同**（9 个区块 15,962,624 格，`tools/check_light_batch.py`）。
表是可选的：`NO_SECTION_TABLE` 表示"没有表，全扫"，除
`DimensionLight._light_chunk` 之外所有调用方（工具、`warm_up`）都传它，所以
`tools/check_light_engine.py` 的 oracle 对比仍然是**通用**内核。

> JIT 那条要看清楚：`cache=True` 的缓存键包含源码，**改一次 `light_engine.py` 就要重编一次**
> （25–34 s）。所以"首次启动慢"实际是"每次改内核源码后的第一次启动慢"，稳态启动
> 只多 0.51 s。

`light-budget-ms = 4.0` 配 p50 约 4.3 ms 的 patch，实际**每 tick 点亮约 0.9 个区块**，
不是计划里估的四个。`light-chunks-per-tick` 只是上限，时钟才是限制；要更快就调预算。

**已知缺口**：

- **未加载邻居不重算**。加载边界上的光可能比 vanilla 暗一档，直到那一侧被点亮
  （vanilla 有 `LevelLightEngine` 的跨区块任务，本轮没有）。而且九宫格里任何一个
  没点亮时，增量路径会把这次方块变更**降级成一次全量 patch**（计入
  `stats()['dim_light_fallbacks']`）—— 因为未点亮区块的数组是构造默认值，读它会把
  假光写进中心区块。所以"增量 == 全量"的测试必须是 5×5 fixture，3×3 里增量路径
  根本不会跑。
- **一个 tick 内连发多次方块变更然后只 drain 一次，不保证等于冷跑。** 实测 13 次
  编辑零 tick 间隔堆在一起会差 1687 格（全在方块光层，增量**偏亮**）；每次编辑之间
  有 tick 就是 0 差异。这是调度不变量（真实玩家一 tick 做不到），不是传播错误，
  但别把它写成断言。
- **存量存档的 `sky_light` 是"全 15 默认值"**，读回来会被当成真的。本轮加了
  `light_populated` 标记让编解码两侧都不信它，但**不做自动重算** —— 全图重算是一次
  显式命令：`/light relight all`（一个区域用 `/light relight <cx> <cz> <radius>`）。
  请求只是往引擎队列里放坐标，patch 在每 tick 的光预算里慢慢跑。
- **光照数据的存档迁移、多线程 patch、平滑光照/AO、维度环境光**：都不做，见
  `LIGHT_PLAN.md` §十一。

## 已知缺口

- **地表规则有 1.9% 的单元格与 vanilla 对不上**（7 个原版区块里 2 个逐格完全一致）。
  残留集中在恶地的陶瓦带、洞穴顶部的深层沙子、以及下界的基岩带。
  `tools/check_surface_rules.py` 把每个区块的上限记在 `RESIDUALS` 里，回归会让
  数字变大。见 `world/vanilla/README.md`。
- **两个 preset 的地形还复现不了 vanilla**：nether 的实心主体算成空的，end 走
  的是 vanilla 另一套生成器（`EndChunkGenerator`）而不是 noise router。见
  `world/DESIGN-density-preset-gaps.md`。
- **水是近似的**：vanilla 用 aquifer 决定地下水往哪流，aquifer 没有移植。
  `world/vanilla/terrain.py` 用的是"每列从最上面的石头填到海平面"，海对了、
  洞穴保持干的。同一个文件里有说明。
- **洞穴和地貌特征没有原版基准**（Pumpkin 的测试数据只有 noise 和 surface 两阶
  段），所以 `check_carvers.py` / `check_features.py` 查的是确定性、分发和可重算
  的分布，不冒充 parity。

## 分层：谁负责什么

地形生成分四层，按"算法"和"驱动"分开：

```
world/vanilla/    算法。采样出结果（密度场、生物群系 id、地表规则…），不认识 Chunk
world/feature/    驱动。把上面的结果写进 Chunk，管生命周期和降级
world/legacy/     自研算法（Perlin 高度图、自研矿脉、自研隧道洞穴）
world/generator.py 工厂 + 共用基类。选后端、定管线顺序，没有算法
```

`world/vanilla/` 只算不写，`world/feature/` 只写不算，所以换一个地形算法不必
动装饰层。反过来，装饰层也不知道脚下的石头是 Perlin 还是 density 来的。

管线的顺序是 `BaseGenerator._generate_chunk_locked` 里的：

```
_generate_terrain   后端自己的活（唯一抽象方法）
biomes              world/feature/biomes.py
surface             world/feature/surface.py
_generate_ores      钩子，只有 legacy 后端实现
carvers             world/feature/carvers.py
features            world/feature/features.py
```

顺序是有讲究的：后三层都要问生物群系"你是谁"，所以 biomes 先跑；矿石夹在
地表和洞穴之间，是为了让矿脉既不落在马上要被泥土盖住的位置，也不会被洞穴
掏空后吊在半空。细节见 `world/feature/README.md`。

`_generate_ores` 是钩子而不是一层，是因为 vanilla 后端的矿石是
`world/feature/features.py` 里的 `minecraft:ore_vein` 地貌特征。两个都跑会把
每一条矿脉放两遍。

**线程安全不是可有可无的**：`NoiseRouter` 的 `FlatCache`、生物群系树的
last-result 缓存都带跨采样的状态，两个线程共用一个 generator 会把彼此的
缓存搅在一起，地形沿 Z 悄悄拉丝、不抛异常。`BaseGenerator` 在整个
`generate_chunk` 外面加了锁，但真正让锁无争用的是：后台生成的 worker 不再
共享 generator —— multiprocess 模式每个进程一套，thread 模式（
`chunk-generation-mode = "thread"`）每个池线程一套线程本地 generator（见
`world/chunk_gen_worker.py`），同步路径用维度自己的那个。锁留着兜底：
无争用的锁相对约一秒一个区块的开销测不出来，却一次性护住全部七个缓存。

## 主要功能

### 1. 噪声生成 (legacy/noise.py)

使用numba JIT编译加速的Perlin/Simplex噪声生成：

```python
from world.legacy.noise import PerlinNoise, get_chunk_heightmap
import numpy as np

# 初始化噪声生成器
noise = PerlinNoise(
    seed=12345,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    scale=256.0
)

# 单点噪声
value = noise.noise2d(100.0, 200.0)

# 快速获取区块高度图
heightmap = get_chunk_heightmap(chunk_x, chunk_z, seed=12345)
```

### 2. 区块管理 (chunk.py)

```python
from world.chunk import Chunk, ChunkPos, BlockState

# 创建区块
pos = ChunkPos(0, 0)
chunk = Chunk(position=pos)

# 设置方块
chunk.set_block(x, y, z, BlockState.GRASS)

# 获取方块
block_id = chunk.get_block(x, y, z)
```

### 3. 世界生成器 (generator.py)

`create_generator` 按 `[world] generator` 选后端；`backend=` 可以显式覆盖。

```python
from world.generator import create_generator, GeneratorSettings, GeneratorType

# 配置生成器。preset 决定海平面/默认方块/流体从哪里来：
#     "overworld" (sea 63, stone/water), "nether" (32, netherrack/lava),
#     "end" (0, end_stone/air) —— 取自 assets/chunk_gen_settings.json
settings = GeneratorSettings(
    generator_type=GeneratorType.NORMAL,
    seed=54321,
    preset="overworld",
    generate_features=True,
)

# 走 config.toml 里的后端（默认 "vanilla"）
generator = create_generator(settings)

# 或者指定后端
legacy = create_generator(settings, backend="legacy")

# 生成区块
chunk = Chunk(position=ChunkPos(0, 0))
generator.generate_chunk(chunk)
```

两个后端只差地形怎么长出来，装饰层一样。见 `world/legacy/README.md` 和
`world/feature/README.md`。

### 4. 维度管理 (dimension.py)

```python
from world.dimension import Dimension, DimensionType

# 创建维度
dim = Dimension(
    dimension_type=DimensionType.OVERWORLD,
    world_seed=12345
)

# 加载区块
chunk = dim.load_chunk(x, z)

# 获取/设置方块
block_id = dim.get_block(world_x, world_y, world_z)
dim.set_block(world_x, world_y, world_z, block_id)
```

### 5. 世界管理 (world.py)

```python
from world import World

# 创建世界
world = World(
    name="my_world",
    seed=12345
)

# 加载区块
world.load_chunk(x, z, dimension=DimensionType.OVERWORLD)

# 保存世界
world.save()

# 获取世界信息
info = world.get_world_info()
```

### 6. 序列化 (serializer.py)

```python
from world.serializer import (
    create_world_directory,
    list_worlds,
    delete_world,
    RegionSerializer,
)

# 创建新世界目录
world_dir = create_world_directory("new_world", Path("worlds"))

# 列出所有世界
worlds = list_worlds(Path("worlds"))

# 保存/加载世界
from world import World
world = World.load(world_dir)
world.save()
```

## 性能优化

噪声模块使用以下优化技术：

1. **Numba JIT编译**: 所有噪声函数使用 `@njit` 装饰器进行JIT编译
2. **并行处理**: 使用 `@njit(parallel=True)` 和 `prange` 进行并行计算
3. **NumPy向量化**: 批量操作使用NumPy数组而非Python循环
4. **缓存**: 使用 `cache=True` 缓存编译后的函数

性能基准（Intel i7, 8核）:

- 单点噪声: ~50,000 ops/sec
- 区块高度图生成: ~500 chunks/sec

## 依赖

```
numpy>=1.24.0
numba>=0.58.0
```

## 测试

运行测试：

```bash
python test_world.py
```

测试覆盖：

- 噪声生成
- 区块数据结构
- 世界生成
- 维度管理
- 世界保存/加载
- 性能基准
