"""
Server Configuration Module

Handles loading and accessing server configuration from config.toml
"""

import os
from pathlib import Path
from typing import Any, Optional

import toml


class Config:
    """Server configuration manager."""
    
    _instance: Optional["Config"] = None
    _config: dict = {}
    _config_path: Path = Path("config.toml")
    
    def __new__(cls) -> "Config":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance
    
    def __init__(self):
        """Initialize the config."""
        if not self._loaded:
            self.load()
    
    def load(self, path: Optional[str] = None) -> None:
        """Load configuration from TOML file."""
        if path:
            self._config_path = Path(path)
        
        # Try multiple locations
        search_paths = [
            self._config_path,
            Path("config.toml"),
            Path(__file__).parent.parent / "config.toml",
        ]
        
        config_file = None
        for p in search_paths:
            if p.exists():
                config_file = p
                break
        
        if config_file is None:
            # Create default config
            self._create_default()
            return
        
        with open(config_file, "r", encoding="utf-8") as f:
            self._config = toml.load(f)
        
        self._loaded = True
        print(f"Configuration loaded from {config_file}")
    
    def _create_default(self) -> None:
        """Create default configuration."""
        # This would typically write a default config file
        # For now, use hardcoded defaults
        self._config = {
            "server": {
                "server-ip": "",
                "server-port": 25565,
                "server-name": "Crotalus",
                "motd": "A Minecraft Server",
                "max-players": 20,
                "view-distance": 10,
                "spawn-radius": 4,
                "simulation-distance": 10,
                "player-idle-timeout": 0,
                "level-name": "world",
                "level-seed": "",
                "level-type": "minecraft:normal",
                "generate-structures": True,
                "hardcore": False,
                "difficulty": "easy",
                "gamemode": "survival",
                "spawn-monsters": True,
                "spawn-protection": 16,
                "allow-nether": True,
                "pvp": True,
                "allow-flight": False,
                "online-mode": True,
                "enable-query": False,
                "enable-rcon": False,
                "network-compression-threshold": 256,
                "use-native-transport": True,
                "sync-chunk-writes": True,
                # True only makes sense when the server initializes signed chat
                # sessions for every player; ours never does, and enforcement
                # turns each Player Chat into a client validation error.
                "enforce-secure-profile": False,
                "prevent-proxy-connections": False,
                "white-list": False,
                "enforce-whitelist": False,
                "player-chat-packet": True,
                "enable-command-block": False,
                "function-permission-level": 2,
                "op-permission-level": 4,
                "resource-pack": "",
                "resource-pack-sha1": "",
                "require-resource-pack": False,
                "log-ips": True,
                "broadcast-console-to-ops": True,
                "max-tick-time": 60000,
                "max-chained-neighbor-updates": 1000000,
                "rate-limit": 0,
                # Registry extensions; see registry/README.md.
                "plugins-dir": "plugins",
                "datapacks-dir": "datapacks",
            },
            "world": {
                # Terrain generator backend; see config.toml. Validated in
                # world.generator.create_generator, not here - this class is a
                # dumb reader and does not know what a backend is.
                "generator": "vanilla",
                # Workers for background chunk generation (processes in the
                # "multiprocess" work mode, threads in "thread" mode).
                # 0 = auto (logical core count, capped at 8); see config.toml.
                "chunk-generation-workers": 0,
                # Work mode for background chunk generation; validated in
                # world.chunk_gen_thread.ChunkGeneratorThread.configure, not
                # here. "thread" or "multiprocess"; see config.toml.
                "chunk-generation-mode": "multiprocess",
            },
            "query": {
                "port": 25565,
                "generate-stats": True,
            },            "rcon": {
                "port": 25575,
                "password": "",
            },
            "spawning": {
                # Natural mob spawning, the runtime half. Vanilla spells this
                # as the `doMobSpawning` gamerule; there is no gamerule engine
                # yet, so the config carries it.
                "natural-mob-spawning": True,
                # Ticks between spawn rounds. Vanilla is 20; see
                # entity/spawner.py for why this is a cadence rather than a
                # work factor.
                "spawn-interval-ticks": 20,
            },
            "permissions": {
                # 简易权限系统（console/permissions.py），仿原版 OP 等级 0-4。
                # operators 为 玩家名 -> 等级；未列出的玩家用 default-level。
                # 每次权限检查实时读取，/reload 后立即生效。
                "default-level": 0,
                "operators": {},
            },
            "advanced": {
                "debug": False,
                "entity-broadcast-range-percentage": 100,
                "server-authoritative-movement": True,
                "movement-correction-threshold": 0.5,
                "bug-report-link": "",
            },
        }
        self._loaded = True
        print("Using default configuration")
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to TOML file."""
        save_path = Path(path) if path else self._config_path
        
        with open(save_path, "w", encoding="utf-8") as f:
            toml.dump(self._config, f)
        
        print(f"Configuration saved to {save_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation.
        
        Examples:
            config.get("server.server-port")
            config.get("server.max-players")
            config.get("query.port", 25565)
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a config value using dot notation."""
        keys = key.split(".")
        target = self._config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    # Convenience properties
    @property
    def server_ip(self) -> str:
        return self.get("server.server-ip", "")
    
    @property
    def server_port(self) -> int:
        return self.get("server.server-port", 25565)
    
    @property
    def max_players(self) -> int:
        return self.get("server.max-players", 20)
    
    @property
    def view_distance(self) -> int:
        return self.get("server.view-distance", 10)
    
    @property
    def spawn_radius(self) -> int:
        return self.get("server.spawn-radius", 4)

    @property
    def simulation_distance(self) -> int:
        return self.get("server.simulation-distance", 10)

    @property
    def natural_mob_spawning(self) -> bool:
        return bool(self.get("spawning.natural-mob-spawning", True))

    @property
    def spawn_interval_ticks(self) -> int:
        return self.get("spawning.spawn-interval-ticks", 20)

    @property
    def level_name(self) -> str:
        return self.get("server.level-name", "world")
    
    @property
    def level_seed(self) -> str:
        return self.get("server.level-seed", "")
    
    @property
    def level_type(self) -> str:
        return self.get("server.level-type", "minecraft:normal")
    
    @property
    def difficulty(self) -> str:
        return self.get("server.difficulty", "easy")
    
    @property
    def gamemode(self) -> str:
        return self.get("server.gamemode", "survival")
    
    @property
    def online_mode(self) -> bool:
        return self.get("server.online-mode", True)
    
    @property
    def white_list(self) -> bool:
        return self.get("server.white-list", False)
    
    @property
    def motd(self) -> str:
        return self.get("server.motd", "A Minecraft Server")
    
    @property
    def compression_threshold(self) -> int:
        return self.get("server.network-compression-threshold", 256)
    
    @property
    def enable_rcon(self) -> bool:
        return self.get("server.enable-rcon", False)
    
    @property
    def rcon_port(self) -> int:
        return self.get("rcon.port", 25575)
    
    @property
    def rcon_password(self) -> str:
        return self.get("rcon.password", "")
    
    @property
    def enable_query(self) -> bool:
        return self.get("server.enable-query", False)
    
    @property
    def query_port(self) -> int:
        return self.get("query.port", 25565)
    
    @property
    def debug(self) -> bool:
        return self.get("advanced.debug", False)

    @property
    def server_authoritative_movement(self) -> bool:
        """Whether player movement packets are checked against the world.

        On (the default), `physics.player.PlayerMoveValidator` judges the path a
        movement packet claims and sends a correction when the collision engine
        says it is impossible. Off means the packet is taken at face value,
        which is what this server did before physics existed.

        Deliberately NOT `allow-flight`: that key exists in `config.toml` and in
        the defaults above, but nothing in this codebase reads it, and it means
        "may a player hover", which is a different question from "is this step
        through a wall".
        """
        return bool(self.get("advanced.server-authoritative-movement", True))

    @property
    def movement_correction_threshold(self) -> float:
        """Blocks of deviation below which a movement packet is never corrected.

        The client predicts its own position and the server re-derives it from
        the same numbers with different rounding, so a threshold is what keeps a
        player walking along a wall from being pulled back every tick. Read by
        `physics.player.PlayerMoveValidator`.
        """
        return float(self.get("advanced.movement-correction-threshold", 0.5))

    @property
    def plugins_dir(self) -> str:
        """Directory scanned for registry plugin modules."""
        return self.get("server.plugins-dir", "plugins")

    @property
    def generator_backend(self) -> str:
        """Which terrain generator builds new chunks: "vanilla" or "legacy".

        Read by `world.generator.create_generator`. An unrecognised value is
        left alone here and rejected there, so that the error can name the
        legal backends.
        """
        return self.get("world.generator", "vanilla")

    @property
    def chunk_gen_workers(self) -> int:
        """Workers for background chunk generation.

        Processes in the "multiprocess" work mode, threads in "thread" mode
        (`chunk_gen_mode`). 0 (or missing) means auto: the machine's logical
        core count, capped at 8 - the pipeline stalls on memory often enough
        that more workers keep helping past the physical core count (measured
        3.7x at 8 workers on a 4-core/8-thread box, see
        tools/bench_chunk_parallelism.py). A positive value is taken
        literally, clamped into [1, 32] so a typo cannot fork 400 children.
        """
        import os

        raw = self.get("world.chunk-generation-workers", 0) or 0
        try:
            requested = int(raw)
        except (TypeError, ValueError):
            requested = 0
        if requested <= 0:
            return max(1, min(os.cpu_count() or 4, 8))
        return max(1, min(requested, 32))

    @property
    def chunk_gen_mode(self) -> str:
        """Work mode for background chunk generation.

        "multiprocess" (default) dispatches to a pool of worker processes:
        true parallelism on a GIL build, at the cost of one interpreter per
        worker and an array payload crossing the process boundary. "thread"
        runs a ThreadPoolExecutor whose threads each build their own
        generators, so no generator lock is ever contended and the payload
        round-trip disappears: the GIL-mode ceiling (~1.0x), but the mode
        that scales on a free-threaded Python 3.13+ build. The value is only
        shaped here; it is validated where it is consumed
        (`ChunkGeneratorThread.configure` falls back to "multiprocess" and
        logs when it does not recognise it).
        """
        raw = self.get("world.chunk-generation-mode", "multiprocess")
        if not isinstance(raw, str) or not raw.strip():
            return "multiprocess"
        return raw.strip().lower()

    @property
    def datapacks_dir(self) -> str:
        """Directory scanned for datapacks that extend registry data."""
        return self.get("server.datapacks-dir", "datapacks")


# Global config instance
config = Config()
