"""
Server Threading Model

Crotalus uses multiple threads:
- Main thread: Server startup/shutdown
- Network thread: Async I/O for client connections (asyncio)
- Ticker + dimension actors: Game logic updates (core/ticker.py)
- Scheduled tasks thread: Delayed/periodic tasks
- RCON thread (optional): Remote console
"""

import asyncio
import threading
import time
from typing import Callable, Optional
from loguru import logger
import psutil

class ServerThread:
    """Base class for server threads."""
    
    def __init__(self, name: str):
        self.name = name
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """Start the thread."""
        if self._thread and self._thread.is_alive():
            logger.warning(f"{self.name} is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        logger.info(f"{self.name} started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop the thread."""
        if not self._thread or not self._thread.is_alive():
            return
        
        self._running = False
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        
        if self._thread.is_alive():
            logger.warning(f"{self.name} did not stop in time")
        else:
            logger.info(f"{self.name} stopped")
    
    def _run(self) -> None:
        """Thread main loop. Override in subclass."""
        raise NotImplementedError
    
    @property
    def is_alive(self) -> bool:
        """Check if thread is running."""
        return self._thread is not None and self._thread.is_alive()


class TickThread(ServerThread):
    """Game tick thread - runs at 20 ticks per second."""
    
    def __init__(self, tick_rate: int = 20):
        super().__init__("TickThread")
        self.tick_rate = tick_rate
        self._tick_count = 0
        self.tick_interval = 1.0 / tick_rate
        self._last_tick_time = time.time()
        self._tps = tick_rate  # Ticks per second
        self._mspt = 0.0  # Milliseconds per tick
        
        # Tick callbacks
        self._tick_callbacks: list[Callable[[], None]] = []
    
    def register_tick_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called each tick."""
        self._tick_callbacks.append(callback)
    
    def _run(self) -> None:
        """Main tick loop."""
        logger.info(f"Tick thread started (target: {self.tick_rate} TPS)")
        
        while self._running:
            start_time = time.time()
            
            # Execute tick callbacks
            for callback in self._tick_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.exception(f"Error in tick callback: {e}")
            
            
            # Calculate timing
            elapsed = time.time() - start_time
            self._mspt = elapsed / 1000

            self._tick_count += 1

            # Calculate actual TPS
            if self._tick_count % 20 == 0:
                self._tick_count = 0
                time_since_last = time.time() - self._last_tick_time
                self._tps = 20 / time_since_last if time_since_last > 0 else self.tick_rate
                self._last_tick_time = time.time()
            
            # Sleep for remaining time
            sleep_time = self.tick_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
            elif elapsed > 10: # default sleep time <= 0 and elapsed time > 10 seconds
                logger.warning(f"Can't keep up! Is server overloaded? Running {0-sleep_time:.2f}s behind.")

        logger.info(f"Tick thread stopped")
    
    
    @property
    def tps(self) -> float:
        """Get current ticks per second."""
        return self._tps
    
    @property
    def mspt(self) -> float:
        """Get milliseconds per tick."""
        return self._mspt


class TaskThread(ServerThread):
    """Scheduled tasks thread - runs delayed and periodic tasks."""
    
    def __init__(self):
        super().__init__("TaskThread")
        self._tasks: list[tuple[float, Callable[[], None]]] = []  # (execute_time, callback)
        self._lock = threading.Lock()
    
    def schedule(self, delay: float, callback: Callable[[], None]) -> None:
        """Schedule a task to run after delay seconds."""
        execute_time = time.time() + delay
        with self._lock:
            self._tasks.append((execute_time, callback))
            self._tasks.sort(key=lambda x: x[0])  # Sort by execution time
    
    def schedule_repeating(self, interval: float, callback: Callable[[], None]) -> None:
        """Schedule a task to run repeatedly."""
        def repeating_wrapper():
            callback()
            self.schedule_repeating(interval, callback)
        
        self.schedule(interval, repeating_wrapper)
    
    def _run(self) -> None:
        """Main task loop."""
        logger.info("Task thread started")
        
        while self._running:
            current_time = time.time()
            
            with self._lock:
                # Execute due tasks
                due_tasks = [(t, cb) for t, cb in self._tasks if t <= current_time]
                self._tasks = [(t, cb) for t, cb in self._tasks if t > current_time]
            
            for _, callback in due_tasks:
                if not self._running:
                    break
                try:
                    callback()
                except Exception as e:
                    logger.exception(f"Error in scheduled task: {e}")
            
            # Sleep briefly
            self._stop_event.wait(0.05)
        
        logger.info("Task thread stopped")
    
    @property
    def pending_tasks(self) -> int:
        """Get number of pending tasks."""
        with self._lock:
            return len(self._tasks)


class SystemMonitorThread(ServerThread):
    """System monitoring thread - monitors CPU, memory, etc."""
    
    def __init__(self, interval: float = 1.0):
        super().__init__("SystemMonitor")
        self.interval = interval
        self._cpu_usage = 0.0
        self._memory_usage = 0.0
        self._process = psutil.Process()
    
    def _run(self) -> None:
        """Main monitoring loop."""
        logger.info("System monitor started")
        
        while self._running:
            try:
                # CPU usage (percentage)
                self._cpu_usage = self._process.cpu_percent()
                
                # Memory usage (MB)
                mem_info = self._process.memory_info()
                self._memory_usage = mem_info.rss / (1024 * 1024)
                
            except Exception as e:
                logger.warning(f"Error monitoring system: {e}")
            
            self._stop_event.wait(self.interval)
        
        logger.info("System monitor stopped")
    
    @property
    def cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return self._cpu_usage
    
    @property
    def memory_usage(self) -> float:
        """Get memory usage in MB."""
        return self._memory_usage


class ThreadManager:
    """Manages all server threads."""
    
    def __init__(self):
        self.tick_thread: Optional[TickThread] = None
        self.task_thread: Optional[TaskThread] = None
        self.monitor_thread: Optional[SystemMonitorThread] = None
    
    def start_all(self) -> None:
        """Start all threads."""
        # Task thread (must start first)
        self.task_thread = TaskThread()
        self.task_thread.start()
        
        # Game ticks run on core.ticker.Ticker (started by the server)

        # System monitor
        self.monitor_thread = SystemMonitorThread()
        self.monitor_thread.start()
    
    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all threads."""
        if self.monitor_thread:
            self.monitor_thread.stop(timeout)
        
        if self.tick_thread:
            self.tick_thread.stop(timeout)
        
        if self.task_thread:
            self.task_thread.stop(timeout)
    
    def get_status(self) -> dict:
        """Get status of all threads."""
        return {
            "tick_thread": {
                "alive": self.tick_thread.is_alive if self.tick_thread else False,
                "current_tick": self.tick_thread.current_tick if self.tick_thread else 0,
                "tps": self.tick_thread.tps if self.tick_thread else 0,
                "mspt": self.tick_thread.mspt if self.tick_thread else 0,
            },
            "task_thread": {
                "alive": self.task_thread.is_alive if self.task_thread else False,
                "pending_tasks": self.task_thread.pending_tasks if self.task_thread else 0,
            },
            "system": {
                "cpu_usage": self.monitor_thread.cpu_usage if self.monitor_thread else 0,
                "memory_usage": self.monitor_thread.memory_usage if self.monitor_thread else 0,
            }
        }


# Global thread manager
thread_manager = ThreadManager()
