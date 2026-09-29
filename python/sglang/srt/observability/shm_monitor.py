import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShmUsage:
    total_bytes: int
    used_bytes: int
    available_bytes: int

    @property
    def usage_ratio(self) -> float:
        return self.used_bytes / self.total_bytes if self.total_bytes else 0.0


def read_shm_usage(path: str = "/dev/shm") -> Optional[ShmUsage]:
    if os.name != "posix":
        return None

    stat = os.statvfs(path)
    block_size = stat.f_frsize or stat.f_bsize
    total_bytes = block_size * stat.f_blocks
    used_bytes = total_bytes - block_size * stat.f_bfree
    available_bytes = block_size * stat.f_bavail
    return ShmUsage(
        total_bytes=total_bytes,
        used_bytes=max(0, used_bytes),
        available_bytes=max(0, available_bytes),
    )


def start_shm_monitor_thread(
    update_metrics: Callable[[ShmUsage], None], interval: float = 30.0
) -> Optional[threading.Thread]:
    def monitor() -> None:
        failure_logged = False
        while True:
            try:
                usage = read_shm_usage()
                if usage is not None:
                    update_metrics(usage)
                failure_logged = False
            except Exception:
                if not failure_logged:
                    logger.warning(
                        "Failed to update shared-memory filesystem metrics",
                        exc_info=True,
                    )
                    failure_logged = True
            time.sleep(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    try:
        thread.start()
    except RuntimeError:
        logger.warning("Failed to start shared-memory monitor thread", exc_info=True)
        return None
    return thread
