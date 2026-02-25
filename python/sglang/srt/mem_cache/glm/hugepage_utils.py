import os
from typing import Optional

GLM_HICACHE_SHM_DIR = os.environ.get("GLM_HICACHE_SHM_DIR", "/dev/shm")

def _align_up(size: int, alignment: Optional[int] = None) -> int:
    if alignment is None:
        alignment = _hugepage_size()
    if alignment <= 0:
        return size
    return ((size + alignment - 1) // alignment) * alignment

def _hugepage_enabled() -> bool:
    return GLM_HICACHE_SHM_DIR != "/dev/shm"

def _hugepage_size() -> int:
    # NOTE: temporarily only support 2MiB page
    return 2 * 1024 * 1024
