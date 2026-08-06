from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    HostPoolGroup,
    LogicalHostPool,
    MambaPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
    NSATokenToKVPoolHost,
    NSATokenToKVPoolHostShared,
    NSATokenToKVPoolHostSharedLayerGroup,
    PoolEntry,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _unwrap_nsa_kv_pool(pool: Any):
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        NSATokenToKVPool,
    )

    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool
    return pool if isinstance(pool, NSATokenToKVPool) else None


def _nsa_hicache_layer_geometry(pool: Any, page_size: int) -> tuple:
    """Return the per-layer geometry that determines NSA host-cache bytes."""
    if pool.page_size != page_size:
        raise ValueError(
            "Target and Draft NSA HiCache page sizes must match: "
            f"pool={pool.page_size}, expected={page_size}."
        )

    use_fp8_index = bool(pool.use_fp8_index_k_cache)
    if use_fp8_index:
        quant_block_size = pool.quant_block_size
        index_row_width = pool.index_head_dim + (
            pool.index_head_dim // quant_block_size * 4
        )
        index_dtype = pool.index_k_with_scale_buffer_dtype
        index_slots_per_page = getattr(pool, "slots_per_page", page_size)
    else:
        quant_block_size = None
        index_row_width = pool.index_head_dim
        index_dtype = pool.index_k_buffer_dtype
        index_slots_per_page = page_size

    return (
        pool.store_dtype,
        pool.kv_cache_dim,
        use_fp8_index,
        quant_block_size,
        index_dtype,
        index_row_width,
        index_slots_per_page,
    )


def _nsa_hicache_bytes_per_token_per_layer(pool: Any, page_size: int) -> int:
    (
        store_dtype,
        kv_cache_dim,
        _,
        _,
        index_dtype,
        index_row_width,
        index_slots_per_page,
    ) = _nsa_hicache_layer_geometry(pool, page_size)
    main_bytes = kv_cache_dim * torch.empty((), dtype=store_dtype).element_size()
    index_bytes = (
        index_row_width
        * torch.empty((), dtype=index_dtype).element_size()
        * index_slots_per_page
        // page_size
    )
    return main_bytes + index_bytes


def _split_hybrid_hicache_budget(
    *,
    total_size_gb: float,
    mamba_full_memory_ratio: float,
    target_pool: Any,
    draft_pool: Any,
    page_size: int,
) -> tuple[float, float, float]:
    """Split an explicit hybrid HiCache budget into target, Draft, and Mamba.

    Ratio mode (``total_size_gb == 0``) intentionally keeps its existing
    per-device-pool semantics. An explicit size is a machine-total budget for
    shared NSA HiCache, so a LayerSplit Draft pool must reserve its share before
    the target host pool is allocated.
    """
    if total_size_gb <= 0:
        return 0.0, 0.0, 0.0

    mamba_size_gb = (
        total_size_gb * mamba_full_memory_ratio / (1 + mamba_full_memory_ratio)
    )
    full_size_gb = total_size_gb - mamba_size_gb

    draft_nsa_pool = _unwrap_nsa_kv_pool(draft_pool)
    if draft_nsa_pool is None or not getattr(
        draft_nsa_pool, "layer_shard_enabled", False
    ):
        return full_size_gb, 0.0, mamba_size_gb

    target_nsa_pool = _unwrap_nsa_kv_pool(target_pool)
    if target_nsa_pool is None or not getattr(
        target_nsa_pool, "layer_shard_enabled", False
    ):
        raise ValueError(
            "Draft NSA LayerSplit HiCache requires a LayerSplit target NSA pool."
        )
    if draft_nsa_pool.layer_num != 1:
        raise ValueError(
            "Draft NSA LayerSplit HiCache currently requires exactly one layer, "
            f"got {draft_nsa_pool.layer_num}."
        )
    if target_nsa_pool.size != draft_nsa_pool.size:
        raise ValueError(
            "Target and Draft NSA LayerSplit pools must have the same token "
            f"capacity: target={target_nsa_pool.size}, draft={draft_nsa_pool.size}."
        )

    target_geometry = _nsa_hicache_layer_geometry(target_nsa_pool, page_size)
    draft_geometry = _nsa_hicache_layer_geometry(draft_nsa_pool, page_size)
    if target_geometry != draft_geometry:
        raise ValueError(
            "Target and Draft NSA LayerSplit pools must have identical per-layer "
            "HiCache geometry before sharing one slot capacity."
        )

    bytes_per_layer = _nsa_hicache_bytes_per_token_per_layer(target_nsa_pool, page_size)
    target_weight = bytes_per_layer * target_nsa_pool.layer_num
    draft_weight = bytes_per_layer * draft_nsa_pool.layer_num
    target_size_gb = full_size_gb * target_weight / (target_weight + draft_weight)
    draft_size_gb = full_size_gb - target_size_gb

    # Mirror HostKVCache's page alignment so an undersized explicit budget
    # fails with a useful message before the large host tensors are allocated.
    target_size_per_token = bytes_per_layer * target_nsa_pool.layer_num
    raw_target_slots = int(target_size_gb * 1e9 // target_size_per_token)
    aligned_target_slots = (raw_target_slots // page_size + 1) * page_size
    if aligned_target_slots <= target_nsa_pool.size:
        raise ValueError(
            "--hicache-size is too small after reserving Draft LayerSplit "
            "capacity: target host slots "
            f"{aligned_target_slots} <= device slots {target_nsa_pool.size}."
        )

    return target_size_gb, draft_size_gb, mamba_size_gb


def _make_layer_mapper(
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
) -> Callable[[int], Optional[int]]:
    def mapper(layer_id: int) -> Optional[int]:
        if not 0 <= layer_id < transfer_layer_num:
            return None
        return layer_mapping.get(layer_id)

    return mapper


def build_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    server_args: ServerArgs,
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
):
    kv_host_pool_cls = MLATokenToKVPoolHost if use_mla else MHATokenToKVPoolHost
    kwargs = {}
    if override_kv_cache_dim is not None:
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim
    return kv_host_pool_cls(
        kv_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        **kwargs,
    )


def build_pool_entry(
    *,
    name: PoolName,
    host_pool: Any,
    device_pool: Any,
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
    is_anchor: bool = False,
    host_evict_fn: Optional[Callable[[int], Any]] = None,
    device_evict_fn: Optional[Callable[[int], Any]] = None,
    device_alloc_fn: Optional[Callable[[int], Any]] = None,
    device_free_fn: Optional[Callable[[Any], Any]] = None,
) -> PoolEntry:
    return PoolEntry(
        name=name,
        host_pool=host_pool,
        device_pool=device_pool,
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_num),
        is_primary_index_anchor=is_anchor,
        host_evict_fn=host_evict_fn,
        device_evict_fn=device_evict_fn,
        device_alloc_fn=device_alloc_fn,
        device_free_fn=device_free_fn,
    )


def build_kv_only_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        )
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_swa_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )
    swa_host_pool = build_kv_host_pool(
        kv_pool=swa_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )

    # For SWA hybrid, the device alloc/free goes through the inner swa_attn_allocator
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=full_kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def _deepseek_v4_num_host_pages(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kvcache: Any,
    page_size: int,
    swa_page_size: int,
) -> tuple[int, int]:
    allocator = params.token_to_kv_pool_allocator
    device_full_size = getattr(allocator, "size_full", kvcache.size)
    device_full_pages = (device_full_size + page_size - 1) // page_size

    device_swa_pages = (kvcache.swa_size + swa_page_size - 1) // swa_page_size

    if server_args.hicache_size > 0:
        raise ValueError(
            "DeepSeek V4 HiCache currently does not support --hicache-size; "
            "use --hicache-ratio instead."
        )
    ratio = server_args.hicache_ratio
    full_host_pages = max(int(device_full_pages * ratio), device_full_pages + 1)
    swa_host_pages = max(int(device_swa_pages * ratio), device_swa_pages + 1)
    return full_host_pages, swa_host_pages


def build_deepseek_v4_hicache_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kvcache: Any,
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    # TODO(hzh0425): Support PP for deepseek v4 with hicache
    transfer_layer_num = kvcache.end_layer - kvcache.start_layer
    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}
    swa_layer_mapping = {
        layer_id: layer_id for layer_id in range(len(kvcache.swa_kv_pool.kv_buffer))
    }

    c4_layer_mapping = {}
    c128_layer_mapping = {}
    c4_state_global_layers = []
    c128_state_global_layers = []
    for layer_id, layer_item in enumerate(
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    ):
        if layer_item.compress_ratio == 4:
            c4_layer_mapping[layer_id] = layer_item.compress_layer_id
            c4_state_global_layers.append(layer_id)
        elif layer_item.compress_ratio == 128:
            c128_layer_mapping[layer_id] = layer_item.compress_layer_id
            c128_state_global_layers.append(layer_id)

    c4_state_mapping = {
        layer_id: local_id for local_id, layer_id in enumerate(c4_state_global_layers)
    }
    c128_state_mapping = {
        layer_id: local_id for local_id, layer_id in enumerate(c128_state_global_layers)
    }
    num_host_pages, swa_num_host_pages = _deepseek_v4_num_host_pages(
        params=params,
        server_args=server_args,
        kvcache=kvcache,
        page_size=page_size,
        swa_page_size=kvcache.swa_page_size,
    )

    logical_host_pool = LogicalHostPool(num_host_pages * page_size, page_size)
    swa_host_pool = DeepSeekV4PagedHostPool(
        pool_name=str(PoolName.SWA),
        device_buffers=kvcache.swa_kv_pool.kv_buffer,
        item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded,
        num_host_pages=swa_num_host_pages,
        slot_page_size=kvcache.swa_page_size,
        allocator_type=server_args.hicache_storage_backend,
    )
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=logical_host_pool,
            device_pool=kvcache,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=kvcache.swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
        ),
    ]

    if c4_layer_mapping:
        c4_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4),
            device_buffers=kvcache.c4_kv_pool.kv_buffer,
            item_bytes=kvcache.c4_kv_pool.bytes_per_page_padded,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_indexer_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER),
            device_buffers=kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            item_bytes=(
                kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].shape[1]
                * kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].element_size()
            ),
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_state_host_pool = DeepSeekV4StateHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),
            state_pools=[
                kvcache.compress_state_pools[layer_id]
                for layer_id in c4_state_global_layers
            ],
            num_host_pages=swa_num_host_pages,
            swa_page_size=kvcache.swa_page_size,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_indexer_state_host_pool = DeepSeekV4StateHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
            state_pools=[
                kvcache.indexer_compress_state_pools[layer_id]
                for layer_id in c4_state_global_layers
            ],
            num_host_pages=swa_num_host_pages,
            swa_page_size=kvcache.swa_page_size,
            allocator_type=server_args.hicache_storage_backend,
        )
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_pool=c4_host_pool,
                    device_pool=kvcache.c4_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                    host_pool=c4_indexer_host_pool,
                    device_pool=kvcache.c4_indexer_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_STATE,
                    host_pool=c4_state_host_pool,
                    device_pool=None,
                    layer_mapping=c4_state_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                    host_pool=c4_indexer_state_host_pool,
                    device_pool=None,
                    layer_mapping=c4_state_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
            ]
        )

    if c128_layer_mapping:
        c128_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C128),
            device_buffers=kvcache.c128_kv_pool.kv_buffer,
            item_bytes=kvcache.c128_kv_pool.bytes_per_page_padded,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            allocator_type=server_args.hicache_storage_backend,
        )
        c128_state_host_pool = DeepSeekV4StateHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C128_STATE),
            state_pools=[
                kvcache.compress_state_pools[layer_id]
                for layer_id in c128_state_global_layers
            ],
            num_host_pages=swa_num_host_pages,
            swa_page_size=kvcache.swa_page_size,
            allocator_type=server_args.hicache_storage_backend,
        )
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C128,
                    host_pool=c128_host_pool,
                    device_pool=kvcache.c128_kv_pool,
                    layer_mapping=c128_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C128_STATE,
                    host_pool=c128_state_host_pool,
                    device_pool=None,
                    layer_mapping=c128_state_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
            ]
        )

    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
    layer_group_cache_group = (
        attn_cp_group if server_args.glm_nsa_shared_layer_group_hicache else tp_group
    )
    # Split --hicache-size (a machine-total host budget for shared NSA) between
    # Target full-attention, LayerSplit Draft, and Mamba. Draft is registered
    # after tree-cache construction, so its share must be reserved here before
    # the Target host tensor consumes the entire full-attention bucket.
    if server_args.hicache_size > 0:
        (
            full_host_size_gb,
            draft_host_size_gb,
            mamba_host_size_gb,
        ) = _split_hybrid_hicache_budget(
            total_size_gb=server_args.hicache_size,
            mamba_full_memory_ratio=server_args.mamba_full_memory_ratio,
            target_pool=kv_pool,
            draft_pool=params.draft_token_to_kv_pool,
            page_size=page_size,
        )
        if getattr(kv_pool, "use_nsa", False) and (
            server_args.glm_nsa_shared_hicache
            or server_args.glm_nsa_shared_layer_group_hicache
        ):
            # With a shared NSA host pool, --hicache-size is a machine-total
            # budget: the full-attention pool is deduped across the cache
            # group via shm, but MambaPoolHost stays per-rank private. Slice
            # the mamba share per rank so summed physical usage stays within
            # budget.
            cache_group_world = (
                1
                if layer_group_cache_group is None
                else torch.distributed.get_world_size(group=layer_group_cache_group)
            )
            mamba_host_size_gb /= cache_group_world
            logger.info(
                "shared NSA host pool enabled: mamba host budget sliced "
                f"to {mamba_host_size_gb:.2f} GB per rank "
                f"(cache group world size {cache_group_world})."
            )
        logger.info(
            "hybrid host hicache split: "
            f"target_full={full_host_size_gb:.2f} GB, "
            f"draft_reserved={draft_host_size_gb:.2f} GB, "
            f"mamba={mamba_host_size_gb:.2f} GB "
            f"(hicache_size={server_args.hicache_size}, "
            "mamba_full_memory_ratio="
            f"{server_args.mamba_full_memory_ratio})"
        )
    else:
        mamba_host_size_gb = 0
        full_host_size_gb = 0
        draft_host_size_gb = 0

    # NSA/DSA (e.g. GLM5-Next) full-attention KV pools carry a separate indexer
    # buffer in addition to the latent KV, and their device KV buffer is
    # kv_cache_dim wide (kv_lora_rank + fp8 scale + FlashMLA rope padding on
    # DCU). A plain MLATokenToKVPoolHost would size/transfer only the latent KV
    # at kv_lora_rank + qk_rope_head_dim, causing a shape mismatch on transfer
    # (e.g. 512 vs 656) and leaving the indexer unbacked. NSATokenToKVPoolHost
    # sizes the host KV from the device kv_cache_dim and backs up/loads the
    # indexer together with the latent KV. Non-NSA Mamba hybrids keep the plain
    # MLA/MHA host pool.
    if getattr(kv_pool, "use_nsa", False) and server_args.glm_nsa_shared_hicache:
        kv_host_pool = NSATokenToKVPoolHostShared(
            kv_pool,
            server_args.hicache_ratio,
            full_host_size_gb,
            page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
            tp_group=tp_group,
        )
    elif (
        getattr(kv_pool, "use_nsa", False)
        and server_args.glm_nsa_shared_layer_group_hicache
    ):
        kv_host_pool = NSATokenToKVPoolHostSharedLayerGroup(
            kv_pool,
            server_args.hicache_ratio,
            full_host_size_gb,
            page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
            tp_group=attn_cp_group,
        )
    elif getattr(kv_pool, "use_nsa", False):
        kv_host_pool = NSATokenToKVPoolHost(
            kv_pool,
            server_args.hicache_ratio,
            full_host_size_gb,
            page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
    else:
        kv_host_pool_cls = MLATokenToKVPoolHost if use_mla else MHATokenToKVPoolHost
        kv_host_pool = kv_host_pool_cls(
            kv_pool,
            server_args.hicache_ratio,
            full_host_size_gb,
            page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
    if (
        isinstance(kv_host_pool, NSATokenToKVPoolHostSharedLayerGroup)
        and getattr(kv_pool, "layer_shard_enabled", False)
        and (
            kv_pool.layer_shard_size != kv_host_pool.tp_size
            or kv_pool.layer_shard_rank != kv_host_pool.tp_rank
        )
    ):
        # The shared layer-group backup writes only the intersection of the
        # host shard and device-owned layers per rank. If the rank spaces
        # diverge, some layers are never written by any rank and later load-back
        # would read stale host data.
        raise ValueError(
            "NSA shared layer-group hicache requires the device layer "
            "split rank space to match the shared host cache group: "
            f"layer_shard_rank={kv_pool.layer_shard_rank}, "
            f"layer_shard_size={kv_pool.layer_shard_size} vs "
            f"cache_rank={kv_host_pool.tp_rank}, "
            f"cache_size={kv_host_pool.tp_size}."
        )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        server_args.hicache_ratio,
        mamba_host_size_gb,
        allocator_type=server_args.hicache_storage_backend,
        layout=server_args.hicache_mem_layout,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_anchor_sidecar_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    sidecar_pool_name: PoolName,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    sidecar_host_pool_factory: Callable[[Any], Any],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    sidecar_host_pool = sidecar_host_pool_factory(kv_host_pool)
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=sidecar_pool_name,
            host_pool=sidecar_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def attach_hybrid_pool_to_unified_cache(
    cache: UnifiedRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Attach HostPoolGroup + HybridCacheController to UnifiedRadixCache."""
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MLATokenToKVPool,
        NSATokenToKVPool,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    try:
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        swa_stack = isinstance(kvcache, SWAKVPool)
        mamba_stack = isinstance(kvcache, HybridLinearKVPool)
        nsa_stack = isinstance(kvcache, NSATokenToKVPool)
        deepseek_v4_stack = isinstance(kvcache, DeepSeekV4TokenToKVPool)

        if deepseek_v4_stack:
            use_mla = False
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.SWA,
            }, "DeepSeekV4TokenToKVPool requires FULL + SWA in UnifiedRadixCache."
        elif mamba_stack:
            full_kv_pool = kvcache.full_kv_pool
            use_mla = kvcache.use_mla
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.MAMBA,
            }, "HybridLinearKVPool currently only supports FULL + MAMBA in UnifiedRadixCache."
        elif swa_stack:
            full_kv_pool = kvcache.full_kv_pool
            use_mla = False
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.SWA,
            }, "SWAKVPool currently only supports FULL + SWA in UnifiedRadixCache."
        else:
            full_kv_pool = kvcache
            use_mla = isinstance(kvcache, MLATokenToKVPool)
            assert set(cache.components.keys()) == {
                ComponentType.FULL
            }, "Non-hybrid KV pool currently only supports FULL-only UnifiedRadixCache."

        if deepseek_v4_stack:
            host_pool_group, cache_controller = build_deepseek_v4_hicache_stack(
                params=params,
                server_args=server_args,
                kvcache=kvcache,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
                device_swa_evict_fn=lambda n: cache.evict(
                    EvictParams(swa_num_tokens=n)
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.swa_kv_pool_host = host_pool_group.get_pool(PoolName.SWA)
            cache.components[ComponentType.SWA]._swa_kv_pool_host = (
                cache.swa_kv_pool_host
            )
            for pool_name, indices_from_pool in (
                (PoolName.DEEPSEEK_V4_C4, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_INDEXER, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C128, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA),
            ):
                if pool_name in host_pool_group.entry_map:
                    cache.register_sidecar_pool(
                        SidecarPoolSpec(
                            pool_name=pool_name,
                            indices_from_pool=indices_from_pool,
                        )
                    )
            transfer_layer_num = kvcache.end_layer - kvcache.start_layer
        elif mamba_stack:
            full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
            mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
            host_pool_group, cache_controller = build_hybrid_mamba_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                mamba_pool=params.req_to_token_pool.mamba_pool,
                full_layer_mapping=full_layer_mapping,
                mamba_layer_mapping=mamba_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
                device_mamba_evict_fn=lambda n: cache.evict(EvictParams(mamba_num=n)),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
            cache.components[ComponentType.MAMBA]._mamba_pool_host = (
                cache.mamba_pool_host
            )
            params.req_to_token_pool.register_layer_transfer_counter(
                cache_controller.layer_done_counter
            )
            transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        elif swa_stack:
            full_layer_mapping = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if not is_swa
            }
            swa_layer_mapping = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if is_swa
            }
            host_pool_group, cache_controller = build_hybrid_swa_stack(
                params=params,
                server_args=server_args,
                full_kv_pool=full_kv_pool,
                swa_kv_pool=kvcache.swa_kv_pool,
                full_layer_mapping=full_layer_mapping,
                swa_layer_mapping=swa_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=False,
                host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
                device_swa_evict_fn=lambda n: cache.evict(
                    EvictParams(swa_num_tokens=n)
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.swa_kv_pool_host = host_pool_group.get_pool(PoolName.SWA)
            cache.components[ComponentType.SWA]._swa_kv_pool_host = (
                cache.swa_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
        elif nsa_stack:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_anchor_sidecar_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                sidecar_pool_name=PoolName.INDEXER,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                override_kv_cache_dim=full_kv_pool.kv_cache_dim,
                sidecar_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                    full_kv_pool,
                    kv_host_pool,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.register_sidecar_pool(
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                )
            )
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)
        else:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_kv_only_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)

        kvcache.register_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )

        if deepseek_v4_stack:
            pools_desc = "KV + SWA + DeepSeekV4 sidecars"
        elif mamba_stack:
            pools_desc = "KV + MAMBA"
        elif swa_stack:
            pools_desc = "KV + SWA"
        elif nsa_stack:
            pools_desc = "KV + INDEXER"
        else:
            pools_desc = "KV"
        logger.info(
            "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
            pools_desc,
            transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def attach_hybrid_nsa_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.

    This entrypoint is currently intended only for HiRadixCache's NSA path.
    """
    try:
        kv = radix_cache.kv_cache
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            server_args=server_args,
            kv_pool=kv,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=layer_mapping,
            page_size=radix_cache.page_size,
            tp_group=radix_cache.tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=True,
            override_kv_cache_dim=kv.kv_cache_dim,
            prefetch_threshold=prefetch_threshold,
            sidecar_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                kv,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=radix_cache.pp_rank,
            pp_size=radix_cache.pp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid NSA pool stack to HiRadixCache: pools=KV + INDEXER, "
            "transfer_layer_num=%s",
            len(layer_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_nsa_pool_to_hiradix_cache failed")
        raise


def attach_hybrid_pool_to_mamba_cache(
    mamba_cache: HiMambaRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    load_cache_event,
    enable_storage_metrics: bool = False,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Attach HostPoolGroup (KV + Mamba) + HybridCacheController for HiMambaRadixCache.

    This entrypoint is currently intended only for HiMambaRadixCache.
    """
    try:
        hybrid_kv = mamba_cache.hybrid_kv_cache
        kvcache = mamba_cache.kvcache
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            server_args=server_args,
            kv_pool=kvcache,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            page_size=params.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=hybrid_kv.use_mla,
            host_mamba_evict_fn=mamba_cache.evict_mamba_host,
            device_mamba_evict_fn=mamba_cache.evict_mamba,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        mamba_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        mamba_cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
        mamba_cache.transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        mamba_cache.host_pool_group = host_pool_group
        mamba_cache.cache_controller = cache_controller
        params.req_to_token_pool.register_layer_transfer_counter(
            cache_controller.layer_done_counter
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter)
        logger.info(
            "Attached hybrid Mamba pool stack to HiMambaRadixCache: pools=KV + MAMBA, "
            "transfer_layer_num=%s",
            mamba_cache.transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_mamba_cache failed")
        raise
