import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    NSATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small")


def _set_hicache_mtp_fix(val):
    envs.GLM_USE_HICACHE_MTP_FIX._value = bool(val)


def _make_layer_split_nsa_pool(
    *,
    rank,
    layer_num=1,
    rank_offset=7,
    ring_size=1,
    scratch_source=None,
):
    """Build the small CPU NSA pool used by the LayerSplit tests."""
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

    # Ownership and all persistent/remote tensor allocation stay real. Only
    # distributed communicator construction is suppressed: these unit tests do
    # not initialize a CP process group and never execute a broadcast.
    with patch.object(
        NSATokenToKVPool, "_init_layer_broadcast_comm", return_value=None
    ):
        return NSATokenToKVPool(
            size=64,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=layer_num,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
            layer_shard_rank=rank,
            layer_shard_size=8,
            layer_shard_rank_offset=rank_offset,
            mla_kv_prefetch_ring_size=ring_size,
            layer_split_scratch_source=scratch_source,
        )


# ── CPU tests ───────────────────────────────────────────────────────────────


class TestBindIndicesCorrectness(unittest.TestCase):
    """_bind_primary_indices — no isinstance checks, pure logic."""

    def test_bind_copies_reference_exactly(self):
        host = torch.tensor([100, 200, 300])
        pt = PoolTransfer(name="mtp_0")
        HybridCacheController._bind_primary_indices([pt], host_indices=host)
        self.assertIs(pt.host_indices, host)

    def test_bind_only_on_unset_indices(self):
        mamba_host = torch.tensor([50, 60])
        pt = PoolTransfer(name="mamba", host_indices=mamba_host)
        HybridCacheController._bind_primary_indices(
            [pt], host_indices=torch.tensor([1, 2])
        )
        self.assertIs(pt.host_indices, mamba_host)

    def test_multiple_pools_share_same_host_indices(self):
        host = torch.tensor([10, 20])
        mtp0, mtp1 = PoolTransfer(name="mtp_0"), PoolTransfer(name="mtp_1")
        HybridCacheController._bind_primary_indices([mtp0, mtp1], host_indices=host)
        self.assertIs(mtp0.host_indices, host)
        self.assertIs(mtp1.host_indices, host)


class TestMTPPoolTransferMerge(unittest.TestCase):
    """_mtp_pool_transfers and _merge_mtp_if_needed."""

    def setUp(self):
        self._orig = envs.GLM_USE_HICACHE_MTP_FIX.get()
        _set_hicache_mtp_fix(True)

    def tearDown(self):
        envs.GLM_USE_HICACHE_MTP_FIX._value = self._orig

    def _make_entry(self, name):
        return PoolEntry(
            name=name,
            host_pool=MagicMock(),
            device_pool=MagicMock(),
            layer_mapper=lambda i: i,
        )

    def test_empty_entries_returns_none(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = MagicMock()
        cache.extra_hicache_entries = []
        result = HiMambaRadixCache._mtp_pool_transfers(cache)
        self.assertIsNone(result)

    def test_generates_correct_number(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = MagicMock()
        cache.extra_hicache_entries = [
            self._make_entry("mtp_0"),
            self._make_entry("mtp_1"),
        ]
        result = HiMambaRadixCache._mtp_pool_transfers(cache)
        self.assertEqual(len(result), 2)

    def test_merge_preserves_existing_order(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = MagicMock()
        cache.extra_hicache_entries = [self._make_entry("mtp_0")]
        cache._mtp_pool_transfers = lambda: HiMambaRadixCache._mtp_pool_transfers(cache)
        cache._merge_mtp_if_needed = (
            lambda e=None: HiMambaRadixCache._merge_mtp_if_needed(cache, e)
        )
        existing = [PoolTransfer(name="mamba"), PoolTransfer(name="indexer")]
        result = cache._merge_mtp_if_needed(existing)
        names = [p.name for p in result]
        self.assertEqual(names, ["mamba", "indexer", "mtp_0"])

    def test_merge_with_none_existing(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = MagicMock()
        cache.extra_hicache_entries = [self._make_entry("mtp_0")]
        cache._mtp_pool_transfers = lambda: HiMambaRadixCache._mtp_pool_transfers(cache)
        cache._merge_mtp_if_needed = (
            lambda e=None: HiMambaRadixCache._merge_mtp_if_needed(cache, e)
        )
        result = cache._merge_mtp_if_needed(None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "mtp_0")


class TestRegisterMTPWithIsinstance(unittest.TestCase):
    """register_mtp_hicache_pools — needs real isinstance, use real objects."""

    def setUp(self):
        self._orig = envs.GLM_USE_HICACHE_MTP_FIX.get()
        _set_hicache_mtp_fix(True)

    def tearDown(self):
        envs.GLM_USE_HICACHE_MTP_FIX._value = self._orig

    def test_register_with_real_nsa_pool(self):
        """Using a real NSATokenToKVPool passes isinstance check."""
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
        from sglang.srt.server_args import ServerArgs

        # Create a minimal real NSATokenToKVPool
        device_pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )

        # Build a real HiRadixCache backed by this pool
        params = CacheInitParams(
            disable=False,
            page_size=64,
            token_to_kv_pool_allocator=MagicMock(),
            tp_cache_group=MagicMock(),
            req_to_token_pool=MagicMock(),
            enable_kv_cache_events=False,
        )
        # Patch the pool into the cache
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.kv_cache = device_pool
        cache.token_to_kv_pool_host = MagicMock()
        cache.token_to_kv_pool_host.size = 1000
        cache.kv_cache.size = 128
        cache.page_size = 64
        cache.extra_hicache_entries = []
        cache.enable_storage = False
        cache.cache_controller = MagicMock()
        cache.tp_group = MagicMock()

        # Create MTP pool
        mtp_pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )

        server_args = MagicMock(spec=ServerArgs)
        server_args.glm_nsa_shared_hicache = False
        server_args.glm_nsa_shared_layer_group_hicache = False
        server_args.hicache_mem_layout = "layer_first"
        server_args.hicache_storage_backend = None

        HiRadixCache.register_mtp_hicache_pools(cache, [mtp_pool], server_args)

        self.assertEqual(len(cache.extra_hicache_entries), 1)
        self.assertEqual(cache.extra_hicache_entries[0].name, "mtp_0")

    def _make_himamba_cache(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        target_pool = NSATokenToKVPool(
            size=64,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=11,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )
        cache = HiMambaRadixCache.__new__(HiMambaRadixCache)
        cache.kvcache = target_pool
        cache.full_kv_pool_host = MagicMock(size=192)
        cache.page_size = 64
        cache.extra_hicache_entries = []
        cache.enable_storage = False
        cache.cache_controller = MagicMock()
        cache.host_pool_group = None
        cache.tp_group = MagicMock()
        return cache

    def _make_server_args(self):
        from sglang.srt.server_args import ServerArgs

        server_args = MagicMock(spec=ServerArgs)
        server_args.glm_nsa_shared_hicache = False
        server_args.glm_nsa_shared_layer_group_hicache = True
        server_args.hicache_mem_layout = "layer_first"
        server_args.hicache_storage_backend = None
        return server_args

    def test_layer_split_non_owner_registers_device_only_in_himamba(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = self._make_himamba_cache()
        draft_pool = _make_layer_split_nsa_pool(rank=0)

        self.assertFalse(draft_pool._is_layer_owned(draft_pool.start_layer))
        self.assertEqual(draft_pool._get_layer_owner_rank(draft_pool.start_layer), 7)
        self.assertEqual(draft_pool.kv_buffer[0].numel(), 0)
        draft_index_buffers = (
            draft_pool.index_k_with_scale_buffer
            if draft_pool.use_fp8_index_k_cache
            else draft_pool.index_k_buffer
        )
        self.assertEqual(draft_index_buffers[0].numel(), 0)

        HiMambaRadixCache.register_mtp_hicache_pools(
            cache, [draft_pool], self._make_server_args()
        )

        cache.cache_controller.set_draft_kv_pool.assert_called_once_with(
            draft_pool, None
        )
        self.assertEqual(cache.extra_hicache_entries, [])

    def test_layer_split_owner_registers_single_host_pool_in_himamba(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = self._make_himamba_cache()
        draft_pool = _make_layer_split_nsa_pool(rank=7)

        self.assertTrue(draft_pool._is_layer_owned(draft_pool.start_layer))
        self.assertEqual(draft_pool._get_layer_owner_rank(draft_pool.start_layer), 7)
        self.assertGreater(draft_pool.kv_buffer[0].numel(), 0)
        draft_index_buffers = (
            draft_pool.index_k_with_scale_buffer
            if draft_pool.use_fp8_index_k_cache
            else draft_pool.index_k_buffer
        )
        self.assertGreater(draft_index_buffers[0].numel(), 0)

        HiMambaRadixCache.register_mtp_hicache_pools(
            cache, [draft_pool], self._make_server_args()
        )

        cache.cache_controller.set_draft_kv_pool.assert_called_once_with(
            draft_pool, None
        )
        self.assertEqual(len(cache.extra_hicache_entries), 1)
        self.assertEqual(cache.extra_hicache_entries[0].name, "mtp_0")
        self.assertIsInstance(
            cache.extra_hicache_entries[0].host_pool,
            NSATokenToKVPoolHost,
        )

    def test_single_draft_layer_with_rank_offset_is_owned_only_by_rank_seven(self):
        owners = []
        for rank in range(8):
            pool = _make_layer_split_nsa_pool(rank=rank)
            if pool._is_layer_owned(pool.start_layer):
                owners.append(rank)
            self.assertEqual(pool._get_layer_owner_rank(pool.start_layer), 7)

        self.assertEqual(owners, [7])

    def test_himamba_rejects_mismatched_draft_token_capacity(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = self._make_himamba_cache()
        draft_pool = _make_layer_split_nsa_pool(rank=7)
        draft_pool.size += draft_pool.page_size

        with self.assertRaisesRegex(
            ValueError,
            "Target and MTP Draft KV pools must have the same token capacity",
        ):
            HiMambaRadixCache.register_mtp_hicache_pools(
                cache, [draft_pool], self._make_server_args()
            )

    def test_register_no_nsa_pool_is_noop(self):
        """Empty pool list → no entries added."""
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        cache = MagicMock()
        device_pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )
        cache.kvcache = device_pool
        cache.extra_hicache_entries = []
        cache.enable_storage = False

        HiMambaRadixCache.register_mtp_hicache_pools(cache, [], MagicMock())
        self.assertEqual(len(cache.extra_hicache_entries), 0)

    def test_registered_mtp_pool_can_alloc(self):
        """After registration, the MTP host pool can alloc and store data."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
        from sglang.srt.server_args import ServerArgs

        device_pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=11,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )

        cache = HiRadixCache.__new__(HiRadixCache)
        cache.kv_cache = device_pool
        cache.token_to_kv_pool_host = MagicMock()
        cache.token_to_kv_pool_host.size = 1000
        cache.kv_cache.size = 128
        cache.page_size = 64
        cache.extra_hicache_entries = []
        cache.enable_storage = False
        cache.cache_controller = MagicMock()
        cache.tp_group = MagicMock()

        mtp_pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )

        server_args = MagicMock(spec=ServerArgs)
        server_args.glm_nsa_shared_hicache = False
        server_args.glm_nsa_shared_layer_group_hicache = False
        server_args.hicache_mem_layout = "layer_first"
        server_args.hicache_storage_backend = None

        HiRadixCache.register_mtp_hicache_pools(cache, [mtp_pool], server_args)

        # Verify MTP host pool was created correctly
        entry = cache.extra_hicache_entries[0]
        self.assertEqual(entry.name, "mtp_0")
        self.assertEqual(entry.device_pool.layer_num, 1)
        self.assertIsInstance(entry.host_pool, NSATokenToKVPoolHost)

        # Verify host pool can alloc
        indices = entry.host_pool.alloc(64)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 64)

        # Verify host pool can free
        freed = entry.host_pool.free(indices)
        self.assertEqual(freed, 64)

    def test_register_rebuilds_host_pool_group(self):
        """After registration, cache_controller.mem_pool_host is re-assigned."""
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup
        from sglang.srt.server_args import ServerArgs

        device_pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=11,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )

        cache = MagicMock(spec=HiMambaRadixCache)
        cache.kvcache = device_pool
        cache.full_kv_pool_host = MagicMock()
        cache.full_kv_pool_host.size = 1000
        cache.kvcache.size = 128
        cache.page_size = 64
        cache.extra_hicache_entries = []
        cache.enable_storage = False
        cache.cache_controller = MagicMock()

        # Simulate existing host_pool_group (created by attach_hybrid_pool_to_mamba_cache)
        kv_entry = PoolEntry(
            name="kv",
            host_pool=MagicMock(),
            device_pool=MagicMock(),
            layer_mapper=lambda i: i,
            is_primary_index_anchor=True,
        )
        mamba_entry = PoolEntry(
            name="mamba",
            host_pool=MagicMock(),
            device_pool=MagicMock(),
            layer_mapper=lambda i: i,
        )
        cache.host_pool_group = HostPoolGroup([kv_entry, mamba_entry])
        cache.cache_controller.mem_pool_host = cache.host_pool_group

        mtp_pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )

        server_args = MagicMock(spec=ServerArgs)
        server_args.glm_nsa_shared_hicache = False
        server_args.glm_nsa_shared_layer_group_hicache = False
        server_args.hicache_mem_layout = "layer_first"
        server_args.hicache_storage_backend = None

        HiMambaRadixCache.register_mtp_hicache_pools(cache, [mtp_pool], server_args)

        # Verify HostPoolGroup was rebuilt with MTP entry
        self.assertEqual(len(cache.extra_hicache_entries), 1)
        # The host_pool_group should now include kv + mamba + mtp
        self.assertIsInstance(cache.host_pool_group, HostPoolGroup)
        entry_names = [e.name for e in cache.host_pool_group.entries]
        self.assertIn("kv", entry_names)
        self.assertIn("mamba", entry_names)
        self.assertIn("mtp_0", entry_names)


class TestDraftLayerSplitScratchReuse(unittest.TestCase):
    def test_draft_reuses_target_main_kv_ring_tensors(self):
        target_pool = _make_layer_split_nsa_pool(
            rank=7,
            layer_num=11,
            rank_offset=0,
            ring_size=2,
        )
        draft_pool = _make_layer_split_nsa_pool(
            rank=7,
            layer_num=1,
            rank_offset=7,
            ring_size=2,
            scratch_source=target_pool,
        )

        self.assertIsNot(draft_pool.remote_kv_buffers, target_pool.remote_kv_buffers)
        self.assertEqual(len(draft_pool.remote_kv_buffers), 2)
        for draft_buffer, target_buffer in zip(
            draft_pool.remote_kv_buffers, target_pool.remote_kv_buffers
        ):
            self.assertIs(draft_buffer, target_buffer)
            self.assertEqual(draft_buffer.data_ptr(), target_buffer.data_ptr())
        remote_index_attr = (
            "remote_index_k_with_scale_buffer"
            if draft_pool.use_fp8_index_k_cache
            else "remote_index_k_buffer"
        )
        self.assertIsNot(
            getattr(draft_pool, remote_index_attr),
            getattr(target_pool, remote_index_attr),
        )

    def test_rejects_incompatible_target_scratch_geometry(self):
        target_pool = _make_layer_split_nsa_pool(
            rank=7,
            layer_num=11,
            rank_offset=0,
        )
        draft_pool = _make_layer_split_nsa_pool(rank=7)
        draft_pool.layer_split_scratch_source = target_pool
        target_pool.size += target_pool.page_size

        with self.assertRaisesRegex(
            ValueError,
            "Incompatible LayerSplit Main-KV scratch source: size",
        ):
            draft_pool._get_shared_main_kv_scratch()


class TestSchedulerMTPRegistration(unittest.TestCase):
    """_register_mtp_hicache_pools in scheduler (draft_pool passed by the gate)."""

    def setUp(self):
        self._orig = envs.GLM_USE_HICACHE_MTP_FIX.get()

    def tearDown(self):
        envs.GLM_USE_HICACHE_MTP_FIX._value = self._orig

    def _make_nsa_pool(self):
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        return NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )

    def test_noop_when_draft_pool_none(self):
        """draft_pool=None → early return, tree_cache not touched."""
        from sglang.srt.managers.scheduler import Scheduler

        sched = MagicMock()
        Scheduler._register_mtp_hicache_pools(sched, None)
        sched.tree_cache.register_mtp_hicache_pools.assert_not_called()

    def test_registers_with_himamba(self):
        """draft_pool passed through → forwarded to tree_cache.register_mtp_hicache_pools."""
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        sched = MagicMock()
        sched.enable_hierarchical_cache = True
        sched.server_args = MagicMock()
        sched.tree_cache = MagicMock(spec=HiMambaRadixCache)

        pool = self._make_nsa_pool()
        Scheduler._register_mtp_hicache_pools(sched, pool)
        sched.tree_cache.register_mtp_hicache_pools.assert_called_once_with(
            [pool], sched.server_args
        )


class TestHybridMTPHiCacheBudget(unittest.TestCase):
    def _make_nsa_pool(self, layer_num):
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        pool = NSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=layer_num,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )
        pool.layer_shard_enabled = True
        pool.layer_shard_size = 8
        return pool

    def test_explicit_size_reserves_one_of_twelve_full_layers(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _split_hybrid_hicache_budget,
        )

        target_pool = self._make_nsa_pool(layer_num=11)
        draft_pool = self._make_nsa_pool(layer_num=1)

        target_gb, draft_gb, mamba_gb = _split_hybrid_hicache_budget(
            total_size_gb=12.0,
            mamba_full_memory_ratio=1.0,
            target_pool=target_pool,
            draft_pool=draft_pool,
            page_size=64,
        )

        self.assertAlmostEqual(target_gb, 5.5)
        self.assertAlmostEqual(draft_gb, 0.5)
        self.assertAlmostEqual(mamba_gb, 6.0)
        self.assertAlmostEqual(target_gb + draft_gb + mamba_gb, 12.0)

    def test_target_host_ratio_preserves_draft_slots_within_page_budget(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _nsa_hicache_bytes_per_token_per_layer,
            _split_hybrid_hicache_budget,
        )

        page_size = 64
        target_pool = self._make_nsa_pool(layer_num=11)
        draft_pool = self._make_nsa_pool(layer_num=1)
        target_gb, draft_gb, _ = _split_hybrid_hicache_budget(
            total_size_gb=12.0,
            mamba_full_memory_ratio=1.0,
            target_pool=target_pool,
            draft_pool=draft_pool,
            page_size=page_size,
        )
        bytes_per_layer = _nsa_hicache_bytes_per_token_per_layer(target_pool, page_size)
        target_bytes_per_token = bytes_per_layer * target_pool.layer_num
        draft_bytes_per_token = bytes_per_layer * draft_pool.layer_num

        # These are the exact HostKVCache sizing rules: an explicit Target
        # budget is aligned first. HiMamba seeds the Draft ratio one slot below
        # that page boundary, so Draft's align-up lands on the same slot count.
        target_raw_slots = int(target_gb * 1e9 // target_bytes_per_token)
        target_host_slots = (target_raw_slots // page_size + 1) * page_size
        target_host_ratio = (target_host_slots - 1) / target_pool.size
        draft_ratio_slots = int(draft_pool.size * target_host_ratio)
        draft_host_slots = (draft_ratio_slots // page_size + 1) * page_size

        self.assertEqual(draft_host_slots, target_host_slots)

        physical_full_bytes = (
            target_host_slots * target_bytes_per_token
            + draft_host_slots * draft_bytes_per_token
        )
        explicit_full_budget_bytes = (target_gb + draft_gb) * 1e9
        # One shared slot alignment applies to all 11 Target + 1 Draft layers.
        page_alignment_tolerance = page_size * (
            target_bytes_per_token + draft_bytes_per_token
        )
        self.assertLessEqual(
            physical_full_bytes,
            explicit_full_budget_bytes + page_alignment_tolerance,
        )

    def test_ratio_mode_does_not_split_an_explicit_budget(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _split_hybrid_hicache_budget,
        )

        target_pool = self._make_nsa_pool(layer_num=11)
        draft_pool = self._make_nsa_pool(layer_num=1)

        self.assertEqual(
            _split_hybrid_hicache_budget(
                total_size_gb=0,
                mamba_full_memory_ratio=1.0,
                target_pool=target_pool,
                draft_pool=draft_pool,
                page_size=64,
            ),
            (0.0, 0.0, 0.0),
        )

    def test_rejects_explicit_budget_too_small_after_draft_reservation(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _split_hybrid_hicache_budget,
        )

        with self.assertRaisesRegex(
            ValueError,
            "--hicache-size is too small after reserving Draft LayerSplit",
        ):
            _split_hybrid_hicache_budget(
                total_size_gb=1e-6,
                mamba_full_memory_ratio=1.0,
                target_pool=self._make_nsa_pool(layer_num=11),
                draft_pool=self._make_nsa_pool(layer_num=1),
                page_size=64,
            )


# ── GPU tests ───────────────────────────────────────────────────────────────


class TestMTPHiCacheGPU(unittest.TestCase):
    """GPU tests."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available")

    def test_bind_indices_on_gpu(self):
        host = torch.tensor([0, 64], dtype=torch.int64)
        device = torch.tensor([100, 164], dtype=torch.int64)
        mtp = PoolTransfer(name="mtp_0")
        HybridCacheController._bind_primary_indices(
            [mtp], host_indices=host, device_indices=device
        )
        self.assertIs(mtp.host_indices, host)
        self.assertIs(mtp.device_indices, device)


if __name__ == "__main__":
    unittest.main()
