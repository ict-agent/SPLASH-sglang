import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    NSATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small")


def _set_hicache_mtp_fix(val):
    envs.GLM_USE_HICACHE_MTP_FIX._value = bool(val)


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
        HybridCacheController._bind_primary_indices([pt], host_indices=torch.tensor([1, 2]))
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
        return PoolEntry(name=name, host_pool=MagicMock(), device_pool=MagicMock(), layer_mapper=lambda i: i)

    def test_empty_entries_returns_none(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        cache = MagicMock()
        cache.extra_hicache_entries = []
        result = HiMambaRadixCache._mtp_pool_transfers(cache)
        self.assertIsNone(result)

    def test_generates_correct_number(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        cache = MagicMock()
        cache.extra_hicache_entries = [self._make_entry("mtp_0"), self._make_entry("mtp_1")]
        result = HiMambaRadixCache._mtp_pool_transfers(cache)
        self.assertEqual(len(result), 2)

    def test_merge_preserves_existing_order(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        cache = MagicMock()
        cache.extra_hicache_entries = [self._make_entry("mtp_0")]
        cache._mtp_pool_transfers = lambda: HiMambaRadixCache._mtp_pool_transfers(cache)
        cache._merge_mtp_if_needed = lambda e=None: HiMambaRadixCache._merge_mtp_if_needed(cache, e)
        existing = [PoolTransfer(name="mamba"), PoolTransfer(name="indexer")]
        result = cache._merge_mtp_if_needed(existing)
        names = [p.name for p in result]
        self.assertEqual(names, ["mamba", "indexer", "mtp_0"])

    def test_merge_with_none_existing(self):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        cache = MagicMock()
        cache.extra_hicache_entries = [self._make_entry("mtp_0")]
        cache._mtp_pool_transfers = lambda: HiMambaRadixCache._mtp_pool_transfers(cache)
        cache._merge_mtp_if_needed = lambda e=None: HiMambaRadixCache._merge_mtp_if_needed(cache, e)
        result = cache._merge_mtp_if_needed(None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "mtp_0")


class TestRegisterMTPWithIsinstance(unittest.TestCase):
    """register_mtp_hicache_pools — needs real isinstance, use real objects."""

    def setUp(self):
        self._orig = envs.GLM_USE_HICACHE_MTP_FIX.get()

    def tearDown(self):
        envs.GLM_USE_HICACHE_MTP_FIX._value = self._orig

    def test_register_with_real_nsa_pool(self):
        """Using a real NSATokenToKVPool passes isinstance check."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams

        # Create a minimal real NSATokenToKVPool
        device_pool = NSATokenToKVPool(
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=1,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
        )

        # Build a real HiRadixCache backed by this pool
        params = CacheInitParams(
            disable=False, page_size=64,
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
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=1,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
        )

        server_args = MagicMock(spec=ServerArgs)
        server_args.glm_nsa_shared_hicache = False
        server_args.glm_nsa_shared_layer_group_hicache = False
        server_args.hicache_mem_layout = "layer_first"
        server_args.hicache_storage_backend = None

        HiRadixCache.register_mtp_hicache_pools(cache, [mtp_pool], server_args)

        self.assertEqual(len(cache.extra_hicache_entries), 1)
        self.assertEqual(cache.extra_hicache_entries[0].name, "mtp_0")

    def test_register_no_nsa_pool_is_noop(self):
        """Empty pool list → no entries added."""
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        cache = MagicMock()
        device_pool = NSATokenToKVPool(
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=1,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
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
        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup
        from sglang.srt.server_args import ServerArgs

        device_pool = NSATokenToKVPool(
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=11,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
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
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=1,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
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
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=11,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
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
        kv_entry = PoolEntry(name="kv", host_pool=MagicMock(), device_pool=MagicMock(), layer_mapper=lambda i: i, is_primary_index_anchor=True)
        mamba_entry = PoolEntry(name="mamba", host_pool=MagicMock(), device_pool=MagicMock(), layer_mapper=lambda i: i)
        cache.host_pool_group = HostPoolGroup([kv_entry, mamba_entry])
        cache.cache_controller.mem_pool_host = cache.host_pool_group

        mtp_pool = NSATokenToKVPool(
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=1,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
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


class TestSchedulerMTPRegistration(unittest.TestCase):
    """_register_mtp_hicache_pools in scheduler (draft_pool passed by the gate)."""

    def setUp(self):
        self._orig = envs.GLM_USE_HICACHE_MTP_FIX.get()

    def tearDown(self):
        envs.GLM_USE_HICACHE_MTP_FIX._value = self._orig

    def _make_nsa_pool(self):
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
        return NSATokenToKVPool(
            size=128, page_size=64, dtype=torch.bfloat16,
            kv_lora_rank=512, qk_rope_head_dim=0, layer_num=1,
            device="cpu", index_head_dim=128,
            enable_memory_saver=False, kv_cache_dim=576,
        )

    def test_noop_when_draft_pool_none(self):
        """draft_pool=None → early return, tree_cache not touched."""
        from sglang.srt.managers.scheduler import Scheduler
        sched = MagicMock()
        Scheduler._register_mtp_hicache_pools(sched, None)
        sched.tree_cache.register_mtp_hicache_pools.assert_not_called()

    def test_registers_with_himamba(self):
        """draft_pool passed through → forwarded to tree_cache.register_mtp_hicache_pools."""
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
        from sglang.srt.managers.scheduler import Scheduler

        sched = MagicMock()
        sched.enable_hierarchical_cache = True
        sched.server_args = MagicMock()
        sched.tree_cache = MagicMock(spec=HiMambaRadixCache)

        pool = self._make_nsa_pool()
        Scheduler._register_mtp_hicache_pools(sched, pool)
        sched.tree_cache.register_mtp_hicache_pools.assert_called_once_with(
            [pool], sched.server_args
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
        HybridCacheController._bind_primary_indices([mtp], host_indices=host, device_indices=device)
        self.assertIs(mtp.host_indices, host)
        self.assertIs(mtp.device_indices, device)


if __name__ == "__main__":
    unittest.main()
