import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_req(
    rid: str,
    input_len: int,
    *,
    prefix_len: int = 0,
    max_new_tokens: int = 32,
    ignore_eos: bool = False,
) -> Req:
    req = Req.__new__(Req)
    req.rid = rid
    req.origin_input_ids = list(range(input_len))
    req.output_ids = []
    req.fill_ids = list(req.origin_input_ids)
    req.prefix_indices = list(range(prefix_len))
    req.extend_input_len = input_len - prefix_len
    req.extend_logprob_start_len = 0
    req.logprob_start_len = -1
    req.host_hit_length = 0
    req.last_node = None
    req.mamba_pool_idx = None
    req.mamba_ping_pong_track_buffer = None
    req.chunk_starved_rounds = 0
    req.sampling_params = SimpleNamespace(
        max_new_tokens=max_new_tokens,
        ignore_eos=ignore_eos,
    )
    return req


def _make_scheduler(waiting_queue, long_req, *, mamba_available=32) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        prefill_short_req_threshold=256,
        prefill_short_req_max_reserve_ratio=0.5,
        prefill_short_req_scan_depth=4,
        prefill_long_req_starve_threshold=2,
        prefill_short_req_max_total_len=4096,
    )
    scheduler.enable_prefill_short_req_reserve = True
    scheduler.waiting_queue = list(waiting_queue)
    scheduler.chunked_req = long_req
    scheduler.req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(available_size=lambda: mamba_available),
        enable_mamba_extra_buffer=True,
        mamba_ping_pong_track_buffer_size=2,
    )
    return scheduler


def _make_plan_adder(*, rem_total_tokens=4096):
    return SimpleNamespace(
        page_size=64,
        rem_chunk_tokens=1024,
        rem_total_tokens=rem_total_tokens,
        ceil_paged_tokens=lambda tokens: -(-tokens // 64) * 64,
    )


def _make_prefill_adder(*, disable_cache=False) -> PrefillAdder:
    tree_cache = MagicMock()
    tree_cache.disable = disable_cache
    tree_cache.supports_mamba.return_value = False
    tree_cache.evictable_size.return_value = 0
    allocator = MagicMock()
    allocator.available_size.return_value = 100_000
    return PrefillAdder(
        page_size=64,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=SimpleNamespace(reqs=[]),
        new_token_ratio=1.0,
        rem_input_tokens=100_000,
        rem_chunk_tokens=1024,
    )


class TestGlm5NextAutoChunkPlan(CustomTestCase):
    def test_runtime_enablement_is_limited_to_glm5_next(self):
        def build(architecture, disaggregation_mode=DisaggregationMode.PREFILL):
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.server_args = SimpleNamespace(
                chunked_prefill_size=1024,
                prefill_short_req_reserve=True,
                enable_mixed_chunk=False,
                enable_dynamic_chunking=False,
            )
            scheduler.model_config = SimpleNamespace(
                is_multimodal=False,
                hf_config=SimpleNamespace(architectures=[architecture]),
            )
            scheduler.disaggregation_mode = disaggregation_mode
            scheduler.pp_size = 1
            with patch(
                "sglang.srt.managers.scheduler.get_resolved_model_impl",
                return_value=None,
            ):
                scheduler.init_chunked_prefill()
            return scheduler

        causal = build("Glm5NextForCausalLM")
        conditional = build("Glm5NextForConditionalGeneration")
        self.assertTrue(causal.enable_prefill_short_req_reserve)
        self.assertTrue(conditional.enable_prefill_short_req_reserve)
        self.assertFalse(build("OtherForCausalLM").enable_prefill_short_req_reserve)
        self.assertFalse(
            build(
                "Glm5NextForCausalLM", DisaggregationMode.NULL
            ).enable_prefill_short_req_reserve
        )

    def test_page_aligned_plan_and_stable_promotion(self):
        long_req = _make_req("long", 4096)
        skipped_long = _make_req("not-short", 512)
        short_a = _make_req("short-a", 193, prefix_len=64)
        short_b = _make_req("short-b", 64)
        tail = _make_req("tail", 32)
        scheduler = _make_scheduler(
            [skipped_long, short_a, short_b, tail], long_req
        )

        plan = scheduler._plan_auto_chunk(_make_plan_adder(), max_short_reqs=2)

        # 256 short-input tokens plus one 64-token guard page are reserved.
        self.assertEqual(plan.reserved_tokens, 320)
        self.assertEqual(plan.chunk_cap, 704)
        self.assertEqual(plan.short_reqs, [short_a, short_b])

        scheduler._promote_auto_chunk_reqs(plan.short_reqs)
        self.assertEqual(
            scheduler.waiting_queue,
            [short_a, short_b, skipped_long, tail],
        )

    def test_kv_budget_includes_max_new_tokens_and_page_overhead(self):
        long_req = _make_req("long", 4096)
        short_req = _make_req("short", 64, max_new_tokens=256)
        scheduler = _make_scheduler([short_req], long_req)

        # After the full chunk and its page overhead, exactly 320 tokens remain.
        # The short request needs max_new_tokens + one page == 320, and the
        # adder rejects equality, so it must not be selected.
        plan = scheduler._plan_auto_chunk(
            _make_plan_adder(rem_total_tokens=1408), max_short_reqs=1
        )

        self.assertEqual(plan.reserved_tokens, 0)
        self.assertEqual(plan.chunk_cap, 1024)

    def test_mamba_extra_buffer_and_request_slot_limits(self):
        long_req = _make_req("long", 4096)
        short_a = _make_req("short-a", 64)
        short_b = _make_req("short-b", 64)

        no_mamba_room = _make_scheduler([short_a], long_req, mamba_available=2)
        plan = no_mamba_room._plan_auto_chunk(
            _make_plan_adder(), max_short_reqs=1
        )
        self.assertEqual(plan.short_reqs, [])

        one_req_slot = _make_scheduler([short_a, short_b], long_req)
        plan = one_req_slot._plan_auto_chunk(
            _make_plan_adder(), max_short_reqs=1
        )
        self.assertEqual(plan.short_reqs, [short_a])

    def test_starvation_state_commits_only_after_long_request_admission(self):
        long_req = _make_req("long", 4096)
        short_req = _make_req("short", 64)
        scheduler = _make_scheduler([short_req], long_req)
        adder = _make_plan_adder()

        long_req.chunk_starved_rounds = 2
        full_plan = scheduler._plan_auto_chunk(adder, 1)
        self.assertEqual(full_plan.reserved_tokens, 0)
        scheduler._commit_auto_chunk_plan(long_req, full_plan, scheduled=True)
        self.assertEqual(long_req.chunk_starved_rounds, 0)

        compressed_plan = scheduler._plan_auto_chunk(adder, 1)
        scheduler._commit_auto_chunk_plan(
            long_req, compressed_plan, scheduled=False
        )
        self.assertEqual(long_req.chunk_starved_rounds, 0)
        scheduler._commit_auto_chunk_plan(long_req, compressed_plan, scheduled=True)
        self.assertEqual(long_req.chunk_starved_rounds, 1)


class TestGlm5NextAutoChunkArgs(CustomTestCase):
    @staticmethod
    def _make_args(**overrides):
        args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            prefill_short_req_reserve=True,
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_requires_pd_prefill_server(self):
        for mode in ("null", "decode"):
            with self.subTest(mode=mode):
                args = self._make_args(disaggregation_mode=mode)
                with self.assertRaisesRegex(ValueError, "PD prefill server"):
                    args._validate_auto_chunking_args()

    def test_allows_round_robin_nsa_cp(self):
        args = self._make_args(
            enable_nsa_prefill_context_parallel=True,
            nsa_prefill_cp_mode="round-robin-split",
        )

        args._validate_auto_chunking_args()

    def test_rejects_single_request_cp_modes(self):
        cases = (
            (
                {
                    "enable_nsa_prefill_context_parallel": True,
                    "nsa_prefill_cp_mode": "in-seq-split",
                },
                "in-seq-split",
            ),
            (
                {"enable_prefill_context_parallel": True},
                "general prefill context parallelism",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                args = self._make_args(**overrides)
                with self.assertRaisesRegex(ValueError, expected):
                    args._validate_auto_chunking_args()

    def test_rejects_other_static_incompatible_modes(self):
        cases = (
            ({"chunked_prefill_size": -1}, "disabled chunked prefill"),
            ({"schedule_policy": "lpm"}, "non-FCFS scheduling"),
            ({"enable_priority_scheduling": True}, "priority scheduling"),
            ({"enable_dynamic_chunking": True}, "dynamic chunking"),
            ({"dllm_algorithm": "dream"}, "diffusion LLM scheduling"),
            ({"enable_prefill_delayer": True}, "prefill delaying"),
            ({"enable_lora": True}, "LoRA serving"),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                args = self._make_args(**overrides)
                with self.assertRaisesRegex(ValueError, expected):
                    args._validate_auto_chunking_args()


class TestPrefillAdderAutoChunkGuards(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_chunk_cap_is_applied(self):
        adder = _make_prefill_adder()
        long_req = _make_req("long", 2048)

        unfinished = adder.add_chunked_req(long_req, max_tokens=320)

        self.assertIs(unfinished, long_req)
        self.assertEqual(long_req.extend_input_len, 320)
        self.assertEqual(adder.rem_chunk_tokens, 704)

    def test_second_unfinished_request_is_rejected(self):
        adder = _make_prefill_adder()
        long_req = _make_req("long", 2048)
        adder.add_chunked_req(long_req, max_tokens=512)
        candidate = _make_req("candidate", 1024)

        result = adder.add_one_req(
            candidate,
            has_chunked_req=True,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertNotIn(candidate, adder.can_run_list)

    def test_second_ignore_eos_unfinished_request_is_rejected(self):
        adder = _make_prefill_adder(disable_cache=True)
        long_req = _make_req("long", 2048)
        adder.add_chunked_req(long_req, max_tokens=512)
        candidate = _make_req("candidate", 1024, ignore_eos=True)

        result = adder.add_one_req(
            candidate,
            has_chunked_req=True,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertNotIn(candidate, adder.can_run_list)


if __name__ == "__main__":
    unittest.main()
