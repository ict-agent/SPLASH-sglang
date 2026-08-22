"""Unit tests for GLM speculative-token streaming output handling."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.glm_utils import interleave_batch_token_id_out
from sglang.srt.managers.io_struct import BatchStrOutput, BatchTokenIDOutput
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_batch_token_id_output() -> BatchTokenIDOutput:
    return BatchTokenIDOutput(
        rids=["r0", "r1"],
        http_worker_ipcs=["ipc0", "ipc1"],
        spec_verify_ct=[2, 2],
        spec_num_correct_drafts=[1, 2],
        spec_correct_drafts_histogram=[[0, 1], [0, 0, 1]],
        finished_reasons=[None, None],
        decoded_texts=["", ""],
        decode_ids=[[101, 102], [201, 202]],
        read_offsets=[0, 0],
        output_ids=[[101, 102], [201, 202]],
        skip_special_tokens=[True, True],
        spaces_between_special_tokens=[True, True],
        no_stop_trim=[False, False],
        prompt_tokens=[4, 5],
        reasoning_tokens=[0, 0],
        completion_tokens=[2, 2],
        cached_tokens=[0, 0],
        cached_tokens_details=[None, None],
        input_token_logprobs_val=[["input-val-0"], ["input-val-1"]],
        input_token_logprobs_idx=[["input-idx-0"], ["input-idx-1"]],
        output_token_logprobs_val=[[0.1, 0.2], [0.3, 0.4]],
        output_token_logprobs_idx=[[101, 102], [201, 202]],
        input_top_logprobs_val=[["input-top-val-0"], ["input-top-val-1"]],
        input_top_logprobs_idx=[["input-top-idx-0"], ["input-top-idx-1"]],
        output_top_logprobs_val=[
            ["top-val-00", "top-val-01"],
            ["top-val-10", "top-val-11"],
        ],
        output_top_logprobs_idx=[
            ["top-idx-00", "top-idx-01"],
            ["top-idx-10", "top-idx-11"],
        ],
        input_token_ids_logprobs_val=[
            ["input-token-val-0"],
            ["input-token-val-1"],
        ],
        input_token_ids_logprobs_idx=[
            ["input-token-idx-0"],
            ["input-token-idx-1"],
        ],
        output_token_ids_logprobs_val=[
            ["token-val-00", "token-val-01"],
            ["token-val-10", "token-val-11"],
        ],
        output_token_ids_logprobs_idx=[
            ["token-idx-00", "token-idx-01"],
            ["token-idx-10", "token-idx-11"],
        ],
        output_token_entropy_val=[
            ["entropy-00", "entropy-01"],
            ["entropy-10", "entropy-11"],
        ],
        output_hidden_states=[["hidden-00", "hidden-01"], ["hidden-10", "hidden-11"]],
        routed_experts=[["expert-00", "expert-01"], ["expert-10", "expert-11"]],
        indexer_topk=[["indexer-00", "indexer-01"], ["indexer-10", "indexer-11"]],
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        retraction_counts=[0, 0],
        token_steps=[[1, 2], [1, 2]],
        dp_ranks=[0, 1],
        time_stats=[None, None],
    )


def _make_scheduler_req(rid: str, token_id: int, return_optional: bool):
    return SimpleNamespace(
        rid=rid,
        http_worker_ipc=f"ipc-{rid}",
        finished=lambda: False,
        stream=True,
        sampling_params=SimpleNamespace(
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        ),
        check_match_stop_str_prefix=lambda: False,
        output_ids=[token_id],
        output_ids_through_stop=[token_id],
        send_token_offset=0,
        send_output_token_logprobs_offset=0,
        send_decode_id_offset=0,
        finished_reason=None,
        decoded_text="",
        init_incremental_detokenize=lambda: ([token_id], 0),
        origin_input_ids=[1, 2],
        reasoning_tokens=0,
        cached_tokens=0,
        retraction_count=0,
        time_stats=None,
        return_hidden_states=return_optional,
        hidden_states=f"hidden-{rid}",
        return_routed_experts=return_optional,
        routed_experts=f"experts-{rid}",
        return_indexer_topk=return_optional,
        indexer_topk=f"indexer-{rid}",
        customized_info=None,
    )


def _make_batch_str_output(text: str, token_id: int) -> BatchStrOutput:
    return BatchStrOutput(
        rids=["r0"],
        http_worker_ipcs=None,
        spec_verify_ct=[],
        spec_num_correct_drafts=[],
        spec_correct_drafts_histogram=None,
        finished_reasons=[None],
        output_strs=[text],
        output_ids=[[token_id]],
        prompt_tokens=[2],
        completion_tokens=[1],
        reasoning_tokens=[0],
        cached_tokens=[0],
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=None,
        output_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_token_entropy_val=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        retraction_counts=[0],
        token_steps=None,
        load=None,
        customized_info=None,
        cached_tokens_details=None,
        dp_ranks=None,
        time_stats=None,
    )


class TestGlmStreamSpeculatedTokens(CustomTestCase):
    def test_non_incremental_chunks_keep_per_token_snapshots(self):
        state = ReqState(
            out_list=[],
            finished=False,
            event=MagicMock(),
            obj=SimpleNamespace(
                stream=True,
                return_logprob=False,
                log_metrics=False,
            ),
            time_stats=SimpleNamespace(first_token_time=1.0),
        )
        manager = SimpleNamespace(
            rid_to_state={"r0": state},
            enable_metrics=False,
            dump_requests_folder=None,
            crash_dump_folder=None,
            server_args=SimpleNamespace(
                incremental_streaming_output=False,
                glm_stream_speculated_tokens=True,
                speculative_algorithm=None,
                enable_lora=False,
                batch_notify_size=64,
                dp_size=1,
                weight_version=None,
            ),
        )

        async def run_outputs():
            await TokenizerManager._handle_batch_output(
                manager, _make_batch_str_output("a", 101)
            )
            await TokenizerManager._handle_batch_output(
                manager, _make_batch_str_output("b", 102)
            )

        asyncio.run(run_outputs())

        self.assertEqual([out["text"] for out in state.out_list], ["a", "ab"])
        self.assertEqual(
            [out["output_ids"] for out in state.out_list], [[101], [101, 102]]
        )

    def test_interleave_uses_current_schema_and_preserves_alignment(self):
        manager = SimpleNamespace(
            glm_stream_speculated_tokens=True,
            decode_status={"r0": object(), "r1": object()},
        )
        handler = interleave_batch_token_id_out(lambda _self, output: output)

        first, second = handler(manager, _make_batch_token_id_output())

        self.assertEqual(first.decode_ids, [[101], [201]])
        self.assertEqual(second.decode_ids, [[102], [202]])
        self.assertEqual(first.spec_num_correct_drafts, [1, 2])
        self.assertEqual(first.spec_correct_drafts_histogram, [[0, 1], [0, 0, 1]])
        self.assertEqual(first.completion_tokens, [1, 1])
        self.assertEqual(second.completion_tokens, [2, 2])
        self.assertEqual(first.indexer_topk, [["indexer-00"], ["indexer-10"]])
        self.assertEqual(second.indexer_topk, [["indexer-01"], ["indexer-11"]])
        self.assertEqual(
            first.input_token_logprobs_val,
            [["input-val-0"], ["input-val-1"]],
        )
        self.assertEqual(second.input_token_logprobs_val, [[], []])

    def test_scheduler_optional_outputs_stay_request_aligned(self):
        scheduler = SimpleNamespace(
            stream_interval=1,
            get_loads=lambda _request: None,
            _get_cached_tokens_details=lambda _request: None,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            disaggregation_mode=DisaggregationMode.NULL,
            dp_rank=0,
            attn_tp_rank=0,
            server_args=SimpleNamespace(enable_request_time_stats_logging=False),
            send_to_detokenizer=SimpleNamespace(send_output=MagicMock()),
        )
        reqs = [
            _make_scheduler_req("normal", 101, return_optional=False),
            _make_scheduler_req("optional", 201, return_optional=True),
        ]

        SchedulerOutputProcessorMixin.stream_output_generation(
            scheduler, reqs, return_logprob=False
        )

        output = scheduler.send_to_detokenizer.send_output.call_args.args[0]
        self.assertEqual(output.rids, ["normal", "optional"])
        self.assertEqual(output.output_hidden_states, [None, "hidden-optional"])
        self.assertEqual(output.routed_experts, [None, "experts-optional"])
        self.assertEqual(output.indexer_topk, [None, "indexer-optional"])


if __name__ == "__main__":
    unittest.main()
