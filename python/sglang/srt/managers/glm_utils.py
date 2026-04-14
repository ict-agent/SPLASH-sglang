"""GLM specific utilities for tokenization and streaming outputs."""

import functools

from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
)

# -- Dev Note (xunkai) --
#
# It seems that recv_obj.output_routed_expert (in v0.5.7) is renamed to
# recv_obj.routed_expert in v0.5.10 in commit #b74a57a8.


def _slice_token_aligned_value(value, decode_idx):
    if value is None:
        return None
    if isinstance(value, (str, bytes, dict)):
        return value
    try:
        return value[decode_idx : decode_idx + 1]
    except Exception:
        try:
            return [value[decode_idx]]
        except Exception:
            return value


def interleave_batch_token_id_out(func):
    @functools.wraps(func)
    def wrapper(self, recv_obj: BatchTokenIDOutput):
        if not self.glm_stream_speculated_tokens or len(recv_obj.decode_ids) == 0:
            return func(self, recv_obj)
        has_decode_status = [rid in self.decode_status for rid in recv_obj.rids]
        max_decode_len = max(
            [
                len(decode_ids) if has_decode_status[request_idx] else 1
                for request_idx, decode_ids in enumerate(recv_obj.decode_ids)
            ]
        )
        outputs = []
        for decode_idx in range(max_decode_len):
            rids = []
            finished_reasons = []

            decoded_texts = []
            decode_ids = []
            read_offsets = []
            output_ids = []

            skip_special_tokens = []
            spaces_between_special_tokens = []
            no_stop_trim = []
            prompt_tokens = []
            reasoning_tokens = []
            completion_tokens = []
            cached_tokens = []
            cached_tokens_details = [] if recv_obj.cached_tokens_details else None
            http_worker_ipcs = [] if recv_obj.http_worker_ipcs else None
            spec_verify_ct = []
            spec_accepted_tokens = []
            spec_acceptance_histogram = (
                [] if recv_obj.spec_acceptance_histogram else None
            )
            retraction_counts = []
            time_stats = [] if recv_obj.time_stats else None

            input_token_logprobs_val = [] if recv_obj.input_token_logprobs_val else None
            input_token_logprobs_idx = [] if recv_obj.input_token_logprobs_idx else None
            output_token_logprobs_val = (
                [] if recv_obj.output_token_logprobs_val else None
            )
            output_token_logprobs_idx = (
                [] if recv_obj.output_token_logprobs_idx else None
            )
            input_top_logprobs_val = [] if recv_obj.input_top_logprobs_val else None
            input_top_logprobs_idx = [] if recv_obj.input_top_logprobs_idx else None
            output_top_logprobs_val = [] if recv_obj.output_top_logprobs_val else None
            output_top_logprobs_idx = [] if recv_obj.output_top_logprobs_idx else None
            input_token_ids_logprobs_val = (
                [] if recv_obj.input_token_ids_logprobs_val else None
            )
            input_token_ids_logprobs_idx = (
                [] if recv_obj.input_token_ids_logprobs_idx else None
            )
            output_token_ids_logprobs_val = (
                [] if recv_obj.output_token_ids_logprobs_val else None
            )
            output_token_ids_logprobs_idx = (
                [] if recv_obj.output_token_ids_logprobs_idx else None
            )
            output_token_entropy_val = [] if recv_obj.output_token_entropy_val else None
            output_hidden_states = [] if recv_obj.output_hidden_states else None
            routed_experts = [] if recv_obj.routed_experts else None
            token_steps = [] if recv_obj.token_steps else None
            customized_info = {} if recv_obj.customized_info else None
            dp_ranks = [] if recv_obj.dp_ranks else None

            for request_idx in range(len(recv_obj.rids)):
                split_output_aligned = has_decode_status[request_idx]
                decode_len = (
                    len(recv_obj.decode_ids[request_idx]) if split_output_aligned else 1
                )
                if decode_len <= decode_idx:
                    continue
                rids.append(recv_obj.rids[request_idx])
                finished_reasons.append(
                    recv_obj.finished_reasons[request_idx]
                    if decode_idx == (decode_len - 1)
                    else None
                )
                decoded_texts.append("")
                decode_ids.append(
                    [recv_obj.decode_ids[request_idx][decode_idx]]
                    if split_output_aligned
                    else recv_obj.decode_ids[request_idx]
                )
                output_ids.append(
                    [recv_obj.output_ids[request_idx][decode_idx]]
                    if split_output_aligned
                    else recv_obj.output_ids[request_idx]
                )
                read_offsets.append(recv_obj.read_offsets[request_idx])
                skip_special_tokens.append(recv_obj.skip_special_tokens[request_idx])
                spaces_between_special_tokens.append(
                    recv_obj.spaces_between_special_tokens[request_idx]
                )
                no_stop_trim.append(recv_obj.no_stop_trim[request_idx])

                prompt_tokens.append(recv_obj.prompt_tokens[request_idx])
                reasoning_tokens.append(recv_obj.reasoning_tokens[request_idx])
                _completion_tokens = recv_obj.completion_tokens[request_idx] - (
                    decode_len - 1 - decode_idx
                )
                completion_tokens.append(_completion_tokens)
                cached_tokens.append(recv_obj.cached_tokens[request_idx])
                if cached_tokens_details is not None:
                    cached_tokens_details.append(
                        recv_obj.cached_tokens_details[request_idx]
                    )
                if http_worker_ipcs is not None:
                    http_worker_ipcs.append(recv_obj.http_worker_ipcs[request_idx])
                if request_idx < len(recv_obj.spec_verify_ct):
                    spec_verify_ct.append(
                        min(recv_obj.spec_verify_ct[request_idx], _completion_tokens)
                    )
                if request_idx < len(recv_obj.spec_accepted_tokens):
                    spec_accepted_tokens.append(recv_obj.spec_accepted_tokens[request_idx])
                if spec_acceptance_histogram is not None and request_idx < len(
                    recv_obj.spec_acceptance_histogram
                ):
                    spec_acceptance_histogram.append(
                        recv_obj.spec_acceptance_histogram[request_idx]
                    )
                if request_idx < len(recv_obj.retraction_counts):
                    retraction_counts.append(recv_obj.retraction_counts[request_idx])
                if time_stats is not None:
                    time_stats.append(recv_obj.time_stats[request_idx])
                if dp_ranks is not None:
                    dp_ranks.append(recv_obj.dp_ranks[request_idx])

                for src, dest in [
                    (recv_obj.input_token_logprobs_val, input_token_logprobs_val),
                    (recv_obj.input_token_logprobs_idx, input_token_logprobs_idx),
                    (recv_obj.input_top_logprobs_val, input_top_logprobs_val),
                    (recv_obj.input_top_logprobs_idx, input_top_logprobs_idx),
                    (recv_obj.input_token_ids_logprobs_val, input_token_ids_logprobs_val),
                    (recv_obj.input_token_ids_logprobs_idx, input_token_ids_logprobs_idx),
                ]:
                    if src is not None and decode_idx == 0:
                        dest.append(src[request_idx])

                for src, dest in [
                    (recv_obj.output_token_logprobs_val, output_token_logprobs_val),
                    (recv_obj.output_token_logprobs_idx, output_token_logprobs_idx),
                    (recv_obj.output_top_logprobs_val, output_top_logprobs_val),
                    (recv_obj.output_top_logprobs_idx, output_top_logprobs_idx),
                    (recv_obj.output_token_ids_logprobs_val, output_token_ids_logprobs_val),
                    (recv_obj.output_token_ids_logprobs_idx, output_token_ids_logprobs_idx),
                    (recv_obj.output_token_entropy_val, output_token_entropy_val),
                    (recv_obj.output_hidden_states, output_hidden_states),
                    (recv_obj.routed_experts, routed_experts),
                    (recv_obj.token_steps, token_steps),
                ]:
                    if src is not None:
                        value = src[request_idx]
                        dest.append(
                            _slice_token_aligned_value(value, decode_idx)
                            if split_output_aligned  # in v0.5.7 we don't have this check.
                            else value
                        )

                # Note (xunkai): We by any means split customized-info, but is it correct?
                if customized_info is not None:
                    for key, values in recv_obj.customized_info.items():
                        customized_info.setdefault(key, []).append(
                            _slice_token_aligned_value(values[request_idx], decode_idx)
                            if split_output_aligned
                            else values[request_idx]
                        )

            outputs.append(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_accepted_tokens=spec_accepted_tokens,
                    spec_acceptance_histogram=spec_acceptance_histogram,
                    time_stats=time_stats,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cached_tokens_details=cached_tokens_details,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=output_token_entropy_val,
                    output_hidden_states=output_hidden_states,
                    routed_experts=routed_experts,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    token_steps=token_steps,
                    load=recv_obj.load,
                    dp_ranks=dp_ranks,
                )
            )
        return [func(self, output) for output in outputs]

    return wrapper
