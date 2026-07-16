import pytest
import torch

import sgl_kernel


@pytest.mark.parametrize("num_draft_tokens,has_per_q", [(1, False), (2, True)])
def test_kpool_write_plan(num_draft_tokens: int, has_per_q: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = "cuda"
    bs = 5
    pool_size = 4
    slots_per_page = 16
    max_pages = 8

    write_start = torch.tensor([0, 3, 4, 7, 15], dtype=torch.int32, device=device)
    req_pool_indices = torch.arange(100, 100 + bs, dtype=torch.int64, device=device)
    real_page_table = torch.arange(
        bs * num_draft_tokens * max_pages, dtype=torch.int32, device=device
    ).reshape(bs * num_draft_tokens, max_pages)

    req_out = torch.empty(bs, dtype=torch.int64, device=device)
    write_start_out = torch.empty(bs, dtype=torch.int32, device=device)
    tail_logical_start_out = torch.empty(bs, dtype=torch.int32, device=device)
    write_loc_out = torch.empty(bs, dtype=torch.int64, device=device)

    if has_per_q:
        pool_seqlens_per_q_out = torch.empty(
            bs * num_draft_tokens, dtype=torch.int32, device=device
        )
        seqlens_per_q_out = torch.empty(
            bs * num_draft_tokens, dtype=torch.int32, device=device
        )
    else:
        pool_seqlens_per_q_out = None
        seqlens_per_q_out = None

    sgl_kernel.kpool_write_plan(
        write_start,
        req_pool_indices,
        real_page_table,
        req_out,
        write_start_out,
        tail_logical_start_out,
        write_loc_out,
        pool_seqlens_per_q_out,
        seqlens_per_q_out,
        pool_size,
        num_draft_tokens,
        slots_per_page,
    )

    base_pool = torch.div(write_start, pool_size, rounding_mode="floor")
    expected_tail = base_pool * pool_size
    row = torch.arange(bs, dtype=torch.long, device=device) * num_draft_tokens
    page_group = torch.div(base_pool, slots_per_page, rounding_mode="floor").to(
        torch.long
    )
    expected_loc = real_page_table[row, page_group].to(torch.int64) * slots_per_page
    expected_loc += torch.remainder(base_pool, slots_per_page).to(torch.int64)

    torch.testing.assert_close(req_out, req_pool_indices)
    torch.testing.assert_close(write_start_out, write_start)
    torch.testing.assert_close(tail_logical_start_out, expected_tail)
    torch.testing.assert_close(write_loc_out, expected_loc)

    if has_per_q:
        offsets = torch.arange(num_draft_tokens, dtype=torch.int32, device=device)
        expected_seqlens = (write_start[:, None] + offsets[None, :] + 1).reshape(-1)
        torch.testing.assert_close(seqlens_per_q_out, expected_seqlens)
        torch.testing.assert_close(
            pool_seqlens_per_q_out,
            torch.div(expected_seqlens, pool_size, rounding_mode="floor"),
        )
