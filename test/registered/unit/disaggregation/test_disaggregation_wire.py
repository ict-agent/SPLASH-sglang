import unittest

import numpy as np

from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.utils import filter_kv_indices_for_cp_rank
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestGroupConcurrentContiguous(unittest.TestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_both_empty(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([])), ([], [])
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


class TestCPPageFiltering(unittest.TestCase):
    @staticmethod
    def _mgr(cp_rank, cp_size):
        class Manager:
            pass

        mgr = Manager()
        mgr.attn_cp_rank = cp_rank
        mgr.attn_cp_size = cp_size
        return mgr

    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_uses_request_positions_not_global_page_ids(self):
        pages = self._arr([10, 2, 30, 4, 50, 6])

        rank0_pages, rank0_slice = filter_kv_indices_for_cp_rank(
            self._mgr(0, 2), pages, slice(0, 6), total_pages=6
        )
        rank1_pages, rank1_slice = filter_kv_indices_for_cp_rank(
            self._mgr(1, 2), pages, slice(0, 6), total_pages=6
        )

        np.testing.assert_array_equal(rank0_pages, self._arr([10, 2, 30]))
        self.assertEqual((rank0_slice.start, rank0_slice.stop), (0, 3))
        np.testing.assert_array_equal(rank1_pages, self._arr([4, 50, 6]))
        self.assertEqual((rank1_slice.start, rank1_slice.stop), (3, 6))

    def test_intersects_cp_range_with_chunk_offset(self):
        chunk = self._arr([30, 4, 50])

        rank0_pages, rank0_slice = filter_kv_indices_for_cp_rank(
            self._mgr(0, 2), chunk, slice(2, 5), total_pages=6
        )
        rank1_pages, rank1_slice = filter_kv_indices_for_cp_rank(
            self._mgr(1, 2), chunk, slice(2, 5), total_pages=6
        )

        np.testing.assert_array_equal(rank0_pages, self._arr([30]))
        self.assertEqual((rank0_slice.start, rank0_slice.stop), (2, 3))
        np.testing.assert_array_equal(rank1_pages, self._arr([4, 50]))
        self.assertEqual((rank1_slice.start, rank1_slice.stop), (3, 5))

    def test_empty_intersection_preserves_chunk_start(self):
        pages, index_slice = filter_kv_indices_for_cp_rank(
            self._mgr(1, 2), self._arr([10, 2]), slice(0, 2), total_pages=6
        )

        np.testing.assert_array_equal(pages, self._arr([]))
        self.assertEqual((index_slice.start, index_slice.stop), (0, 0))

    def test_total_pages_default_keeps_legacy_full_chunk_behavior(self):
        pages, index_slice = filter_kv_indices_for_cp_rank(
            self._mgr(1, 2), self._arr([10, 2, 30, 4]), slice(0, 4)
        )

        np.testing.assert_array_equal(pages, self._arr([30, 4]))
        self.assertEqual((index_slice.start, index_slice.stop), (2, 4))


if __name__ == "__main__":
    unittest.main()
