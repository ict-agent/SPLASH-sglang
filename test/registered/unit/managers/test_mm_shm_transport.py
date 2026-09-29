import gc
import os
import pickle
import unittest
import weakref
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import (
    ShmOwnerTable,
    ShmPointerMMData,
    unwrap_shm_features,
    wrap_shm_features,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestMultimodalShmTransport(unittest.TestCase):
    @patch("sglang.srt.managers.mm_utils._get_is_default_transport", return_value=False)
    @patch(
        "sglang.srt.managers.mm_utils.get_global_server_args",
        return_value=SimpleNamespace(skip_tokenizer_init=False),
    )
    def test_batch_wraps_feature_and_precomputed_embeddings(
        self, _server_args, _transport
    ):
        feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        embedding = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
        feature_item = SimpleNamespace(feature=feature, precomputed_embeddings=None)
        embedding_item = SimpleNamespace(
            feature=None, precomputed_embeddings=embedding
        )
        request = SimpleNamespace(
            mm_inputs=SimpleNamespace(mm_items=[feature_item, embedding_item])
        )
        batch = SimpleNamespace(batch=[request])

        wrapped = wrap_shm_features(batch)
        self.assertIs(feature_item.feature, feature)
        self.assertIs(embedding_item.precomputed_embeddings, embedding)
        wrapped_items = wrapped.batch[0].mm_inputs.mm_items
        self.assertIsInstance(wrapped_items[0].feature, ShmPointerMMData)
        self.assertIsInstance(
            wrapped_items[1].precomputed_embeddings, ShmPointerMMData
        )

        # Pickle round-trip models tokenizer ZMQ plus scheduler object broadcast.
        received = pickle.loads(pickle.dumps(wrapped))
        unwrap_shm_features(received)
        received_items = received.batch[0].mm_inputs.mm_items
        torch.testing.assert_close(received_items[0].feature, feature)
        torch.testing.assert_close(
            received_items[1].precomputed_embeddings, embedding
        )

    def test_materialize_clones_and_releases_receiver_mapping(self):
        tensor = torch.arange(8, dtype=torch.float32)
        owner = ShmPointerMMData(tensor)
        pointer = pickle.loads(pickle.dumps(owner))

        self.assertIsNone(pointer.tensor)
        self.assertIsNone(pointer._shm_buffer)
        pointer._ensure_open()
        shm_data_ptr = pointer.tensor.data_ptr()
        mapping_ref = (
            weakref.ref(pointer._shm_buffer) if os.name == "posix" else None
        )
        materialized = pointer.materialize()

        self.assertNotEqual(materialized.data_ptr(), shm_data_ptr)
        torch.testing.assert_close(materialized, tensor)
        self.assertIsNone(pointer.tensor)
        self.assertIsNone(pointer._shm_buffer)
        gc.collect()
        if mapping_ref is not None:
            self.assertIsNone(mapping_ref())
        owner.release()
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=owner.shm_name)

    @patch("sglang.srt.managers.mm_utils._get_is_default_transport", return_value=False)
    @patch(
        "sglang.srt.managers.mm_utils.get_global_server_args",
        return_value=SimpleNamespace(skip_tokenizer_init=False),
    )
    def test_zero_copy_can_be_disabled(self, _server_args, _transport):
        expected = torch.arange(8, dtype=torch.float32)
        owner = ShmPointerMMData(expected)
        pointer = pickle.loads(pickle.dumps(owner))
        self.assertIsNone(pointer.tensor)
        request = SimpleNamespace(
            mm_inputs=SimpleNamespace(
                mm_items=[SimpleNamespace(feature=pointer, precomputed_embeddings=None)]
            )
        )

        with envs.SGLANG_MM_SHM_ZERO_COPY.override(False):
            unwrap_shm_features(request)

        materialized = request.mm_inputs.mm_items[0].feature
        torch.testing.assert_close(materialized, expected)
        self.assertIsNone(pointer.tensor)
        self.assertTrue(pointer._released)
        owner.release()

    @unittest.skipUnless(os.name == "posix", "POSIX SHM receiver path")
    def test_receiver_is_untracked_and_read_only(self):
        owner = ShmPointerMMData(torch.arange(8, dtype=torch.float32))
        payload = pickle.dumps(owner)

        with patch(
            "sglang.srt.managers.mm_utils.resource_tracker.register"
        ) as register:
            pointer = pickle.loads(payload)

        register.assert_not_called()
        self.assertIsNone(pointer._shm_buffer)
        pointer._ensure_open()
        self.assertTrue(memoryview(pointer._shm_buffer).readonly)
        pointer.release()
        owner.release()

    def test_borrowed_view_keeps_shm_mapping_alive(self):
        owner = ShmPointerMMData(torch.arange(8, dtype=torch.float32))
        pointer = pickle.loads(pickle.dumps(owner))
        pointer._ensure_open()
        mapping_ref = weakref.ref(pointer._shm_buffer)
        tensor = pointer.borrow()
        view = tensor.detach()[1:]
        owner.release()

        del tensor
        gc.collect()

        self.assertIsNotNone(mapping_ref())
        torch.testing.assert_close(view, torch.arange(1, 8, dtype=torch.float32))

        del view
        gc.collect()
        self.assertIsNone(mapping_ref())

    def test_eight_receivers_borrow_one_owner_until_request_release(self):
        expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        owner = ShmPointerMMData(expected)
        payload = pickle.dumps(owner)
        receivers = [pickle.loads(payload) for _ in range(8)]
        borrowed = [receiver.borrow() for receiver in receivers]

        table = ShmOwnerTable()
        request = SimpleNamespace(
            mm_inputs=SimpleNamespace(
                mm_items=[
                    SimpleNamespace(feature=None, precomputed_embeddings=owner)
                ]
            )
        )
        owner_handle = table.register(request)
        table.release(owner_handle)

        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=owner.shm_name)
        for tensor in borrowed:
            torch.testing.assert_close(tensor, expected)

    def test_receiver_can_be_repickled_for_scheduler_broadcast(self):
        expected = torch.arange(8, dtype=torch.float32)
        owner = ShmPointerMMData(expected)
        first_receiver = pickle.loads(pickle.dumps(owner))
        second_receiver = pickle.loads(pickle.dumps(first_receiver))

        self.assertIsNone(first_receiver.tensor)
        self.assertIsNone(second_receiver.tensor)
        torch.testing.assert_close(first_receiver.borrow(), expected)
        torch.testing.assert_close(second_receiver.borrow(), expected)
        owner.release()

        with self.assertRaisesRegex(RuntimeError, "released shared-memory tensor"):
            pickle.dumps(first_receiver)

    def test_receiver_does_not_open_shm_before_consumption(self):
        owner = ShmPointerMMData(torch.arange(8, dtype=torch.float32))
        payload = pickle.dumps(owner)

        with patch("sglang.srt.managers.mm_utils._open_shm_for_receiver") as opener:
            receiver = pickle.loads(payload)
            pickle.dumps(receiver)
            receiver.release()

        opener.assert_not_called()
        owner.release()

    def test_owner_table_uses_independent_internal_handles(self):
        owner = ShmPointerMMData(torch.arange(8, dtype=torch.float32))
        request = SimpleNamespace(
            mm_inputs=SimpleNamespace(
                mm_items=[SimpleNamespace(feature=owner, precomputed_embeddings=None)]
            )
        )
        table = ShmOwnerTable()
        first_handle = table.register(request)
        second_handle = table.register(request)
        self.assertNotEqual(first_handle, second_handle)

        table.release(first_handle)
        probe = shared_memory.SharedMemory(name=owner.shm_name)
        probe.close()
        table.release(second_handle)

        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=owner.shm_name)

    def test_tokenizer_dispatch_registers_owner_and_drops_source_tensor(self):
        tensor = torch.arange(8, dtype=torch.float32)
        tokenized_obj = SimpleNamespace(
            rid="request",
            time_stats=Mock(),
            mm_inputs=SimpleNamespace(
                mm_items=[
                    SimpleNamespace(feature=None, precomputed_embeddings=tensor)
                ]
            ),
        )
        manager = object.__new__(TokenizerManager)
        manager.shm_owner_table = ShmOwnerTable()
        manager.send_to_scheduler = Mock()
        state = SimpleNamespace(shm_owner_handle=None)
        manager.rid_to_state = {"request": state}

        with (
            patch(
                "sglang.srt.managers.mm_utils._get_is_default_transport",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.mm_utils.get_global_server_args",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
        ):
            manager._send_one_request(tokenized_obj)

        sent_obj = manager.send_to_scheduler.send_pyobj.call_args.args[0]
        owner = sent_obj.mm_inputs.mm_items[0].precomputed_embeddings
        self.assertIsInstance(owner, ShmPointerMMData)
        self.assertIsNone(tokenized_obj.mm_inputs)
        self.assertEqual(len(manager.shm_owner_table), 1)
        self.assertIsNotNone(state.shm_owner_handle)

        manager._release_shm_owner(state)
        self.assertIsNone(state.shm_owner_handle)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=owner.shm_name)

    def test_tokenizer_dispatch_failure_releases_owner(self):
        tensor = torch.arange(8, dtype=torch.float32)
        tokenized_obj = SimpleNamespace(
            rid="request",
            time_stats=Mock(),
            mm_inputs=SimpleNamespace(
                mm_items=[
                    SimpleNamespace(feature=None, precomputed_embeddings=tensor)
                ]
            ),
        )
        manager = object.__new__(TokenizerManager)
        manager.shm_owner_table = ShmOwnerTable()
        manager.send_to_scheduler = Mock()
        manager.send_to_scheduler.send_pyobj.side_effect = RuntimeError("send failed")
        state = SimpleNamespace(shm_owner_handle=None)
        manager.rid_to_state = {"request": state}

        with (
            patch(
                "sglang.srt.managers.mm_utils._get_is_default_transport",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.mm_utils.get_global_server_args",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
            self.assertRaisesRegex(RuntimeError, "send failed"),
        ):
            manager._send_one_request(tokenized_obj)

        sent_obj = manager.send_to_scheduler.send_pyobj.call_args.args[0]
        owner = sent_obj.mm_inputs.mm_items[0].precomputed_embeddings
        self.assertEqual(len(manager.shm_owner_table), 0)
        self.assertIsNone(state.shm_owner_handle)
        self.assertIsNotNone(tokenized_obj.mm_inputs)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=owner.shm_name)

    def test_unserialized_pointer_unlinks_on_delete(self):
        pointer = ShmPointerMMData(torch.arange(8, dtype=torch.float32))
        shm_name = pointer.shm_name

        del pointer
        gc.collect()

        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)


if __name__ == "__main__":
    unittest.main()
