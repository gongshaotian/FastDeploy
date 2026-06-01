# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import (
    get_position_ids,
    get_position_ids_and_slot_mapping,
)


class TestGetPositionIdsAndSlotMapping(unittest.TestCase):
    """Test the fused get_position_ids_and_slot_mapping kernel.

    Variable meanings:
    - seq_lens_encoder: 0 if decode stage, else prefill length in current step
    - seq_lens_decoder: total context length (processed history, prefill + decode)
    - seq_lens_this_time: tokens to process in current step
    """

    def setUp(self):
        np.random.seed(42)
        paddle.set_device("gpu")

    def _compute_position_ids_and_slot_mapping_old(
        self,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        block_tables,
        block_size,
    ):
        """Old implementation for comparison."""
        sum_token_num = int(seq_lens_this_time.numpy().sum())

        # get_position_ids expects int32, so use int32 and then cast to int64
        position_ids_int32 = paddle.zeros([sum_token_num], dtype="int32")
        get_position_ids(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, position_ids_int32)

        block_idx = position_ids_int32 // block_size
        block_ids = block_tables[batch_id_per_token[:sum_token_num], block_idx]
        block_offset = position_ids_int32 % block_size
        slot_mapping = (block_ids * block_size + block_offset).cast(paddle.int64)

        # Cast position_ids to int64 for comparison with new kernel
        position_ids = position_ids_int32.cast(paddle.int64)

        return position_ids.numpy(), slot_mapping.numpy()

    def _compute_position_ids_and_slot_mapping_new(
        self,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        block_tables,
        block_size,
    ):
        """New fused kernel implementation."""
        sum_token_num = int(seq_lens_this_time.numpy().sum())
        # Create output buffers (int64 for kernel compatibility)
        position_ids = paddle.zeros([sum_token_num], dtype="int64")
        slot_mapping = paddle.zeros([sum_token_num], dtype="int64")

        # Kernel writes directly to buffers
        get_position_ids_and_slot_mapping(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            block_tables,
            position_ids,
            slot_mapping,
            block_size,
        )

        return position_ids.numpy(), slot_mapping.numpy()

    def _generate_batch_id_per_token(self, seq_lens_this_time, bsz):
        """Generate batch_id_per_token based on seq_lens_this_time."""
        total_tokens = int(seq_lens_this_time.numpy().sum())
        batch_id_per_token = np.zeros([total_tokens], dtype=np.int32)
        offset = 0
        for bid in range(bsz):
            seq_len = int(seq_lens_this_time[bid].numpy())
            batch_id_per_token[offset : offset + seq_len] = bid
            offset += seq_len
        return paddle.to_tensor(batch_id_per_token, dtype="int32", place=paddle.CUDAPlace(0))

    def _generate_block_tables(self, bsz, max_num_blocks):
        """Generate block_tables with sequential block ids for reproducibility."""
        block_tables = np.arange(bsz * max_num_blocks, dtype=np.int32).reshape(bsz, max_num_blocks)
        return paddle.to_tensor(block_tables, dtype="int32", place=paddle.CUDAPlace(0))

    def test_single_batch_decode(self):
        """Test single batch in decode stage."""
        # Decode stage: already processed 10 tokens, now decode 1 more
        seq_lens_encoder = paddle.to_tensor([0], dtype="int32")  # decode stage
        seq_lens_decoder = paddle.to_tensor([10], dtype="int32")  # history length
        seq_lens_this_time = paddle.to_tensor([1], dtype="int32")  # current step

        batch_id_per_token = paddle.to_tensor([0], dtype="int32")
        block_tables = self._generate_block_tables(1, 100)
        block_size = 64

        pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )
        # Expected: pos_old=[10], slot_old=[10] (block_id=0, block_offset=10, slot=0*64+10=10)
        # logger.info(f"test_single_batch_decode: pos_old={pos_old}, slot_old={slot_old}")
        pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )

        np.testing.assert_array_equal(pos_old, pos_new, "position_ids mismatch")
        np.testing.assert_array_equal(slot_old, slot_new, "slot_mapping mismatch")

        # Verify position_id starts from seq_lens_decoder
        self.assertEqual(pos_new[0], 10)

    def test_single_batch_prefill(self):
        """Test single batch in prefill stage."""
        # Prefill stage: no history, processing 5 tokens
        seq_lens_encoder = paddle.to_tensor([5], dtype="int32")  # prefill length
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")  # no history
        seq_lens_this_time = paddle.to_tensor([5], dtype="int32")  # current step

        batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 0], dtype="int32")
        block_tables = self._generate_block_tables(1, 100)
        block_size = 64

        pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )
        # Expected: pos_old=[0,1,2,3,4], slot_old=[0,1,2,3,4] (all in block 0)
        # logger.info(f"test_single_batch_prefill: pos_old={pos_old}, slot_old={slot_old}")
        pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )

        np.testing.assert_array_equal(pos_old, pos_new, "position_ids mismatch")
        np.testing.assert_array_equal(slot_old, slot_new, "slot_mapping mismatch")

        # Verify position_ids start from 0
        np.testing.assert_array_equal(pos_new, np.array([0, 1, 2, 3, 4]))

    def test_multiple_batches_decode(self):
        """Test multiple batches all in decode stage."""
        # Batch 0: history 10, now 1
        # Batch 1: history 20, now 2
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")  # both decode
        seq_lens_decoder = paddle.to_tensor([10, 20], dtype="int32")  # history lengths
        seq_lens_this_time = paddle.to_tensor([1, 2], dtype="int32")  # current step

        batch_id_per_token = self._generate_batch_id_per_token(seq_lens_this_time, 2)
        block_tables = self._generate_block_tables(2, 100)
        block_size = 64

        pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )
        # Expected: pos_old=[10,20,21]
        # Batch 0: position_id=10, block_id=0, block_offset=10, slot=10
        # Batch 1: position_ids=[20,21], batch_id=1, block_tables[1][0]=100
        #         slot[1]=100*64+20=6420, slot[2]=100*64+21=6421
        # logger.info(f"test_multiple_batches_decode: pos_old={pos_old}, slot_old={slot_old}")
        pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )

        np.testing.assert_array_equal(pos_old, pos_new, "position_ids mismatch")
        np.testing.assert_array_equal(slot_old, slot_new, "slot_mapping mismatch")

        # Batch 0: position_id = 10
        # Batch 1: position_ids = [20, 21]
        np.testing.assert_array_equal(pos_new, np.array([10, 20, 21]))

    def test_different_block_sizes(self):
        """Test with different block sizes."""
        for block_size in [1, 8, 16, 32, 64]:
            with self.subTest(block_size=block_size):
                seq_lens_encoder = paddle.to_tensor([0], dtype="int32")  # decode
                seq_lens_decoder = paddle.to_tensor([10], dtype="int32")  # history
                seq_lens_this_time = paddle.to_tensor([5], dtype="int32")  # current
                batch_id_per_token = paddle.to_tensor([0] * 5, dtype="int32")
                block_tables = self._generate_block_tables(1, 100)

                pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
                    seq_lens_encoder,
                    seq_lens_decoder,
                    seq_lens_this_time,
                    batch_id_per_token,
                    block_tables,
                    block_size,
                )
                # Expected: pos_old=[10,11,12,13,14]
                # For block_size=64: block_id=0, slot=[10,11,12,13,14]
                # For block_size=16: block_id=0 for all (10-14<16), slot=[10,11,12,13,14]
                # logger.info(f"test_different_block_sizes[{block_size}]: pos_old={pos_old}, slot_old={slot_old}")
                pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
                    seq_lens_encoder,
                    seq_lens_decoder,
                    seq_lens_this_time,
                    batch_id_per_token,
                    block_tables,
                    block_size,
                )

                np.testing.assert_array_equal(pos_old, pos_new)
                np.testing.assert_array_equal(slot_old, slot_new)

    def test_block_boundary_crossing(self):
        """Test tokens crossing block boundaries."""
        # block_size=64, history=60, so position_ids will be [60, 61, 62, 63, 64]
        # This crosses the block boundary (60-63 in block 0, 64 in block 1)
        seq_lens_encoder = paddle.to_tensor([0], dtype="int32")  # decode
        seq_lens_decoder = paddle.to_tensor([60], dtype="int32")  # history
        seq_lens_this_time = paddle.to_tensor([5], dtype="int32")  # current
        batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 0], dtype="int32")
        block_tables = self._generate_block_tables(1, 100)
        block_size = 64

        pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )
        # Expected: pos_old=[60,61,62,63,64]
        # position 60-63: block_id=0, block_offset=60-63, slot=60-63
        # position 64: block_id=1, block_offset=0, slot=64
        # logger.info(f"test_block_boundary_crossing: pos_old={pos_old}, slot_old={slot_old}")
        pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )

        np.testing.assert_array_equal(pos_old, pos_new)
        np.testing.assert_array_equal(slot_old, slot_new)

        # Verify position_ids
        np.testing.assert_array_equal(pos_new, np.array([60, 61, 62, 63, 64]))

    def test_large_batch(self):
        """Test with larger batch size."""
        bsz = 16
        # All in decode stage
        seq_lens_encoder = paddle.to_tensor([0] * bsz, dtype="int32")
        seq_lens_decoder = paddle.to_tensor(np.random.randint(0, 100, size=bsz), dtype="int32")
        seq_lens_this_time = paddle.to_tensor(np.random.randint(1, 5, size=bsz), dtype="int32")

        batch_id_per_token = self._generate_batch_id_per_token(seq_lens_this_time, bsz)
        block_tables = self._generate_block_tables(bsz, 100)
        block_size = 64

        pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )
        # Too many tokens to list expected values
        # logger.info(f"test_large_batch: shape pos_old={pos_old.shape}, slot_old={slot_old.shape}")
        pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )

        np.testing.assert_array_equal(pos_old, pos_new)
        np.testing.assert_array_equal(slot_old, slot_new)

    def test_empty_batch(self):
        """Test with some batches having zero tokens this step."""
        # Batch 0: decode (1 token)
        # Batch 1: skip (0 tokens)
        # Batch 2: decode (2 tokens)
        seq_lens_encoder = paddle.to_tensor([0, 0, 0], dtype="int32")  # all decode
        seq_lens_decoder = paddle.to_tensor([10, 20, 5], dtype="int32")  # history
        seq_lens_this_time = paddle.to_tensor([1, 0, 2], dtype="int32")  # current

        batch_id_per_token = self._generate_batch_id_per_token(seq_lens_this_time, 3)
        block_tables = self._generate_block_tables(3, 100)
        block_size = 64

        pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )
        # Expected: pos_old=[10,5,6]
        # Batch 0: position_id=10, batch_id=0, block_id=0, slot=10
        # Batch 2: position_ids=[5,6], batch_id=2, block_tables[2][0]=200
        #         slot[1]=200*64+5=12805, slot[2]=200*64+6=12806
        # logger.info(f"test_empty_batch: pos_old={pos_old}, slot_old={slot_old}")
        pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )

        np.testing.assert_array_equal(pos_old, pos_new)
        np.testing.assert_array_equal(slot_old, slot_new)

        # Batch 0: position_id = 10
        # Batch 1: skipped
        # Batch 2: position_ids = [5, 6]
        np.testing.assert_array_equal(pos_new, np.array([10, 5, 6]))

    def test_mtp_scenario(self):
        """Test MTP scenario where seq_lens_this_time varies per batch."""
        # All in decode stage, different accepted tokens per batch
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")  # decode
        seq_lens_decoder = paddle.to_tensor([10, 20], dtype="int32")  # history
        # Batch 0: 2 accepted tokens, Batch 1: 1 accepted token
        seq_lens_this_time = paddle.to_tensor([2, 1], dtype="int32")

        batch_id_per_token = self._generate_batch_id_per_token(seq_lens_this_time, 2)
        block_tables = self._generate_block_tables(2, 100)
        block_size = 64

        pos_old, slot_old = self._compute_position_ids_and_slot_mapping_old(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )
        # Expected: pos_old=[10,11,20]
        # Batch 0: position_ids=[10,11], batch_id=0, block_id=0, slot=[10,11]
        # Batch 1: position_ids=[20], batch_id=1, block_tables[1][0]=100
        #         slot[2]=100*64+20=6420
        # logger.info(f"test_mtp_scenario: pos_old={pos_old}, slot_old={slot_old}")
        pos_new, slot_new = self._compute_position_ids_and_slot_mapping_new(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, batch_id_per_token, block_tables, block_size
        )

        np.testing.assert_array_equal(pos_old, pos_new)
        np.testing.assert_array_equal(slot_old, slot_new)

        # Batch 0: position_ids = [10, 11]
        # Batch 1: position_id = [20]
        np.testing.assert_array_equal(pos_new, np.array([10, 11, 20]))


if __name__ == "__main__":
    unittest.main()
