"""
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
"""

import paddle
import triton
import triton.language as tl


@triton.jit
def _save_routing_kernel(
    ROUTING_TABLE_BUFFER_PTR,
    TOPK_IDS_PTR,
    BATCH_ID_PER_TOKEN_PTR,
    TOKEN_RELATIVE_INDICES_PTR,
    # SEQ_LENS_ENCODER_PTR,
    SEQ_LENS_DECODER_PTR,
    # SEQ_LENS_THIS_TIME_PTR,
    LAYER_IDX,
    TOKEN_NUM,
    TOP_K,
    NUM_HIDDEN_LAYERS,
    MAX_MODEL_LEN,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = token_offsets < TOKEN_NUM

    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    topk_ids_ptrs = TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :]
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    topk_vals = tl.load(topk_ids_ptrs, mask=token_mask[:, None])

    batch_ids = tl.load(BATCH_ID_PER_TOKEN_PTR + token_offsets, mask=token_mask)
    token_relative_index = tl.load(TOKEN_RELATIVE_INDICES_PTR + token_offsets, mask=token_mask)
    len_decoder = tl.load(SEQ_LENS_DECODER_PTR + batch_ids, mask=token_mask)

    # [BLOCK_SIZE_M]
    token_seq_pos = len_decoder + token_relative_index

    STRIDE_BUF_SEQ = NUM_HIDDEN_LAYERS * MAX_MODEL_LEN * TOP_K
    STRIDE_BUF_LAYER = MAX_MODEL_LEN * TOP_K
    STRIDE_BUF_TOKEN = TOP_K

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    output_ptrs = (
        ROUTING_TABLE_BUFFER_PTR
        + batch_ids[:, None] * STRIDE_BUF_SEQ
        + LAYER_IDX * STRIDE_BUF_LAYER
        + token_seq_pos[:, None] * STRIDE_BUF_TOKEN
        + k_offsets[None, :]
    )

    pos_mask = token_seq_pos < MAX_MODEL_LEN
    final_mask = token_mask[:, None] & pos_mask[:, None]

    tl.store(output_ptrs, topk_vals, mask=final_mask)


def save_routing_to_buffer(
    routing_table_buffer: paddle.Tensor,  # [max_num_seqs, num_layers, max_len, top_k]
    topk_ids: paddle.Tensor,  # [token_num, top_k]
    batch_id_per_token: paddle.Tensor,  # [token_num]
    # seq_lens_encoder: paddle.Tensor,  # [max_num_seqs]
    seq_lens_decoder: paddle.Tensor,  # [max_num_seqs]
    # seq_lens_this_time: paddle.Tensor,  # [max_num_seqs]
    cu_seqlens_q: paddle.Tensor,  # [max_num_seqs + 1]
    layer_idx: int,
):
    token_num, top_k = topk_ids.shape
    if token_num == 0:
        return

    max_num_seqs, num_hidden_layers, max_model_len, _ = routing_table_buffer.shape
    assert topk_ids.shape[1] == routing_table_buffer.shape[3]
    assert batch_id_per_token.shape[0] == token_num
    assert seq_lens_decoder.shape[0] == max_num_seqs

    token_indices = paddle.arange(token_num, dtype="int32")
    # [0, 3, 4, 10, 12][0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3]
    # -> [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] - [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # -> [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1]
    token_relative_indices = token_indices - cu_seqlens_q.view([-1])[batch_id_per_token].view([-1])

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = top_k  # 值一般很小，直接设为 top_k

    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel[grid](
        routing_table_buffer,
        topk_ids,
        batch_id_per_token,
        token_relative_indices,
        # seq_lens_encoder,
        seq_lens_decoder,
        # seq_lens_this_time,
        LAYER_IDX=layer_idx,
        TOKEN_NUM=token_num,
        TOP_K=top_k,
        NUM_HIDDEN_LAYERS=num_hidden_layers,
        MAX_MODEL_LEN=max_model_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


# max_num_seqs = 4
# num_layers = 1
# max_len = 10
# top_k = 8
# token_num = 12

# routing_table_buffer = paddle.full([max_num_seqs, num_layers, max_len, top_k], -1, dtype="int32")
# topk_ids = paddle.randint(0, 384, [token_num, top_k], dtype="int32")
# batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3], dtype="int32").reshape([-1, 1])
# # seq_lens_encoder = paddle.to_tensor([3, 1, 6, 2], dtype="int32").reshape([-1, 1])
# seq_lens_decoder = paddle.to_tensor([0, 2, 0, 3], dtype="int32").reshape([-1, 1])
# # seq_lens_this_time = paddle.to_tensor([3, 1, 6, 2], dtype="int32").reshape([-1, 1])
# cu_seqlens_q = paddle.to_tensor([0, 3, 4, 10, 12], dtype="int32").reshape([-1, 1])
# current_layer_idx = 0

# save_routing_to_buffer(
#     routing_table_buffer=routing_table_buffer,
#     topk_ids=topk_ids,
#     batch_id_per_token=batch_id_per_token,
#     # seq_lens_encoder=seq_lens_encoder,
#     seq_lens_decoder=seq_lens_decoder,
#     # seq_lens_this_time=seq_lens_this_time,
#     cu_seqlens_q=cu_seqlens_q,
#     layer_idx=current_layer_idx,
# )
