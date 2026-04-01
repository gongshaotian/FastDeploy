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

import time
from typing import Dict

import numpy as np
import paddle
import paddle.distributed as dist
import triton
import triton.language as tl
from paddleformers.utils.log import logger

from fastdeploy.cache_manager.routing_cache_manager import RoutingHostBufferView
from fastdeploy.cache_manager.routing_store import StoreWrapper
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)


@enable_compat_on_triton_kernel
@triton.jit
def _save_routing_kernel(
    ROUTING_REPLAY_TABLE_PTR,
    TOPK_IDS_PTR,
    BATCH_ID_PER_TOKEN_PTR,
    CU_SEQLENS_Q_PTR,
    SEQ_LENS_DECODER_PTR,
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

    k_mask = k_offsets < TOP_K

    topk_ids_ptrs = TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :]
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]

    load_mask = token_mask[:, None] & k_mask[None, :]
    topk_vals = tl.load(topk_ids_ptrs, mask=load_mask)

    batch_ids = tl.load(BATCH_ID_PER_TOKEN_PTR + token_offsets, mask=token_mask)
    pad_mask = token_mask & (batch_ids != -1)
    # [0, 3, 4, 10, 12][0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3]
    # -> [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] - [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # -> [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1]
    start_offsets = tl.load(CU_SEQLENS_Q_PTR + batch_ids, mask=pad_mask)
    token_relative_index = token_offsets - start_offsets

    # [BLOCK_SIZE_M]
    len_decoder = tl.load(SEQ_LENS_DECODER_PTR + batch_ids, mask=pad_mask)
    token_seq_pos = len_decoder + token_relative_index

    STRIDE_BUF_SEQ = MAX_MODEL_LEN * NUM_HIDDEN_LAYERS * TOP_K
    STRIDE_BUF_TOKEN = NUM_HIDDEN_LAYERS * TOP_K
    STRIDE_BUF_LAYER = TOP_K

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    output_ptrs = (
        ROUTING_REPLAY_TABLE_PTR
        + batch_ids[:, None] * STRIDE_BUF_SEQ
        + token_seq_pos[:, None] * STRIDE_BUF_TOKEN
        + LAYER_IDX * STRIDE_BUF_LAYER
        + k_offsets[None, :]
    )

    pos_mask = token_seq_pos < MAX_MODEL_LEN
    pos_mask = pos_mask & pad_mask

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    pos_mask = pos_mask[:, None] & k_mask[None, :]

    final_mask = load_mask & pos_mask

    tl.store(output_ptrs, topk_vals, mask=final_mask)


def save_routing_to_buffer(
    routing_replay_table: paddle.Tensor,  # [max_num_seqs, num_layers, max_len, top_k]
    topk_ids: paddle.Tensor,  # [token_num, top_k]
    batch_id_per_token: paddle.Tensor,  # [token_num, 1]
    seq_lens_decoder: paddle.Tensor,  # [max_num_seqs, 1]
    cu_seqlens_q: paddle.Tensor,  # [max_num_seqs + 1, 1]
    layer_idx: int,
    tp_size: int,
    ep_size: int,
    tp_group: dist.communication.group.Group,
):
    if tp_size > 1 and ep_size > 1:
        token_num_per_rank = topk_ids.shape[0]
        if token_num_per_rank == 0:
            return
        topk_ids_all = paddle.zeros([token_num_per_rank * tp_size, topk_ids.shape[1]], dtype=topk_ids.dtype)
        paddle.distributed.all_gather(topk_ids_all, topk_ids, tp_group)
        topk_ids = topk_ids_all[: batch_id_per_token.shape[0], :]

    token_num, top_k = topk_ids.shape
    max_num_seqs, max_model_len, num_hidden_layers, _ = routing_replay_table.shape
    assert token_num > 0
    assert topk_ids.shape[1] == routing_replay_table.shape[3], (topk_ids.shape[1], routing_replay_table.shape[3])
    assert batch_id_per_token.shape[0] == token_num, (batch_id_per_token.shape[0], token_num)
    assert seq_lens_decoder.shape[0] == max_num_seqs, (seq_lens_decoder.shape[0], max_num_seqs)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = triton.next_power_of_2(top_k)  # top_k

    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel[grid](
        routing_replay_table,
        topk_ids,
        batch_id_per_token,
        cu_seqlens_q,
        seq_lens_decoder,
        LAYER_IDX=layer_idx,
        TOKEN_NUM=token_num,
        TOP_K=top_k,
        NUM_HIDDEN_LAYERS=num_hidden_layers,
        MAX_MODEL_LEN=max_model_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


@enable_compat_on_triton_kernel
@triton.jit
def _save_routing_kernel_v2(
    GPU_ROUTING_BUFFER_PTR,
    TOPK_IDS_PTR,
    LAYER_IDX,
    TOKEN_NUM,
    TOP_K,
    NUM_MOE_LAYERS,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = token_offsets < TOKEN_NUM
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < TOP_K

    load_mask = token_mask[:, None] & k_mask[None, :]
    topk_vals = tl.load(
        TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :],
        mask=load_mask,
    )

    STRIDE_TOKEN = NUM_MOE_LAYERS * TOP_K
    STRIDE_LAYER = TOP_K
    output_ptrs = (
        GPU_ROUTING_BUFFER_PTR + token_offsets[:, None] * STRIDE_TOKEN + LAYER_IDX * STRIDE_LAYER + k_offsets[None, :]
    )
    tl.store(output_ptrs, topk_vals, mask=load_mask)


def save_routing_to_buffer_v2(
    gpu_routing_buffer: paddle.Tensor,
    topk_ids: paddle.Tensor,
    layer_idx: int,
    tp_size: int,
    ep_size: int,
    tp_group: dist.communication.group.Group,
):
    token_num_per_rank = topk_ids.shape[0]
    if token_num_per_rank == 0:
        return
    if tp_size > 1 and ep_size > 1:
        topk_ids_all = paddle.zeros([token_num_per_rank * tp_size, topk_ids.shape[1]], dtype=topk_ids.dtype)
        paddle.distributed.all_gather(topk_ids_all, topk_ids, tp_group)
        topk_ids = topk_ids_all[:token_num_per_rank, :]

    token_num, top_k = topk_ids.shape
    num_moe_layers = gpu_routing_buffer.shape[1]

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = triton.next_power_of_2(top_k)
    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel_v2[grid](
        gpu_routing_buffer,
        topk_ids,
        LAYER_IDX=layer_idx,
        TOKEN_NUM=token_num,
        TOP_K=top_k,
        NUM_MOE_LAYERS=num_moe_layers,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


class RoutingReplayManager:
    """Request level routing replay table manager"""

    def __init__(self, fd_config: FDConfig, block_table, total_block_num):
        self.fd_config = fd_config
        self.block_table = block_table
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.max_model_len = fd_config.model_config.max_model_len
        self.num_moe_layers = fd_config.model_config.num_hidden_layers - fd_config.model_config.moe_layer_start_index
        self.only_last_turn = fd_config.routing_replay_config.only_last_turn
        self.use_fused_put = fd_config.routing_replay_config.use_fused_put
        if fd_config.model_config.architectures[0] == "Glm4MoeForCausalLM":
            self.moe_top_k = fd_config.model_config.num_experts_per_tok
        else:
            self.moe_top_k = fd_config.model_config.moe_k
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank

        # Initialize the routing replay table and routing cache
        self.routing_batch_to_request: Dict[int, str] = {}
        num_experts = fd_config.model_config.moe_num_experts + fd_config.model_config.moe_num_shared_experts
        self.routing_dtype = self.get_routing_dtype(num_experts=num_experts)
        self._init_routing_cache(dtype=self.routing_dtype, total_block_num=total_block_num)
        self.pending_update_positions = None

        # Initialize routing store wrapper
        if self.tp_rank == 0:
            self._store_wrapper = StoreWrapper(
                fd_config=fd_config,
            )
            self._store_wrapper.start_store_warpper()

    def _init_routing_cache(self, dtype: str, total_block_num: int):
        """Initialize the device buffer and host buffer."""

        max_num_kv_tokens = total_block_num * self.fd_config.cache_config.block_size

        # Legacy host cache (kept during transition, will be replaced by SharedMemory routing_host_buffer)
        self._host_cache = paddle.full(
            shape=[max_num_kv_tokens, self.num_moe_layers, self.moe_top_k], fill_value=-1, dtype=dtype, device="cpu"
        )

        # Phase 2: Small GPU transient buffer (replaces the old routing_replay_table)
        max_num_batched_tokens = self.fd_config.scheduler_config.max_num_batched_tokens
        self.gpu_routing_buffer = paddle.full(
            shape=[max_num_batched_tokens, self.num_moe_layers, self.moe_top_k],
            fill_value=-1,
            dtype=dtype,
        )

        # Legacy routing_replay_table kept as alias for backward compatibility during transition
        self.routing_replay_table = self.gpu_routing_buffer

        # Lazy attach to SharedMemory routing_host_buffer (created by Engine in _stop_profile)
        # Engine creates SharedMemory after profiling completes, which is after Worker init.
        # So we defer attachment to the first save_captured_routing() call.
        self.routing_host_view = None
        self._routing_host_view_attach_attempted = False
        self._routing_host_view_shm_name = (
            f"routing_host_buffer.{str(self.fd_config.parallel_config.local_engine_worker_queue_port)}"
        )
        self._routing_host_view_shape = (max_num_kv_tokens, self.num_moe_layers, self.moe_top_k)
        self._routing_host_view_dtype = dtype

        gpu_buffer_bytes = int(np.prod(self.gpu_routing_buffer.shape)) * np.dtype(dtype).itemsize
        logger.info(
            f"[R3] GPU transient routing buffer: {self.gpu_routing_buffer.shape} "
            f"({gpu_buffer_bytes / 1024:.1f} KB), "
            f"host cache: {self._host_cache.shape}"
        )

    def get_routing_dtype(self, num_experts: int, reserved_fill_value: int = 1) -> str:
        """Calculate the minimum number of bits required for storage routing."""
        if num_experts <= 0:
            raise ValueError(f"num_experts must be greater than 0 but got {num_experts}, please check model config.")
        dtype = "uint8"
        total_number = num_experts + reserved_fill_value
        if total_number <= 255:  # uint8: 0~255
            dtype = "uint8"
        elif total_number <= 65535:  # uint16: 0~65,535
            dtype = "uint16"
        elif total_number <= 4294967295:  # uint32: 0~4,294,967,295
            dtype = "uint32"
        else:
            raise ValueError(
                f"The number of experts {num_experts} exceeds the representation range of uint32, please check model config."
            )
        logger.info(f"[R3] Routing replay table dtype: {dtype}")
        return dtype

    def update_host_cache(self, positions: paddle.Tensor, slot_mapping: paddle.Tensor):
        """Update the host cache with new tokens (legacy v1 path)"""
        for batch_id, position in enumerate(positions):
            if len(position) > 0 and len(slot_mapping[batch_id]) > 0:
                routing_ids = self.routing_replay_table[batch_id, position, :, :].contiguous()
                routing_ids = routing_ids.cpu()

                self._host_cache[slot_mapping[batch_id], :, :] = routing_ids

    def _try_attach_routing_host_view(self):
        """Lazily attach to SharedMemory routing_host_buffer on first use."""
        if self._routing_host_view_attach_attempted:
            return
        self._routing_host_view_attach_attempted = True
        try:
            self.routing_host_view = RoutingHostBufferView(
                shape=self._routing_host_view_shape,
                dtype=self._routing_host_view_dtype,
                shm_name=self._routing_host_view_shm_name,
            )
            logger.info(f"[R3] Attached to RoutingHostBuffer SharedMemory: {self._routing_host_view_shm_name}")
        except FileNotFoundError:
            logger.warning(
                f"[R3] RoutingHostBuffer SharedMemory {self._routing_host_view_shm_name} not found. "
                "Falling back to legacy _host_cache (no swap sync)."
            )

    def save_captured_routing(self, num_tokens: int, slot_mapping: np.ndarray):
        """
        After forward, scatter GPU buffer routing data to routing_host_buffer.
        Called in step gap (post_process), not during forward. CUDAGraph compatible.

        Args:
            num_tokens: Number of tokens processed in the current step
            slot_mapping: [num_tokens], each token's routing_host_buffer slot index
        """
        if num_tokens == 0:
            return

        # Lazy attach to SharedMemory (Engine creates it after profiling completes)
        if self.routing_host_view is None and not self._routing_host_view_attach_attempted:
            self._try_attach_routing_host_view()

        # D2H copy: GPU → CPU numpy
        data = self.gpu_routing_buffer[:num_tokens].cpu().numpy()

        if self.routing_host_view is not None:
            # Phase 2: scatter to SharedMemory routing_host_buffer
            self.routing_host_view.scatter(slot_mapping, data)
        else:
            # Fallback: scatter to legacy _host_cache
            self._host_cache[slot_mapping, :, :] = paddle.to_tensor(data, place="cpu")

    def compute_slot_mapping_flat(self, positions) -> np.ndarray:
        """
        Compute flat slot_mapping for all tokens in the step.
        Returns a 1D numpy array of slot indices.
        """
        all_slots = []
        block_size = self.fd_config.cache_config.block_size
        for batch_id, position in enumerate(positions):
            if len(position) == 0:
                continue
            block_table_indices = position // block_size
            token_block_ids = self.block_table[batch_id, block_table_indices]
            block_offset = position % block_size
            token_cache_ids = np.array(token_block_ids) * block_size + block_offset
            all_slots.append(token_cache_ids)
        if all_slots:
            return np.concatenate(all_slots)
        return np.array([], dtype=np.int64)

    def get_token_positions(self, seq_lens_decoder, seq_lens_this_time):
        """Get token position of each sequence in a batch."""
        starts = seq_lens_decoder.numpy()
        increase_num = seq_lens_this_time.numpy()

        positions = []
        for i in range(self.max_num_seqs):
            if seq_lens_this_time[i] == 0:
                positions.append([])
                continue
            repeated_base = np.repeat(starts[i], increase_num[i])
            positions.append(repeated_base + np.arange(0, increase_num[i]))

        return positions

    def compute_slot_mapping(self, positions: np.ndarray):
        """Compute the mapping between token ids and kvcache slots"""
        slot_mapping = []
        for batch_id, position in enumerate(positions):
            if len(position) == 0:
                slot_mapping.append([])
                continue
            block_table_indices = position // self.fd_config.cache_config.block_size
            token_block_ids = self.block_table[batch_id, block_table_indices]
            block_offset = position % self.fd_config.cache_config.block_size

            token_cache_ids = np.array(token_block_ids) * self.fd_config.cache_config.block_size + block_offset
            slot_mapping.append(token_cache_ids)

        return slot_mapping

    def _get_routing_from_cache(self, finished_batch_ids, seq_lens_decoder):
        """
        When request is finished or cleared the length of the request is recorded at seq_lens_decoder
            1. finish the step: after update input, lens = seq_lens_decoder_buffer
            2. clear parameter: after update input, lens = seq_lens_decoder_buffer
        """
        # Get the slot mapping of the request cache.
        current_token_nums = seq_lens_decoder.numpy()
        positions = []
        for batch_id in range(self.max_num_seqs):
            position = []
            if batch_id in finished_batch_ids:
                position = np.arange(0, current_token_nums[batch_id])
            positions.append(position)

        # Collection the cached routing information
        token_cache_ids = self.compute_slot_mapping(positions=positions)
        for slot_map in token_cache_ids:
            if len(slot_map) > 0:
                token_cached_routing = self._host_cache[slot_map, :, :]
                return paddle.transpose(token_cached_routing, [1, 0, 2])
        raise ValueError("No cached routing found")

    def put_finished_batch(
        self,
        finished_batch_ids,
        seq_lens_decoder,
    ):
        finished_batch_ids_list = finished_batch_ids.cpu().tolist()
        for batch_id, finished in enumerate(finished_batch_ids_list):
            if finished:
                assert batch_id in self.routing_batch_to_request.keys()
                # Deregister the request
                request_id = self._deregister_request(batch_id)
                # Put the routing of finished request to store
                self._put_request_to_store(
                    batch_id=batch_id,
                    request_id=request_id,
                    seq_lens_decoder=seq_lens_decoder,
                )
                # Clear the slot of the finished batch
                self._clear_table_slot(batch_id)

    def register_request(self, batch_id: int, request_id: str):
        """
        Register a new request to routing replay table
        Args:
            batch_id: The batch ID of this request
            request_id: The global ID of the request is usually executed by the training process in RL
        """
        # The chunked prefill tasks will be registered repeatedly
        if batch_id in self.routing_batch_to_request:
            if self.routing_batch_to_request[batch_id] == request_id:
                logger.warning(f"[R3] Request {request_id} has been registered at {batch_id}.")
                return
            else:
                raise RuntimeError(
                    f"[R3] The Batch {batch_id} has been registered by request {self.routing_batch_to_request[batch_id]}, now robed by {request_id},"
                )

        # Register the new request
        self.routing_batch_to_request[batch_id] = request_id
        logger.info(f"[R3] Register request {request_id} with batch id {batch_id}")

    def _deregister_request(self, batch_id: int) -> str:
        """
        Deregister a request from routing replay table
        """
        assert batch_id in self.routing_batch_to_request
        return self.routing_batch_to_request.pop(batch_id)

    def _put_request_to_store(
        self,
        batch_id: int,
        request_id: str,
        seq_lens_decoder,
    ):
        if self.tp_rank == 0:
            before_put_request_time = time.perf_counter()

            # Collect the routing of finished request
            batch_buffer = self._get_routing_from_cache(
                finished_batch_ids=[batch_id], seq_lens_decoder=seq_lens_decoder
            )
            rollout_id = self.split_request_id(request_id)

            if self.use_fused_put:
                self._store_wrapper.submit_put_task(routing_indices=batch_buffer, rollout_id=rollout_id)
            else:
                for layer_id in range(self.num_moe_layers):
                    layer_buffer = batch_buffer[layer_id]
                    self._store_wrapper.submit_put_task(
                        routing_indices=layer_buffer, rollout_id=rollout_id, layer_idx=layer_id
                    )

            # Only store the routing of last turn
            if self.only_last_turn:
                self._store_wrapper.submit_clear_prefix_batch_task(rollout_id=rollout_id)

            logger.info(f"[R3] Submit {request_id} time cost: {time.perf_counter() - before_put_request_time}")

    def clear_request(self, batch_id: int):
        """Clear the routing indices of the request"""
        # With gpu_routing_buffer (v2), no per-batch-id slot to clear —
        # buffer is reused each step. Just remove from tracking dict.
        self.routing_batch_to_request.pop(batch_id, None)

    def _clear_table_slot(self, batch_id: int):
        # No-op with gpu_routing_buffer (v2): buffer is linear, reused each step
        pass

    def get_routing_table(self) -> paddle.Tensor:
        return self.gpu_routing_buffer

    def get_gpu_routing_buffer(self) -> paddle.Tensor:
        return self.gpu_routing_buffer

    def split_request_id(self, request_id: str):
        """
        Split the request id to get rollout id.

        request_id: "chatcmpl-request.user-uuid"
        rollout_id: "request.user"
            example: "chatcmpl-xxx_xxx_epoch_15:2:2:1-d9f16c5c-65f6-4815-b44d-14e2c581907c_0" -> "xxx_xxx_epoch_15:2:2:1"
        """
        chat_type, tmp_str = request_id.split("-", 1)
        # NOTE(gongshaotian): only support chatcmpl now
        assert (
            chat_type == "chatcmpl"
        ), "Rollout Routing Replay only supports chatcmpl. Please check whether the request type and userid settings are correct."
        reversed_tmp_str = tmp_str[::-1].split("-", 5)
        rollout_id = reversed_tmp_str[-1][::-1]
        return rollout_id

    def clear_all_request(self):
        """Clear all requests"""
        self.gpu_routing_buffer.fill_(-1)
        self.routing_batch_to_request = {}
