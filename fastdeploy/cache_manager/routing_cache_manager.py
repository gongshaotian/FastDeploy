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

import multiprocessing
import multiprocessing.shared_memory

import numpy as np
from paddleformers.utils.log import logger


class RoutingHostBuffer:
    """
    Manages routing_host_buffer (corresponds to KVCache GPU cache).
    Indexed by gpu_block_id * block_size + offset.
    Shared across processes via POSIX SharedMemory.
    Each DP rank creates its own instance; name includes dp_suffix.
    """

    def __init__(
        self, num_gpu_blocks: int, block_size: int, num_moe_layers: int, top_k: int, dtype: str, dp_suffix: str = ""
    ):
        max_num_gpu_tokens = num_gpu_blocks * block_size
        self.shape = (max_num_gpu_tokens, num_moe_layers, top_k)
        self.dtype = np.dtype(dtype)
        self.block_size = block_size
        total_bytes = int(np.prod(self.shape)) * self.dtype.itemsize

        self.shm_name = f"routing_host_buffer.{dp_suffix}"
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=max(total_bytes, 1), name=self.shm_name
        )
        self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.buffer[:] = 0xFF if dtype == "uint8" else 0  # -1 for uint8

        logger.info(
            f"[R3] Created RoutingHostBuffer: shape={self.shape}, "
            f"size={total_bytes / 1024:.1f} KB, name={self.shm_name}"
        )

    def close(self):
        self.shm.close()
        self.shm.unlink()


class RoutingHostBufferView:
    """Read/write view of routing_host_buffer (cross-process, does not own)."""

    def __init__(self, shape, dtype: str, shm_name: str):
        self.shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=False)
        self.dtype = np.dtype(dtype)
        self.buffer = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)

    def scatter(self, slot_mapping: np.ndarray, data: np.ndarray):
        """Scatter GPU buffer data to corresponding slots (Worker calls this)."""
        self.buffer[slot_mapping] = data

    def gather(self, slot_mapping: np.ndarray) -> np.ndarray:
        """Gather data from specified slots (TokenProcessor calls this)."""
        return self.buffer[slot_mapping].copy()

    def close(self):
        self.shm.close()


class RoutingSwapBuffer:
    """
    Manages routing_swap_buffer (corresponds to KVCache CPU cache).
    Indexed by cpu_block_id * block_size + offset.
    CacheTransferManager creates this; shared via SharedMemory.
    """

    def __init__(
        self, num_cpu_blocks: int, block_size: int, num_moe_layers: int, top_k: int, dtype: str, dp_suffix: str = ""
    ):
        max_num_cpu_tokens = num_cpu_blocks * block_size
        self.shape = (max_num_cpu_tokens, num_moe_layers, top_k)
        self.dtype = np.dtype(dtype)
        self.block_size = block_size
        total_bytes = int(np.prod(self.shape)) * self.dtype.itemsize

        self.shm_name = f"routing_swap_buffer.{dp_suffix}"
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=max(total_bytes, 1), name=self.shm_name
        )
        self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.buffer[:] = 0xFF if dtype == "uint8" else 0

        logger.info(
            f"[R3] Created RoutingSwapBuffer: shape={self.shape}, "
            f"size={total_bytes / 1024:.1f} KB, name={self.shm_name}"
        )

    def close(self):
        self.shm.close()
        self.shm.unlink()


class RoutingSwapBufferView:
    """Read/write view of routing_swap_buffer (cross-process, does not own)."""

    def __init__(self, shape, dtype: str, shm_name: str):
        self.shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=False)
        self.dtype = np.dtype(dtype)
        self.buffer = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)

    def close(self):
        self.shm.close()
