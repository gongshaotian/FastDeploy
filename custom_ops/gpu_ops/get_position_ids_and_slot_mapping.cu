// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"
#include "paddle/extension.h"

__global__ void GetPositionIdsAndSlotMappingKernel(
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const int* __restrict__ seq_lens_this_time,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ block_tables,
    const int bsz,
    const int max_num_blocks,
    const int block_size,
    int64_t* __restrict__ position_ids,
    int64_t* __restrict__ slot_mapping) {
  int current_bid = threadIdx.x;
  if (current_bid >= bsz) return;

  // Calculate the offset of current batch in the position_ids buffer
  int buffer_offset = 0;
  for (int i = 0; i < current_bid; i++) {
    buffer_offset += seq_lens_this_time[i];
  }

  // Calculate the token offset in the current batch
  int token_offset = seq_lens_decoder[current_bid];
  int token_num_this_batch = seq_lens_this_time[current_bid];
  if (token_num_this_batch == 0) return;

  // Write position ids and slot mapping for current batch
#pragma unroll
  for (int i = 0; i < token_num_this_batch; i++) {
    int pos_id = token_offset + i;
    int idx = buffer_offset + i;

    // Write position_id
    position_ids[idx] = pos_id;

    // Calculate slot mapping directly
    int block_idx = pos_id / block_size;
    int block_offset = pos_id % block_size;
    int batch_id = batch_id_per_token[idx];

    // Get block_id from block_tables
    int block_id = block_tables[batch_id * max_num_blocks + block_idx];

    // Calculate slot mapping
    slot_mapping[idx] = static_cast<int64_t>(
        static_cast<int64_t>(block_id) * block_size + block_offset);
  }
}

void GetPositionIdsAndSlotMapping(const paddle::Tensor& seq_lens_encoder,
                                  const paddle::Tensor& seq_lens_decoder,
                                  const paddle::Tensor& seq_lens_this_time,
                                  const paddle::Tensor& batch_id_per_token,
                                  const paddle::Tensor& block_tables,
                                  const paddle::Tensor& position_ids,
                                  const paddle::Tensor& slot_mapping,
                                  const int block_size) {
  const int bsz = seq_lens_this_time.shape()[0];
  const int total_token_num = position_ids.shape()[0];
  const int max_num_blocks = block_tables.shape()[1];

  GetPositionIdsAndSlotMappingKernel<<<1,
                                       bsz,
                                       0,
                                       seq_lens_this_time.stream()>>>(
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      seq_lens_this_time.data<int>(),
      batch_id_per_token.data<int>(),
      block_tables.data<int>(),
      bsz,
      max_num_blocks,
      block_size,
      const_cast<int64_t*>(position_ids.data<int64_t>()),
      const_cast<int64_t*>(slot_mapping.data<int64_t>()));
}

PD_BUILD_STATIC_OP(get_position_ids_and_slot_mapping)
    .Inputs({
        "seq_lens_encoder",
        "seq_lens_decoder",
        "seq_lens_this_time",
        "batch_id_per_token",
        "block_tables",
        "position_ids",
        "slot_mapping",
    })
    .Attrs({"block_size: int"})
    .Outputs({"position_ids_out", "slot_mapping_out"})
    .SetInplaceMap({{"position_ids", "position_ids_out"},
                    {"slot_mapping", "slot_mapping_out"}})
    .SetKernelFn(PD_KERNEL(GetPositionIdsAndSlotMapping));
