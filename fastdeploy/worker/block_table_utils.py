import numpy as np
import paddle


def get_token_positions(seq_lens_decoder: paddle.Tensor, seq_lens_this_time: paddle.Tensor, max_num_seqs: int):
    """Get token position of each sequence in a batch."""
    print("seq_lens_decoder", seq_lens_decoder)
    print("seq_lens_this_time", seq_lens_this_time)
    starts = seq_lens_decoder.numpy()[:, 0]
    increase_num = seq_lens_this_time.numpy()[:, 0]

    positions = []
    for i in range(max_num_seqs):
        if seq_lens_this_time[i] == 0:
            positions.append([])
            continue
        repeated_base = np.repeat(starts[i], increase_num[i])
        positions.append(repeated_base + np.arange(0, increase_num[i]))

    return positions


def compute_slot_mapping(block_table, positions: np.ndarray, block_size: int = 64):
    """ """
    slot_mapping = []
    for batch_id, position in enumerate(positions):
        print("position", position)
        if len(position) == 0:
            slot_mapping.append([])
            continue
        block_table_indices = position // block_size
        print("block_table_indices", block_table_indices)
        token_block_ids = block_table[batch_id, block_table_indices]
        block_offset = position % block_size

        token_cache_ids = np.array(token_block_ids) * block_size + block_offset
        slot_mapping.append(token_cache_ids)

    print("slot_mapping", slot_mapping)
    return slot_mapping


def get_token_cache_ids(finished_batch_ids, seq_lens_decoder, seq_lens_this_time, block_table, block_size: int = 64):
    """ """
    current_token_nums = seq_lens_decoder.numpy()[:, 0] + seq_lens_this_time.numpy()[:, 0]

    positions = []
    for batch_id in range(len(seq_lens_decoder)):
        position = []
        if batch_id in finished_batch_ids:
            position = np.arange(0, current_token_nums[batch_id])
        positions.append(position)

    return compute_slot_mapping(block_table=block_table, positions=positions, block_size=block_size)
