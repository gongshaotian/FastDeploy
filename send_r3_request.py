import os
import shutil
import time

import openai
import paddle


def openai_client():
    client = openai.Client(
        base_url="http://0.0.0.0:8888/v1",
        api_key="EMPTY_API_KEY",
    )
    return client


# ==========================
# Test Rollout Routing Replay
# ==========================
def calculate_routing_ratio(expected_routing: paddle.Tensor, actual_routing: paddle.Tensor) -> float:
    """Caculate routing overlap ratio"""
    assert (
        expected_routing.shape == actual_routing.shape
    ), "Routing shapes not equal. Expected shape {expected_routing.shap} actual shape {actual_routing.shape}."
    expected_routing_length = get_real_routing_length(expected_routing)
    actual_routing_length = get_real_routing_length(actual_routing)

    for i in range(max(expected_routing_length, actual_routing_length)):
        if not paddle.all(paddle.equal(expected_routing[i], actual_routing[i])).item():
            print(f"token index {i}:\n expected_routing:{expected_routing[i]}\n actual_routing: {actual_routing[i]}\n")
    # print(f"Expected routing length {expected_routing_length} actual routing length {actual_routing_length}.")

    assert (
        expected_routing_length == actual_routing_length
    ), f"Routing real lengths do not match. Expected length {expected_routing_length} actual length {actual_routing_length}."
    total_rows, elements_per_row = expected_routing.shape

    mask1 = paddle.any(expected_routing != -1, axis=1)
    mask2 = paddle.any(actual_routing != -1, axis=1)
    valid_mask = mask1 & mask2

    if paddle.sum(valid_mask.cast("int32")) == 0:
        return paddle.to_tensor(0.0)

    valid_expected_routing = expected_routing[valid_mask]  # [n_valid, top_k]
    valid_actual_routing = actual_routing[valid_mask]  # [n_valid, top_k]
    # n_valid = valid_expected_routing.shape[0]

    # valid_expected_routing: [n_valid, top_k, 1], valid_actual_routing: [n_valid, 1, top_k]
    # -> equals: [n_valid, top_k, top_k]
    equals = valid_expected_routing.unsqueeze(2) == valid_actual_routing.unsqueeze(1)

    overlap_mask = paddle.any(equals, axis=2)  # [n_valid, 8]

    overlap_counts = paddle.sum(overlap_mask.cast("float32"), axis=1)  # [n_valid]
    overlap_ratios = overlap_counts / elements_per_row  # [n_valid]

    return paddle.mean(overlap_ratios)


def get_real_routing_length(routing: paddle.Tensor) -> int:
    mask = routing == -1
    mask_float = mask.astype(paddle.float32)
    row_has_true = paddle.any(mask_float, axis=1).astype(paddle.float32)

    first_true_index = paddle.argmax(row_has_true, axis=0)
    if row_has_true.any().item():
        return first_true_index.item()
    else:
        return -1


# Streaming test
def send_r3_streaming_chat(openai_client, user_id: str = ""):
    """
    Test streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
            {
                "role": "assistant",
                "content": "China(Beijing), France(Paris), Australia(Canberra).",
            },
            {"role": "user", "content": "OK, tell more."},
        ],
        temperature=1,
        top_p=0,
        max_tokens=1024,
        stream=True,
        user=user_id,  # "r3_chat_completion_stream_test",
    )

    return response


def send_r3_non_streaming_chat(openai_client, user_id: str = ""):
    """
    Test non-streaming chat functionality with the local service
    """
    # send test request
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=1,
        top_p=0,
        max_tokens=1024,
        stream=False,
        user=user_id,  # "rollout_routing_replay_chat_completion_nonstream_test"
    )

    return response


def generated_base_line_routing_index(openai_client):
    # generate streaming chat routing index
    send_r3_streaming_chat(openai_client, user_id="r3_chat_completion_stream")
    # generate non streaming chat routing index
    send_r3_non_streaming_chat(openai_client, user_id="r3_chat_completion_nonstream")

    # check the routing is generated correctly
    ori_dir = "./routing_replay_output"
    wait_for_file(f"{ori_dir}/r3_chat_completion_stream")
    wait_for_file(f"{ori_dir}/r3_chat_completion_nonstream")

    # move the baseline to the routing_replay_output_baseline folder
    target_dir = "./routing_replay_output_baseline"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.move(f"{ori_dir}/r3_chat_completion_stream", f"{target_dir}/r3_chat_completion_stream")
    shutil.move(f"{ori_dir}/r3_chat_completion_nonstream", f"{target_dir}/r3_chat_completion_nonstream")


def wait_for_file(file_path, timeout=20, check_interval=0.1):
    start_time = time.perf_counter()
    deadline = start_time + timeout

    while True:
        # check timeout or not
        current_time = time.perf_counter()
        if current_time >= deadline:
            return False

        # check file generated
        if os.path.exists(file_path):
            return True

        sleep_time = min(check_interval, deadline - current_time)
        time.sleep(sleep_time)


def test_routing_replay_chat_completion(openai_client):
    """Test rollout routing replay chat completion"""
    moe_layer_num = 37  # EB45:27 EB5:37
    ori_dir = "./routing_replay_output"
    # maybe need to generate baseline routing index
    if not os.path.exists("./routing_replay_output_baseline/r3_chat_completion_stream") or not os.path.exists(
        "./routing_replay_output_baseline/r3_chat_completion_nonstream"
    ):
        generated_base_line_routing_index(openai_client)
    routing_layer_num_1 = len(os.listdir("./routing_replay_output_baseline/r3_chat_completion_stream"))
    routing_layer_num_2 = len(os.listdir("./routing_replay_output_baseline/r3_chat_completion_nonstream"))
    assert (
        routing_layer_num_1 == moe_layer_num
    ), f"routing index number {routing_layer_num_1} should equal to moe layer number {moe_layer_num}"
    assert (
        routing_layer_num_2 == moe_layer_num
    ), f"routing index number {routing_layer_num_2} should equal to moe layer number {moe_layer_num}"

    # test streaming chat
    send_r3_streaming_chat(openai_client, user_id="r3_chat_completion_stream")
    for layer_index in range(moe_layer_num):
        # print(f"Stream chat layer index {layer_index}")
        routing_path = f"r3_chat_completion_stream/layer_{layer_index}.pdtensor"
        wait_for_file(f"./routing_replay_output/{routing_path}")

        generated_routing = paddle.load(f"./routing_replay_output/{routing_path}")
        baseline_routing = paddle.load(f"./routing_replay_output_baseline/{routing_path}")
        overlap_ratio = calculate_routing_ratio(baseline_routing, generated_routing)
        assert (
            overlap_ratio >= 0.999
        ), f"the routing overlap ratio of the layer {layer_index} should be equal to baseline routing index, but got {overlap_ratio}"

    # test non streaming chat
    send_r3_non_streaming_chat(openai_client, user_id="r3_chat_completion_nonstream")
    for layer_index in range(moe_layer_num):
        # print(f"Non-Stream chat layer index {layer_index}")
        routing_path = f"r3_chat_completion_nonstream/layer_{layer_index}.pdtensor"
        wait_for_file(f"./routing_replay_output/{routing_path}")

        generated_routing = paddle.load(f"./routing_replay_output/{routing_path}")
        baseline_routing = paddle.load(f"./routing_replay_output_baseline/{routing_path}")
        overlap_ratio = calculate_routing_ratio(baseline_routing, generated_routing)
        assert (
            overlap_ratio >= 0.999
        ), f"the routing overlap ratio of the layer {layer_index} should be equal to baseline routing index, but got {overlap_ratio}"

    shutil.rmtree("./routing_replay_output")


if __name__ == "__main__":
    client = openai_client()
    for i in range(100):
        print(f"range {i}")
        test_routing_replay_chat_completion(client)
