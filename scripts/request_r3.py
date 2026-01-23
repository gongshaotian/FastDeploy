import openai


def openai_client():
    ip = "0.0.0.0"
    service_http_port = 8888
    client = openai.Client(
        base_url=f"http://{ip}:{service_http_port}/v1",
        api_key="EMPTY_API_KEY",
    )
    return client


def send_r3_streaming_chat(openai_client, user_id: str = "r3_chat_completion_stream_test_prefixcache"):
    """
    Test streaming chat functionality with the local service
    """
    response = openai_client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": "总结下下面这道美食的菜名：老母鸡汤方便面+煎蛋+青菜的做法可以按照以下步骤进行：\n\n### 泰式风味老母鸡汤方便面\n\n1. **准备材料**：\n\n\t* 老母鸡半只或整只（根据人数调整）\n\t* 方便面一包\n\t* 姜几片\n\t* 葱一根\n\t* 香菜一根（可选）\n\t* 香茅草或柠檬叶（可选，增添泰式风味）\n\t* 椰浆适量（可选）\n\t* 鱼露或生抽适量（调味）\n\t* 盐和胡椒粉适量（调味）\n\n2. **处理食材**：\n\n\t* 老母鸡洗净，斩块。\n\t* 方便面取出备用。\n\t* 姜切片，葱切段，香菜切碎（可选）。\n\n3. **煮鸡汤**：\n\n\t* 将老母鸡块放入锅中，加入足够的水，放入姜片、葱段，大火煮沸后撇去浮沫。\n\t* 转小火慢炖，至少1小时，直到鸡肉软烂，汤色浓白。\n\n4. **调味与配料**：\n\n\t* 根据个人口味，加入适量鱼露或生抽调味。\n\t* 如果喜欢泰式风味，可以加入香茅草或柠檬叶，煮出香味后捞出。\n\t* 加入椰浆（可选），增加汤的醇厚感。\n\n5. **煮方便面**：\n\n\t* 鸡汤煮好后，放入方便面，用筷子拨散，煮至方便面变软。\n\t* 根据方便面的口味，可以调整煮面的时间。\n\n6. **装盘**：\n\n\t* 将煮好的方便面捞出，放入碗中。\n\t* 倒入浓白的鸡汤，撒上葱花或香菜碎（可选）。\n\n### 煎蛋\n\n1. **准备材料**：\n\n\t* 鸡蛋1-2个\n\t* 食用油适量\n\n2. **煎蛋步骤**：\n\n\t* 平底锅加热，倒入适量食用油。\n\t* 油热后，打入鸡蛋，保持中小火。\n\t* 当蛋白凝固，边缘微微翘起时，用铲子轻轻翻面，煎至两面金黄。\n\t* 根据个人口味，可以撒上少许盐或黑胡椒调味。\n\n### 青菜\n\n1. **准备材料**：\n\n\t* 青菜适量（如小白菜、油麦菜等）\n\t* 食用油适量\n\t* 盐适量\n\n2. **炒青菜步骤**：\n\n\t* 青菜洗净，切成适当大小的段。\n\t* 平底锅加热，倒入适量食用油。\n\t* 油热后，放入青菜段，大火快速翻炒。\n\t* 青菜变软后，加入适量盐调味，继续翻炒均匀。\n\t* 炒至青菜断生即可出锅。\n\n### 组合享用\n\n将煮好的泰式风味老母鸡汤方便面、煎蛋和青菜一起装盘，即可享用这道美味又营养的餐食。",
            },
        ],
        temperature=1,
        top_p=0,
        max_tokens=32768,
        seed=13,
        stream=True,
        user=user_id,  # "r3_chat_completion_stream_test",
    )

    return response


if __name__ == "__main__":
    openai_client = openai_client()
    response = send_r3_streaming_chat(openai_client, user_id="r3_chat_completion_stream_test_prefixcache_1")
    output = ""
    for chunk in response:
        output += chunk.choices[0].delta.content
    print("\nr3_chat_completion_stream_test_prefixcache_1\n", output)

    response = send_r3_streaming_chat(openai_client, user_id="r3_chat_completion_stream_test_prefixcache_2")
    output = ""
    for chunk in response:
        output += chunk.choices[0].delta.content
    print("\nr3_chat_completion_stream_test_prefixcache_2\n", output)
