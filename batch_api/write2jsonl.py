import json
import os
def step1(system_prompt: str, user_prompt: str, index: int, output_file: str = "input.jsonl") -> None:
    """
    ## usage
        your_top10queries: 放前十筆資料
        product_name: 放商品名稱
        index: 第幾筆資料
    """

    # 構建 JSON 結構
    json_object = {
        "custom_id": f"request-{index}", # 這裡要改成for loop編號
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 2000
        }
    }

    # 將 JSON 寫入文件
    with open(output_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(json_object, ensure_ascii=False) + "\n")


    return None