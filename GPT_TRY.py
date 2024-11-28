# Import libraries
import torch
import os
import csv
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import get_topk_items
from opencc import OpenCC
import pickle
import random
from gradio_client import Client
from openai import OpenAI
import threading

import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

t2s = OpenCC('t2s')
s2t = OpenCC('s2t')

# Download the dataset if it does not exist
import get_dataset  # This import itself will download the dataset, DO NOT REMOVE

# Define argparse for handling arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with various configurations")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help="Name of the model")
    parser.add_argument('--top_k', type=int, default=10, help="Top K related items to retrieve")
    parser.add_argument('--inference_file', type=str, default='./data/test.pickle', help="Input file for inference")
    # parser.add_argument('--dtype', type=str, default='int8', choices=['int8', 'int4'], help="Data type for model precision (int8 or int4)")
    parser.add_argument('--num_inference', type=int, default=-1, help="Number of items to infer, -1 means all")
    parser.add_argument('--temperature', type=float, default=5e-6, help="Temperature for generation")
    # parser.add_argument('--use_qwen_api', type=bool, default=False, help="Whether to use Qwen API for inference")
    parser.add_argument('-hw', '--hw_matching', action='store_true', help="Run in HW mode")
    # parser.add_argument('-m', '--match', action='store_true', help="Run in interactive match mode")
    # parser.add_argument('-a', '--automatch', action='store_true', help="Run in auto match mode")
    # parser.add_argument('--input1', type = str, help="input product name 1")
    # parser.add_argument('--input2', type = str, help="input product name 2")

    return parser.parse_args()

# Load items dataset
def load_items(path: str):
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(path, file_name)) for file_name in file_names]
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

# Declare retrieval function
def get_related_items(current_item_names: str, items_dataset: pd.DataFrame, top_k: int = 5):
    related_items, _ = get_topk_items.tf_idf(current_item_names, top_k=top_k)
    return related_items

# Declare multi-prompts inference function
def run_instructions(model: str, 
                     prompts: list[str], system_message: str = None, temperature: float = 1e-5,
                     test_mode: bool = True):
    messages = []
    print('\n\n========== Start Conversation ==========')
    if system_message:
        print('---------- System Message ----------')
        system_message = t2s.convert(system_message)
        messages.append({"role": "system", "content": system_message})
        print(system_message)

    history = []
    client = OpenAI()
    
    for i in range(len(prompts)):
        print(f'---------- Instruction {i} ----------')
        prompts[i] = t2s.convert(prompts[i])
        messages.append({"role": "user", "content": prompts[i]})
        print(prompts[i])

        print(f'---------- Response {i} ----------')
        if not test_mode:
            completion = client.chat.completions.create(
                model = model,
                temperature=1e-4,
                messages=messages
            )
            response = completion.choices[0].message.content
        else:
            response = "This is a placeholder response."
        messages.append({"role": "assistant", "content": response})
        print(response)
    print('========== End Conversation ==========')
    return messages



def main():
    args = parse_args()
    print(args)

    # Load items dataset
    items_dataset = load_items('random_samples_1M')

    # Load inference data
    with open(args.inference_file, 'rb') as file:
        pc_test_data = pickle.load(file)

    p_names = [item['context'] for item in pc_test_data]
    p_names = list(dict.fromkeys(p_names))

    # Control number of inference items
    if args.num_inference != -1:
        p_names = p_names[:args.num_inference]

    system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"
    prompts_t = [
        '详细了解以下商品名称，尽可能辨认出你认识的所有关键词，并解释。\n{item}',
    ]

    if args.hw_matching:
        input_csv_dirs = './C2C_all'
        output_csv_dirs = './C2C_all_labelled'
        for input_csv in os.listdir(input_csv_dirs):
            input_csv_path = os.path.join(input_csv_dirs, input_csv)
            # 確認'./C2C_all_labelled'中有無相對應的檔案
            if os.path.exists(os.path.join(output_csv_dirs, input_csv.replace('.csv', '_with_results.csv'))):
                print(f"File {input_csv} already processed. Skip.")
                continue

            output_csv_path = os.path.join(output_csv_dirs, input_csv.replace('.csv', '_with_results.csv'))

            # 使用 pandas 读取 CSV 文件
            df = pd.read_csv(input_csv_path)

            # 添加新的列用于存储结果
            df['LLM result'] = ""

            # 创建或加载产品描述 DataFrame
            if not os.path.exists('prod_desc.parquet'):
                prod_desc = pd.DataFrame(columns=['p_name', 'desc'])
            else:
                prod_desc = pd.read_parquet('prod_desc.parquet')

            for index, row in df.iterrows():
                # 获取商品名称
                p1_name = row['商品名稱A']
                p2_name = row['商品名稱B']

                # 如果描述不存在，就创建并保存
                if p1_name not in prod_desc['p_name'].values:
                    corpus = '\n'.join(get_related_items(p1_name, items_dataset, top_k=args.top_k))
                    system_message = system_message_t.format(corpus=corpus)
                    prompts = [prompt.format(item=p1_name) for prompt in prompts_t]

                    messages = run_instructions(
                        model=args.model_name,
                        prompts=prompts,
                        system_message=system_message,
                        temperature=args.temperature,
                        test_mode=False
                    )

                    # 保存新的描述到 DataFrame
                    prod_desc = pd.concat([prod_desc, pd.DataFrame({'p_name': [p1_name], 'desc': [messages[2]["content"]]})], ignore_index=True)
                    prod_desc.to_parquet('prod_desc.parquet')

                # 同样地处理 p2_name
                if p2_name not in prod_desc['p_name'].values:
                    corpus = '\n'.join(get_related_items(p2_name, items_dataset, top_k=args.top_k))
                    system_message = system_message_t.format(corpus=corpus)
                    prompts = [prompt.format(item=p2_name) for prompt in prompts_t]

                    messages = run_instructions(
                        model=args.model_name,
                        prompts=prompts,
                        system_message=system_message,
                        temperature=args.temperature,
                        test_mode=False
                    )

                    prod_desc = pd.concat([prod_desc, pd.DataFrame({'p_name': [p2_name], 'desc': [messages[2]["content"]]})], ignore_index=True)
                    prod_desc.to_parquet('prod_desc.parquet')

                # 获取产品描述
                desc1 = prod_desc.loc[prod_desc['p_name'] == p1_name, 'desc'].values[0]
                desc2 = prod_desc.loc[prod_desc['p_name'] == p2_name, 'desc'].values[0]

                # 打印描述
                print(f"Description for product 1: {desc1}")
                print(f"Description for product 2: {desc2}")

                # 创建匹配请求
                match_prompts = [
                    f"""###Product1
                    {p1_name}
                    ###Description1
                    {desc1}
                    ###Product2
                    {p2_name}
                    ###Description2
                    {desc2}
                    ###Instruction
                    1. 請根據上述描述，判斷 Product1 和 Product2 是否為一模一樣的商品、可進行進一步比價？任何顏色、規格上的差異都不被允許。
                    2. 請直接回答「是」或「否」。不要回傳其他文字，或是標點符號。"""
                ]

                messages = run_instructions(
                    model=args.model_name,
                    prompts=match_prompts,
                    temperature=args.temperature,
                    test_mode=False
                )

                # 保存 LLM 结果
                df.at[index, 'LLM result'] = messages[1]['content'][-1]

                # 如果想要限制行数（测试用途）
                # if index == 1:
                #     break

            # 保存结果到新的 CSV 文件
            df.to_csv(output_csv_path, index=False)
    
if __name__ == "__main__":
    main()