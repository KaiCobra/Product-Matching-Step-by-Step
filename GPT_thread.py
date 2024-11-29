import os
import csv
import pandas as pd
import argparse
import threading
from openai import OpenAI
from tqdm import tqdm
import warnings
import pickle
from opencc import OpenCC
from utils import get_topk_items  # 假設你有這個工具模組
import cfg

# 忽略警告
warnings.filterwarnings("ignore")

# 轉換簡體/繁體中文
t2s = OpenCC('t2s')
s2t = OpenCC('s2t')

# 定義 argparse 來處理參數
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with various configurations")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help="Name of the model")
    parser.add_argument('--top_k', type=int, default=10, help="Top K related items to retrieve")
    parser.add_argument('--inference_file', type=str, default='./data/test.pickle', help="Input file for inference")
    parser.add_argument('--num_inference', type=int, default=-1, help="Number of items to infer, -1 means all")
    parser.add_argument('--temperature', type=float, default=5e-6, help="Temperature for generation")
    parser.add_argument('-hw', '--hw_matching', action='store_true', help="Run in HW mode")

    return parser.parse_args()

# 載入資料集
def load_items(path: str):
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(path, file_name)) for file_name in file_names]
    return pd.concat(df_list, ignore_index=True)

# 定義查詢相關商品的函式
def get_related_items(current_item_names: str, items_dataset: pd.DataFrame, top_k: int = 5):
    related_items, _ = get_topk_items.tf_idf(current_item_names, top_k=top_k)
    return related_items

# 定義多提示的推論函式
def run_instructions(model: str, prompts: list[str], system_message: str = None, temperature: float = 1e-5, test_mode: bool = True):
    messages = []
    if system_message:
        system_message = t2s.convert(system_message)
        messages.append({"role": "system", "content": system_message})

    history = []
    client = OpenAI()  # 使用 openai.client 可能需要設定 API 金鑰

    for i in range(len(prompts)):
        prompts[i] = t2s.convert(prompts[i])
        messages.append({"role": "user", "content": prompts[i]})

        if not test_mode:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            response = completion.choices[0].message.content
        else:
            response = "This is a placeholder response."
        
        messages.append({"role": "assistant", "content": response})

    return messages

# 定義鎖


# 定義每行處理的函式
def process_row(index, row, prod_desc, items_dataset, args, system_message_t, prompts_t, df, pbar):
    try:
        p1_name = row['商品名稱A']
        p2_name = row['商品名稱B']
        
    
        if p1_name not in prod_desc['p_name'].values:
            corpus = '\n'.join(get_related_items(p1_name, items_dataset, top_k=args.top_k))
            system_message = system_message_t.format(corpus=corpus)
            prompts = [prompt.format(item=p1_name) for prompt in prompts_t]

            messages = run_instructions(
                model=args.model_name,
                prompts=prompts,
                system_message=system_message,
                temperature=cfg.temperature,
                test_mode=False
            )

            prod_desc = pd.concat([prod_desc, pd.DataFrame({'p_name': [p1_name], 'desc': [messages[2]["content"]]})], ignore_index=True)
            prod_desc.to_parquet('prod_desc.parquet')

        if p2_name not in prod_desc['p_name'].values:
            corpus = '\n'.join(get_related_items(p2_name, items_dataset, top_k=args.top_k))
            system_message = system_message_t.format(corpus=corpus)
            prompts = [prompt.format(item=p2_name) for prompt in prompts_t]

            messages = run_instructions(
                model=args.model_name,
                prompts=prompts,
                system_message=system_message,
                temperature=cfg.temperature,
                test_mode=False
            )

            prod_desc = pd.concat([prod_desc, pd.DataFrame({'p_name': [p2_name], 'desc': [messages[2]["content"]]})], ignore_index=True)
            prod_desc.to_parquet('prod_desc.parquet')

        desc1 = prod_desc.loc[prod_desc['p_name'] == p1_name, 'desc'].values[0]
        desc2 = prod_desc.loc[prod_desc['p_name'] == p2_name, 'desc'].values[0]

        match_prompts = [
f"""
### Instruction
{cfg.testing_prompt}
###Product1
{p1_name}

###Product1 Description
{desc1}

###Product2
{p2_name}

###Product2 Description
{desc2}

"""
        ]

        messages = run_instructions(
            model=args.model_name,
            prompts=match_prompts,
            temperature=cfg.temperature,
            test_mode=False
        )

        df.at[index, 'LLM result'] = messages[1]['content'][:-5]
        df.at[index, 'Desc A'] = desc1
        df.at[index, 'Desc B'] = desc2
        df.at[index, 'decide'] = messages[1]['content']

    except Exception as e:
        print(f"Error processing row {index}: {e}")

    pbar.update(1)

# 主程式
def main():
    """""
        usage: GPT_thread.py 
        rememeber to change the input_csv_dirs and output_csv_dirs to your own path
    """

    max_threads = os.cpu_count() * 2
    args = parse_args()
    system_message_t = cfg.system_message_t
    prompts_t = cfg.prompts_t
    
    items_dataset = load_items('random_samples_1M')
    input_csv_dirs = './M11307002/b2c_output_kai'
    output_csv_dirs = './M11307002/b2c_desc_with_rag_kai_2'

    for input_csv in os.listdir(input_csv_dirs):
        input_csv_path = os.path.join(input_csv_dirs, input_csv)
        print(f"Processing file {input_csv}")
        if os.path.exists(os.path.join(output_csv_dirs, input_csv.replace('.csv', '_with_results.csv'))):
            print(f"File {input_csv} already processed. Skipping.")
            continue

        output_csv_path = os.path.join(output_csv_dirs, input_csv.replace('.csv', '_with_results.csv'))
        df = pd.read_csv(input_csv_path)

        if 'LLM result' not in df.columns:
            df['LLM result'] = ""

        if 'Desc.A' not in df.columns:
            df['Desc A'] = ""

        if 'Desc.B' not in df.columns:
            df['Desc B'] = ""

        if 'decide' not in df.columns:
            df['decide'] = ""

        if not os.path.exists('prod_desc.parquet'):
            prod_desc = pd.DataFrame(columns=['p_name', 'desc'])
        else:
            prod_desc = pd.read_parquet('prod_desc.parquet')
        
        threads = []
        with tqdm(total=len(df), desc="Processing Rows") as pbar:
            for index, row in df.iterrows():
                thread = threading.Thread(target=process_row, args=(index, row, prod_desc, items_dataset, args, system_message_t, prompts_t, df, pbar))
                threads.append(thread)
                thread.start()

                if len(threads) >= max_threads:
                    for t in threads:
                        t.join()
                    threads = []

            for t in threads:
                t.join()

        df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()