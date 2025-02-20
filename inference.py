# Import libraries
import torch
import os
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import get_topk_items
from opencc import OpenCC
import pickle
import random
from gradio_client import Client

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
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Name of the model")
    parser.add_argument('--top_k', type=int, default=10, help="Top K related items to retrieve")
    parser.add_argument('--inference_file', type=str, default='./data/test.pickle', help="Input file for inference")
    parser.add_argument('--dtype', type=str, default='int8', choices=['int8', 'int4'], help="Data type for model precision (int8 or int4)")
    parser.add_argument('--num_inference', type=int, default=-1, help="Number of items to infer, -1 means all")
    parser.add_argument('--temerature', type=float, default=1e-5, help="Temperature for generation")
    parser.add_argument('--use_qwen_api', type=bool, default=False, help="Whether to use Qwen API for inference")
    parser.add_argument('-i', '--interactive', action='store_true', help="Run in interactive mode")
    parser.add_argument('-m', '--match', action='store_true', help="Run in interactive match mode")

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
def run_instructions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                     prompts: list[str], system_message: str = None, temperature: float = 1e-5,
                     test_mode: bool = True, use_qwen_api: bool = False):
    messages = []
    print('\n\n========== Start Conversation ==========')
    if system_message:
        print('---------- System Message ----------')
        system_message = t2s.convert(system_message)
        messages.append({"role": "system", "content": system_message})
        print(system_message)
    
    if use_qwen_api:
        history = []
        client = Client("Qwen/Qwen2.5")

        for i in range(len(prompts)):
            prompts[i] = t2s.convert(prompts[i])
            print(prompts[i])

            if not test_mode:
                response = client.predict(
                    query=prompts[i],
                    history=history,
                    system=system_message,
                    radio='72B',
                    api_name="/model_chat_1"
                )
                history = response[1]
            else:
                response = "This is a placeholder response."
            
            print(response[1])
        print('========== End Conversation ==========')
        return response[1]
    
    for i in range(len(prompts)):
        print(f'---------- Instruction {i} ----------')
        prompts[i] = t2s.convert(prompts[i])
        messages.append({"role": "user", "content": prompts[i]})
        print(prompts[i])

        print(f'---------- Response {i} ----------')
        if not test_mode:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=temperature)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            response = "This is a placeholder response."
        messages.append({"role": "assistant", "content": response})
        print(response)
    print('========== End Conversation ==========')
    return messages

# Main function for inference
def main():
    args = parse_args()
    print(args)

    # If use_qwen_api is set to True, skip model and tokenizer loading
    if not args.use_qwen_api:
        # Conditionally define BitsAndBytesConfig based on dtype
        if args.dtype == 'int8':
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,          
                llm_int8_threshold=6.0,     
                llm_int8_enable_fp32_cpu_offload=True  
            )
        elif args.dtype == 'int4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  
                bnb_4bit_use_double_quant=True,  
                bnb_4bit_quant_type="nf4",  
                bnb_4bit_compute_dtype=torch.float16 
            )

        # Load the model with the BitsAndBytes configuration
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,  
            device_map="auto",               
            torch_dtype=torch.float16        
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        model, tokenizer = None, None  # Set these to None if using API

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

    if args.interactive:
        while True:
            p_name = input("Enter product name: ")

            # create or load prod_desc.parquet
            if not os.path.exists('prod_desc.parquet'):
                prod_desc = pd.DataFrame(columns=['p_name', 'desc'])
            else:
                prod_desc = pd.read_parquet('prod_desc.parquet')

            if p_name in prod_desc['p_name'].values:
                print(f"Description for the entered product already exists! Here is the saved description:")
                print(prod_desc[prod_desc['p_name'] == p_name]['desc'].values[0])
                continue

            corpus = '\n'.join(get_related_items(p_name, items_dataset, top_k=args.top_k))
            system_message = system_message_t.format(corpus=corpus)

            prompts = [prompt.format(item=p_name) for prompt in prompts_t]
            messages = run_instructions(
                model, tokenizer, prompts, system_message, args.temerature,
                test_mode=False, use_qwen_api=args.use_qwen_api
            )

            # save the desc with the p_name to prod_desc.parquet by pandas
            prod_desc.loc[prod_desc.shape[0]] = [p_name, messages[2]["content"]]
            prod_desc.to_parquet('prod_desc.parquet')

    elif args.match:
        while True:
            p1_name = input("Enter product 1 name: ")
            p2_name = input("Enter product 2 name: ")

            # create or load prod_desc.parquet
            if not os.path.exists('prod_desc.parquet'):
                prod_desc = pd.DataFrame(columns=['p_name', 'desc'])
            else:
                prod_desc = pd.read_parquet('prod_desc.parquet')

            if p1_name not in prod_desc['p_name'].values:
                print(f"Description for product 1 does not exist! Creating description for product 1...")

                corpus = '\n'.join(get_related_items(p1_name, items_dataset, top_k=args.top_k))
                system_message = system_message_t.format(corpus=corpus)

                prompts = [prompt.format(item=p1_name) for prompt in prompts_t]
                messages = run_instructions(
                    model, tokenizer, prompts, system_message, args.temerature,
                    test_mode=False, use_qwen_api=args.use_qwen_api
                )

                # save the desc with the p_name to prod_desc.parquet by pandas
                prod_desc.loc[prod_desc.shape[0]] = [p1_name, messages[2]["content"]]
                prod_desc.to_parquet('prod_desc.parquet')

            # create or load prod_desc.parquet
            if not os.path.exists('prod_desc.parquet'):
                prod_desc = pd.DataFrame(columns=['p_name', 'desc'])
            else:
                prod_desc = pd.read_parquet('prod_desc.parquet')

            if p2_name not in prod_desc['p_name'].values:
                print(f"Description for product 2 does not exist! Creating description for product 2...")

                corpus = '\n'.join(get_related_items(p2_name, items_dataset, top_k=args.top_k))
                system_message = system_message_t.format(corpus=corpus)

                prompts = [prompt.format(item=p2_name) for prompt in prompts_t]
                messages = run_instructions(
                    model, tokenizer, prompts, system_message, args.temerature,
                    test_mode=False, use_qwen_api=args.use_qwen_api
                )

                # save the desc with the p_name to prod_desc.parquet by pandas
                prod_desc.loc[prod_desc.shape[0]] = [p2_name, messages[2]["content"]]
                prod_desc.to_parquet('prod_desc.parquet')

            print(f"Description for product 1:")
            desc1 = prod_desc[prod_desc['p_name'] == p1_name]['desc'].values[0]
            print(desc1)

            print(f"Description for product 2:")
            desc2 = prod_desc[prod_desc['p_name'] == p2_name]['desc'].values[0]
            print(desc2)

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
2. 請直接回答「是」或「否」。不要回傳其他文字。""",
            ]

            messages = run_instructions(
                model, tokenizer, match_prompts, None, args.temerature,
                test_mode=False, use_qwen_api=args.use_qwen_api
            )

            print(f"Match result: {messages[2]['content']}")

    else:
        with tqdm(total=len(p_names)) as pbar:
            for i, p_name in enumerate(p_names):
                # create or load prod_desc.parquet
                if not os.path.exists('prod_desc.parquet'):
                    prod_desc = pd.DataFrame(columns=['p_name', 'desc'])
                else:
                    prod_desc = pd.read_parquet('prod_desc.parquet')

                if p_name in prod_desc['p_name'].values:
                    print(f"Description for the product already exists! Here is the saved description:")
                    print(prod_desc[prod_desc['p_name'] == p_name]['desc'].values[0])
                    continue

                corpus = '\n'.join(get_related_items(p_name, items_dataset, top_k=args.top_k))
                system_message = system_message_t.format(corpus=corpus)

                prompts = [prompt.format(item=p_name) for prompt in prompts_t]
                messages = run_instructions(
                    model, tokenizer, prompts, system_message, args.temerature,
                    test_mode=False, use_qwen_api=args.use_qwen_api
                )

                # save the desc with the p_name to prod_desc.parquet by pandas
                prod_desc.loc[prod_desc.shape[0]] = [p_name, messages[2]["content"]]
                prod_desc.to_parquet('prod_desc.parquet')

                pbar.update(1)

if __name__ == "__main__":
    main()