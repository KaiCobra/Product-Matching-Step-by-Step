import os
import pandas as pd

if __name__ == '__main__':
    folder_path = './M11307002/b2c_desc_kai'
    csv_summary = []

    LLM_total_o = 0
    LLM_total_x = 0
    LLM_total_n = 0
    LLM_total_c = 0
    LLM_total_a = 0
    LLM_total_s = 0


    Human_total_o= 0
    Human_total_x= 0 
    Human_total_n= 0
    Human_total_c= 0 
    Human_total_a= 0 
    Human_total_s= 0 



    # 確保資料夾存在
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在")
    else:
        # 遍歷資料夾中的所有 CSV 檔案
        for file_name in os.listdir(folder_path):
            print(f"處理檔案: {file_name}")
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)

                # 讀取 CSV 檔案
                df = pd.read_csv(file_path)
                
                # 計算 `o` 和 `x` 的數量
                LLM_count_o = df['LLM result'].value_counts().get('是', 0)
                LLM_count_x = len(df) - LLM_count_o
                
                Human_count_o = df['label'].value_counts().get('o', 0)
                Human_count_x = df['label'].value_counts().get('x', 0)
                

                # 累加到總和
                LLM_total_o += LLM_count_o
                LLM_total_x += LLM_count_x
                # LLM_total_n += LLM_count_n
                # LLM_total_c += LLM_count_c
                # LLM_total_a += LLM_count_a
                # LLM_total_s += LLM_count_s

                Human_total_o += Human_count_o
                Human_total_x += Human_count_x
                # Human_total_n += Human_count_n
                # Human_total_c += Human_count_c
                # Human_total_a += Human_count_a
                # Human_total_s += Human_count_s

        # 計算所有檔案的總和比例
        LLM_total = LLM_total_o + LLM_total_x 
        Human_total = Human_total_o + Human_total_x

        # 印出總和結果
        print("\n=== 總和結果LLM ===")
        print(f"總 o: {LLM_total_o}")
        print(f"總 x: {LLM_total_x}")
        # print(f"總 數: {LLM_total_n}")
        # print(f"總 色: {LLM_total_c}")
        # print(f"總 容: {LLM_total_a}")
        # print(f"總 寸: {LLM_total_s}")
        total_ratio = LLM_total_o/(LLM_total)*100
        print(f"總 o/(o+x) 比例: {total_ratio:.2f}%")

        
        print("\n=== 總和結果Human ===")
        print(f"總 o: {Human_total_o}")
        print(f"總 x: {Human_total_x}")
        # print(f"總 數: {Human_total_n}")
        # print(f"總 色: {Human_total_c}")
        # print(f"總 容: {Human_total_a}")
        # print(f"總 寸: {Human_total_s}")
        total_ratio = Human_total_o/(Human_total)*100
        print(f"總 o/(o+x) 比例: {total_ratio:.2f}%")

