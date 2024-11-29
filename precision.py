import os
import pandas as pd

def process_csv(folder_path, output_folder='./error_data'):
    csv_summary = []
    total_o_o, total_x_o, total_o_x, total_x_x = 0, 0, 0, 0
    error_items = []

    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在")
        return

    # 創建錯誤資料的輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"讀取檔案失敗: {file}，錯誤: {e}")
                    continue

                # 檢查必要欄位是否存在
                if not {'label', 'LLM result', '商品名稱A', '商品名稱B'}.issubset(df.columns):
                    print(f"檔案 {file} 缺少必要欄位")
                    continue

                count_label_o_predicted_yes = df[(df['label'] == 'o') & (df['LLM result'] == '是')].shape[0]
                count_label_x_predicted_yes = df[(df['label'] == 'x') & (df['LLM result'] == '是')].shape[0]
                count_label_o_predicted_no = df[(df['label'] == 'o') & (df['LLM result'] != '是')].shape[0]
                count_label_x_predicted_no = df[(df['label'] == 'x') & (df['LLM result'] != '是')].shape[0]

                # 累積統計
                total_o_o += count_label_o_predicted_yes
                total_x_o += count_label_x_predicted_yes
                total_o_x += count_label_o_predicted_no
                total_x_x += count_label_x_predicted_no

                # 過濾錯誤的資料 (o_x 和 x_o)
                o_x_items = df[(df['label'] == 'o') & (df['LLM result'] != '是')]
                x_o_items = df[(df['label'] == 'x') & (df['LLM result'] == '是')]

                # # 記錄錯誤的資料並保存成 CSV
                # if not o_x_items.empty:
                #     error_file_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_o_x_errors.csv")
                #     o_x_items.to_csv(error_file_path, index=False)
                #     error_items.append({'file': file, 'type': 'o_x', 'output_file': error_file_path})

                # if not x_o_items.empty:
                #     error_file_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_x_o_errors.csv")
                #     x_o_items.to_csv(error_file_path, index=False)
                #     error_items.append({'file': file, 'type': 'x_o', 'output_file': error_file_path})

                # 收集檔案統計
                csv_summary.append({
                    'file': file,
                    'o_o': count_label_o_predicted_yes,
                    'x_o': count_label_x_predicted_yes,
                    'o_x': count_label_o_predicted_no,
                    'x_x': count_label_x_predicted_no
                })

        # 總結數據
        total = total_o_o + total_x_o + total_o_x + total_x_x
        accuracy = (total_o_o + total_x_x) / total if total > 0 else 0
        precision = total_o_o / (total_o_o + total_x_o) if (total_o_o + total_x_o) > 0 else 0
        recall = total_o_o / (total_o_o + total_o_x) if (total_o_o + total_o_x) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 輸出結果
        print("總和統計：")
        print(f"label 為 o 且預測為 o: {total_o_o}")
        print(f"label 為 x 且預測為 o: {total_x_o}")
        print(f"label 為 o 且預測為 x: {total_o_x}")
        print(f"label 為 x 且預測為 x: {total_x_x}")

        print("\n模型表現指標：")
        print(f"準確率（Accuracy）：{accuracy*100:.4f} %")
        print(f"精確率（Precision）：{precision*100:.4f} %")
        print(f"召回率（Recall）：{recall*100:.4f} %")
        print(f"F1-score：{f1_score*100:.4f} %")

        print("\n錯誤的資料已保存：")
        for error in error_items:
            print(f"檔案: {error['file']}，類型: {error['type']}，錯誤資料存至: {error['output_file']}")

        print(f"\n總檔案數據：{csv_summary}")
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")

if __name__ == '__main__':
    folder_path = './M11307002/b2c_desc_with_rag_kai'
    process_csv(folder_path)
