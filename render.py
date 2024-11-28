import csv
import pandas as pd

csv_file = 'combined_results_with_results.csv'
df = pd.read_csv(csv_file)
df.sort_values(by=['LLM result'], inplace=True, ascending=False)
df.to_html('data.html',encoding='utf-8-sig', index=False, escape=False)