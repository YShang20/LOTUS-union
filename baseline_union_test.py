import pandas as pd

import lotus
from lotus.models import SentenceTransformersRM, LM, LiteLLMRM
from lotus.vector_store import FaissVS
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('fork')
    vs = FaissVS()
    # Configure models for LOTUS
    # lm = LM(model="gpt-4o-mini")
    lm = LM(model="deepseek/deepseek-chat")
    #lm = LM(model="ollama/llama3.2")
    rm = LiteLLMRM(model="text-embedding-3-small")

    lotus.settings.configure(lm=lm, rm=rm, vs=vs)

    file_path="abeled_data-2.csv"

    # Reload the CSV file, skipping metadata lines
    df = pd.read_csv(file_path, comment='#')

    # Drop the last column (label)
    df = df.iloc[:, :-1]

    # Separate into left (L) and right (R) tables based on column prefixes
    left_columns = [col for col in df.columns if col.startswith('ltable.')]
    right_columns = [col for col in df.columns if col.startswith('rtable.')]

    df_left = df[left_columns].rename(columns=lambda x: x.replace('ltable.', ''))
    df_right = df[right_columns].rename(columns=lambda x: x.replace('rtable.', ''))
    df_left = df_left[:4]
    df_right = df_right[:4]

    # Concatenate DataFrames vertically
    df_stacked = pd.concat([df_left, df_right], ignore_index=True)

    # Merge all columns into a single column with the specified format
    df_result = df_stacked.apply(lambda row: " ".join([str(row[col]) for col in df_stacked.columns if pd.notna(row[col])]), axis=1)

    df = {"union_test": df_result.tolist()}
    df = pd.DataFrame(df)

    df.to_csv("/Users/shangyuntao/Downloads/lotus-main/union_test_results/baseline-deekseek40.csv", index=False)

    result_df= df.sem_index("union_test", "index_dir").sem_dedup("union_test", threshold=0.815)

    print(result_df)
    result_df.to_csv("/Users/shangyuntao/Downloads/lotus-main/union_test_results/baseline-deekseek40.csv", index=False)
