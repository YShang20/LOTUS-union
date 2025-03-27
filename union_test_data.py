import pandas as pd

import lotus
from lotus.models import SentenceTransformersRM, LM
# Configure models for LOTUS
#lm = LM(model="gpt-4o-mini", max_batch_size = 10)
lm = LM(model="deepseek/deepseek-chat")
#lm = LM(model="ollama/llama3.2")
# rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

lotus.settings.configure(lm=lm)

file_path="labeled_data-2.csv"

# Reload the CSV file, skipping metadata lines
df = pd.read_csv(file_path, comment='#')

# Drop the last column (label)
df = df.iloc[:, :-1]

# Separate into left (L) and right (R) tables based on column prefixes
left_columns = [col for col in df.columns if col.startswith('ltable.')]
right_columns = [col for col in df.columns if col.startswith('rtable.')]

df_left = df[left_columns].rename(columns=lambda x: x.replace('ltable.', ''))
df_right = df[right_columns].rename(columns=lambda x: x.replace('rtable.', ''))
df_left = df_left[:10]
df_right = df_right[:10]

result_df = df_left.sem_union(
    df_right,
    join_instruction="{author:left},{date:left},{description:left},{length:left},{price:left},{publisher:left},{title:left},\
                      {author:right},{date:right},{description:right},{length:right},{price:right},{publisher:right},{title:right}",
    safe_mode=False
)

print(result_df)
result_df.to_csv("/Users/shangyuntao/Downloads/lotus-main/union_test_results/test1-deekseek40.csv", index=False)
