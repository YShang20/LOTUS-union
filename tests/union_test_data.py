import pandas as pd

import lotus
from lotus.models import SentenceTransformersRM, LM
from lotus.sem_ops.sem_union import find_semantic_groups
# Configure models for LOTUS
#lm = LM(model="gpt-4o-mini", max_batch_size = 10)
#lm = LM(model="deepseek/deepseek-chat")
#lm = LM(model="ollama/llama3.2")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

#lotus.settings.configure(lm=lm)

# file_path = 'tests/labeled_data-2.csv'
# df = pd.read_csv(file_path, comment='#')
# df = df.iloc[:, :-1]  # drop label column

# Suppose each row is either left or right. We'll create them artificially here.
# left_cols = [c for c in df.columns if c.startswith('ltable.')]
# right_cols = [c for c in df.columns if c.startswith('rtable.')]

# df_left = df[left_cols].rename(columns=lambda x: x.replace('ltable.', ''))
# df_right = df[right_cols].rename(columns=lambda x: x.replace('rtable.', ''))
# df_left = df.iloc[:150]
# df_right = df.iloc[150:]
# # # 2. Merge them and keep track of the 'source'
# df_left = pd.read_csv('Amazon-GoogleProducts/Amazon.csv', encoding='ISO-8859-1',quotechar='"', escapechar='\\')
# df_right = pd.read_csv('Amazon-GoogleProducts/GoogleProducts.csv', encoding='ISO-8859-1',quotechar='"', escapechar='\\')

df_left = pd.read_csv('/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus-584/tests/msd_only.csv',quotechar='"', escapechar='\\')
df_right = pd.read_csv('/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus-584/tests/mxm_only.csv',  quotechar='"', escapechar='\\')

df_left['source'] = 'left'
df_right['source'] = 'right'

# 3. Perform index-based grouping
df_right = df_right.rename(columns={'name': 'title'})
use_columns = [  'title', 'description', 'manufacturer', 'price']
use_columns = [  'lyrics']
#use_columns = ["author","date","description","length","price","publisher","title"]
final_df = find_semantic_groups(
    table1=df_left,
    table2=df_right,
    columns= use_columns,
    rm = rm,
    k_neighbors=3,
    skip_same_side=False  # or True
)

print("Representatives of each connected semantic group:")
print(final_df)
print(len(final_df))
#
final_df.to_csv("tests/knn3_amazon_L6_l2_0.4.csv", index=False)


ground_truth_df = pd.read_csv("processed_results_230k.csv")  

# Extract record_ids
# ground_truth_ids = set(ground_truth_df['record_id'])
# retrieved_ids = set(final_df['record_id'])

ground_truth_ids = set(ground_truth_df['unified_id'])
retrieved_ids = set(final_df['track_id'])

# Compute how many ground truth records are correctly retrieved
true_positives = ground_truth_ids.intersection(retrieved_ids)
recall = len(true_positives) / len(ground_truth_ids)
precision = len(true_positives) / len(retrieved_ids)

print(f"Recall: {recall:.4f}")
print(f"Correctly retrieved: {len(true_positives)} / {len(ground_truth_ids)}")
print(f"Precision: {precision:.4f}")
print(f"Correctly retrieved: {len(true_positives)} / {len(retrieved_ids)}")




# file_path="labeled_data-2.csv"

# # Reload the CSV file, skipping metadata lines
# df = pd.read_csv(file_path, comment='#')

# # Drop the last column (label)
# df = df.iloc[:, :-1]

# # Separate into left (L) and right (R) tables based on column prefixes
# left_columns = [col for col in df.columns if col.startswith('ltable.')]
# right_columns = [col for col in df.columns if col.startswith('rtable.')]

# df_left = df[left_columns].rename(columns=lambda x: x.replace('ltable.', ''))
# df_right = df[right_columns].rename(columns=lambda x: x.replace('rtable.', ''))
# df_left = df_left[:10]
# df_right = df_right[:10]

# result_df = df_left.sem_union(
#     df_right,
#     join_instruction="{author:left},{date:left},{description:left},{length:left},{price:left},{publisher:left},{title:left},\
#                       {author:right},{date:right},{description:right},{length:right},{price:right},{publisher:right},{title:right}",
#     safe_mode=False
# )

# print(result_df)
# result_df.to_csv("/Users/shangyuntao/Downloads/lotus-main/union_test_results/test1-deekseek40.csv", index=False)
