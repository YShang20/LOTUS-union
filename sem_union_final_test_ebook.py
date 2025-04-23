import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import lotus
from lotus.models import SentenceTransformersRM, LM
from lotus.sem_ops.sem_union import sem_union

# Configure models for LOTUS
lm = LM(model="gpt-4o-mini",
        temperature=1,
        num_retries=3,
        max_batch_size=64)

# Configure retrieval model for embeddings
rm = SentenceTransformersRM(model="all-MiniLM-L6-v2")

lotus.settings.configure(lm=lm)

# Load the labeled data
file_path = "/Users/shangyuntao/Downloads/labeled_data-2.csv"
df = pd.read_csv(file_path, comment='#')

# Drop the last column (label)
df = df.iloc[:, :-1]

# Separate into left (L) and right (R) tables based on column prefixes
left_columns = [col for col in df.columns if col.startswith('ltable.')]
right_columns = [col for col in df.columns if col.startswith('rtable.')]

df_left = df[left_columns].rename(columns=lambda x: x.replace('ltable.', ''))
df_right = df[right_columns].rename(columns=lambda x: x.replace('rtable.', ''))


# join_instruction = "author:left, date:left, description:left, length:left, price:left, publisher:left, title:left, author:right, date:right, description:right, length:right, price:right, publisher:right, title:right"

# result_df = df_left.sem.union(
#     df_right,
#     join_instruction=join_instruction,
#     sim_upper_threshold=0.8,
#     sim_lower_threshold=0.3,
#     embedding_model=rm,
#     safe_mode=False,
#     show_progress_bar=True
# )

# Define user instruction
user_instruction = (
            f"Compare the above two rows with the following column names and determine if they represent the same entity or information.\n\n"
            f"Row 1 columns: {"author", "date", "description", "length", "price", "publisher", "title"}\n"
            f"Row 2 columns: {"author", "date", "description", "length", "price", "publisher", "title"}\n\n"
        )

# Run the test with the layered approach
columns = ["author", "date", "description", "length", "price", "publisher", "title"]
result_df = sem_union(
    table1=df_left,
    table2=df_right,
    columns1=columns,
    columns2=columns,
    user_instruction=user_instruction,
    sim_upper_threshold=0.6, 
    sim_lower_threshold=0.6, 
    embedding_model=rm,  # Pass the embedding model directly
    safe_mode=False,
    show_progress_bar=True,
   # auto_threshold="Oracle"
)

# Save results
result_df.to_csv("/Users/shangyuntao/Downloads/lotus-main/union_test_results/layered_sem_union_results.csv", index=False)

# Evaluation with processed_result_400.csv
print("\nEvaluating accuracy using processed_result_400.csv")

try:
    # Load ground truth data
    ground_truth = pd.read_csv("/Users/shangyuntao/Downloads/lotus-main/processed_result_400.csv")
    
    # Count unique book records in ground truth
    unique_ground_truth = len(ground_truth)
    
    # Count unique book records in our result
    unique_results = len(result_df)
    
    print(f"Ground truth unique records: {unique_ground_truth}")
    print(f"Our method unique records: {unique_results}")
    

    matched_records = 0

    match_columns = ['author', 'date', 'description', 'length', 'price', 'publisher', 'title']
    
    for _, our_row in result_df.iterrows():
        match_found_for_our_row = False
        for _, gt_row in ground_truth.iterrows():
            all_columns_match = True
            for col in match_columns:
                # Get values, handle potential missing columns/NaNs, convert to lowercase string
                our_val = str(our_row[col]).lower() if col in our_row and pd.notna(our_row[col]) else ''
                gt_val = str(gt_row[col]).lower() if col in gt_row and pd.notna(gt_row[col]) else ''
                
                # Check for exact match for this column
                if our_val != gt_val:
                    all_columns_match = False
                    break
            
            if all_columns_match:
                matched_records += 1
                match_found_for_our_row = True 
                break 

    # Calculate metrics
    true_positives = matched_records  # Records we correctly found
    false_positives = unique_results - matched_records  # Records in our results that don't match ground truth
    false_negatives = unique_ground_truth - matched_records  # Records in ground truth we missed
    
    # Calculate precision, recall, and F1
    precision = true_positives / unique_results if unique_results > 0 else 0
    recall = true_positives / unique_ground_truth if unique_ground_truth > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"True positives (matches found in ground truth): {true_positives}")
    print(f"False positives (records not in ground truth): {false_positives}")
    print(f"False negatives (ground truth records missed): {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
        
except Exception as e:
    print(f"Error in evaluation: {e}") 