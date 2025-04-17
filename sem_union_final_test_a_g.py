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
rm = SentenceTransformersRM(model="all-MiniLM-L12-v2")

lotus.settings.configure(lm=lm)

# # Comment out the old dataset loading part
# # Load the labeled data
# file_path = "/Users/shangyuntao/Downloads/labeled_data-2.csv"
# df = pd.read_csv(file_path, comment='#')
# 
# # Drop the last column (label)
# df = df.iloc[:, :-1]
# 
# # Separate into left (L) and right (R) tables based on column prefixes
# left_columns = [col for col in df.columns if col.startswith('ltable.')]
# right_columns = [col for col in df.columns if col.startswith('rtable.')]
# 
# df_left = df[left_columns].rename(columns=lambda x: x.replace('ltable.', ''))
# df_right = df[right_columns].rename(columns=lambda x: x.replace('rtable.', ''))

# Load the Amazon-GoogleProducts datasets
amazon_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/Amazon.csv"
google_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/GoogleProducts.csv"

df_left = pd.read_csv(amazon_path, encoding='ISO-8859-1',quotechar='"', escapechar='\\')
df_right = pd.read_csv(google_path, encoding='ISO-8859-1',quotechar='"', escapechar='\\')
df_right = df_right.rename(columns={"name": "title"})

print(f"Amazon dataset shape: {df_left.shape}")
print(f"Google dataset shape: {df_right.shape}")

# Check for column consistency
print("Amazon dataset columns:", df_left.columns.tolist())
print("Google dataset columns:", df_right.columns.tolist())

# Define user instruction
user_instruction = (
            f"Compare the above two rows with the following column names and determine if they represent the same entity or information.\n\n"
            f"Row 1 columns: {"title", "description", "manufacturer"}\n"
            f"Row 2 columns: {"title", "description", "manufacturer"}\n\n"
        )

# Run the test with the layered approach
result_df = sem_union(
    table1=df_left,
    table2=df_right,
    columns1=["title", "description", "manufacturer"],
    columns2=["title", "description", "manufacturer"],
    user_instruction=user_instruction,
    sim_upper_threshold=0.9, 
    sim_lower_threshold=0.35, 
    embedding_model=rm,  # Pass the embedding model directly
    safe_mode=False,
    show_progress_bar=True
)

# Save results
result_df.to_csv("/Users/shangyuntao/Downloads/lotus-main/union_test_results/amazon_google_sem_union_results.csv", index=False)

# Evaluation with processed_result_3290.csv
print("\nEvaluating accuracy using processed_result_3290.csv")

try:
    # Load ground truth data
    ground_truth = pd.read_csv("/Users/shangyuntao/Downloads/lotus-main/processed_result_3290.csv")
    
    # Count unique product records in ground truth
    unique_ground_truth = len(ground_truth)
    
    # Count unique product records in our result
    unique_results = len(result_df)
    
    print(f"Ground truth unique records: {unique_ground_truth}")
    print(f"Our method unique records: {unique_results}")
    
    # Try to match records between our results and ground truth
    # Now using ID for matching instead of title
    matched_records = 0
    
    # Create a list of unified_ids from ground truth for faster matching
    gt_unified_ids = ground_truth['id'].astype(str).tolist()
    
    for _, our_row in result_df.iterrows():
        # Get the ID from our result (could be from Amazon or Google dataset)
        our_id = str(our_row['id']) if 'id' in our_row else ''
        
        # Check if our ID is in the ground truth unified IDs
        if our_id in gt_unified_ids:
            matched_records += 1
            continue
        
        # If no direct match found, try to clean and match the IDs
        # For URLs in Google dataset, extract the ID portion
        # if our_id.startswith('http'):
        #     # Extract the last part of the URL
        #     parts = our_id.split('/')
        #     if parts and len(parts) > 0:
        #         clean_id = parts[-1]
        #         if clean_id in gt_unified_ids:
        #             matched_records += 1
        #             continue
    
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


