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

# Load the DBLP-ACM datasets
dblp_path = "Amazon-GoogleProducts/DBLP2.csv"
acm_path = "Amazon-GoogleProducts/ACM.csv"

df_dblp = pd.read_csv(dblp_path, encoding='ISO-8859-1', quotechar='"', escapechar='\\')
df_acm = pd.read_csv(acm_path, encoding='ISO-8859-1', quotechar='"', escapechar='\\')

print(f"DBLP dataset shape: {df_dblp.shape}")
print(f"ACM dataset shape: {df_acm.shape}")

# Check for column consistency
print("DBLP dataset columns:", df_dblp.columns.tolist())
print("ACM dataset columns:", df_acm.columns.tolist())

# Define user instruction for academic papers
user_instruction = (
            f"Compare the above two academic papers with the following column names and determine if they represent the same paper.\n\n"
            f"Row 1 columns: {"title", "authors", "venue", "year"}\n"
            f"Row 2 columns: {"title", "authors", "venue", "year"}\n\n"
        )

# Run the test with the layered approach
result_df = sem_union(
    table1=df_dblp,
    table2=df_acm,
    columns1=["title", "authors", "venue", "year"],
    columns2=["title", "authors", "venue", "year"],
    user_instruction=user_instruction,
    embedding_model=rm,  # Pass the embedding model directly
    safe_mode=False,
    show_progress_bar=True,
    k_neighbors= len(df_dblp) + len(df_acm),
    auto_threshold="Valley"
    )


# Save results
result_df.to_csv("union_test_results/dblp_acm_sem_union_results.csv", index=False)

# Evaluation with processed ground truth
print("\nEvaluating accuracy using processed_dblp_acm_ground_truth.csv")

try:
    # Load ground truth data
    ground_truth_path = "Amazon-GoogleProducts/processed_dblp_acm_ground_truth.csv"
    ground_truth = pd.read_csv(ground_truth_path)
    
    # Count unique paper records in ground truth
    unique_ground_truth = len(ground_truth)
    
    # Count unique paper records in our result
    unique_results = len(result_df)
    
    print(f"Ground truth unique records: {unique_ground_truth}")
    print(f"Our method unique records: {unique_results}")
    
    # Try to match records between our results and ground truth
    matched_records = 0
    
    # Create a list of unified_ids from ground truth for faster matching
    gt_unified_ids = ground_truth['unified_id'].astype(str).tolist()
    
    for _, our_row in result_df.iterrows():
        # Get the ID from our result
        our_id = str(our_row['id']) if 'id' in our_row else ''
        
        # Check if our ID is in the ground truth unified IDs
        if our_id in gt_unified_ids:
            matched_records += 1
            continue
    
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