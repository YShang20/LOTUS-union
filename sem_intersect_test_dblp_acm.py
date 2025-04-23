import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import lotus
from lotus.models import SentenceTransformersRM, LM
from lotus.sem_ops.sem_intersect import sem_intersect

# Configure models for LOTUS
lm = LM(model="gpt-4o-mini",
        temperature=1,
        num_retries=3,
        max_batch_size=64)

# Configure retrieval model for embeddings
rm = SentenceTransformersRM(model="all-MiniLM-L12-v2")

lotus.settings.configure(lm=lm)

# Load the DBLP-ACM datasets
dblp_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/DBLP2.csv"
acm_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/ACM.csv"
mapping_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/DBLP-ACM_perfectMapping.csv"

df_dblp = pd.read_csv(dblp_path, encoding='ISO-8859-1', quotechar='"', escapechar='\\')
df_acm = pd.read_csv(acm_path, encoding='ISO-8859-1', quotechar='"', escapechar='\\')

# Load the ground truth mapping
gt_mapping = pd.read_csv(mapping_path)

print(f"DBLP dataset shape: {df_dblp.shape}")
print(f"ACM dataset shape: {df_acm.shape}")
print(f"Ground truth mapping shape: {gt_mapping.shape}")

# Check for column consistency
print("DBLP dataset columns:", df_dblp.columns.tolist())
print("ACM dataset columns:", df_acm.columns.tolist())
print("Ground truth mapping columns:", gt_mapping.columns.tolist())

# Create ground truth data - DBLP records that have matches in ACM
dblp_ids_with_matches = gt_mapping['idDBLP'].unique()
ground_truth_dblp = df_dblp[df_dblp['id'].isin(dblp_ids_with_matches)]

print(f"Ground truth: {len(ground_truth_dblp)} DBLP records have matches in ACM")

# Define user instruction for academic papers
user_instruction = (
    f"Compare the above two academic papers with the following column names and determine if they represent the same paper.\n\n"
    f"Row 1 columns: {'title', 'authors', 'venue', 'year'}\n"
    f"Row 2 columns: {'title', 'authors', 'venue', 'year'}\n\n"
)

result_df = sem_intersect(
    table1=df_dblp,
    table2=df_acm,
    columns1=["title", "authors", "venue", "year"],
    columns2=["title", "authors", "venue", "year"],
    user_instruction=user_instruction,
    sim_upper_threshold=0.6, 
    sim_lower_threshold=0.4, 
    embedding_model=rm,  # Pass the embedding model directly
    safe_mode=False,
    show_progress_bar=True
)

# Save results
result_df.to_csv("/Users/shangyuntao/Downloads/lotus-main/intersect_test_results/dblp_acm_sem_intersect_results.csv", index=False)

# Evaluation
print("\nEvaluating accuracy using perfect mapping ground truth")

try:
    # Check how many of our results are in the ground truth
    our_result_ids = set(result_df['id'].astype(str).tolist())
    ground_truth_ids = set(ground_truth_dblp['id'].astype(str).tolist())
    
    # Count metrics
    true_positives = len(our_result_ids.intersection(ground_truth_ids))
    false_positives = len(our_result_ids - ground_truth_ids)
    false_negatives = len(ground_truth_ids - our_result_ids)
    
    # Calculate precision, recall, and F1
    precision = true_positives / len(our_result_ids) if our_result_ids else 0
    recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Ground truth DBLP records with matches: {len(ground_truth_ids)}")
    print(f"Our method found DBLP records: {len(our_result_ids)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
            
except Exception as e:
    print(f"Error in evaluation: {e}") 