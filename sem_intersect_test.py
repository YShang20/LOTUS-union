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

# Load the Amazon-GoogleProducts datasets
amazon_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/Amazon.csv"
google_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/GoogleProducts.csv"
mapping_path = "/Users/shangyuntao/Downloads/lotus-main/Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv"

df_left = pd.read_csv(amazon_path, encoding='ISO-8859-1', quotechar='"', escapechar='\\')
df_right = pd.read_csv(google_path, encoding='ISO-8859-1', quotechar='"', escapechar='\\')
df_right = df_right.rename(columns={"name": "title"})

# Load the ground truth mapping
gt_mapping = pd.read_csv(mapping_path)

print(f"Amazon dataset shape: {df_left.shape}")
print(f"Google dataset shape: {df_right.shape}")
print(f"Ground truth mapping shape: {gt_mapping.shape}")

# Check for column consistency
print("Amazon dataset columns:", df_left.columns.tolist())
print("Google dataset columns:", df_right.columns.tolist())
print("Ground truth mapping columns:", gt_mapping.columns.tolist())

# Create ground truth data - Amazon records that have matches in Google
amazon_ids_with_matches = gt_mapping['idAmazon'].unique()
ground_truth_amazon = df_left[df_left['id'].isin(amazon_ids_with_matches)]

print(f"Ground truth: {len(ground_truth_amazon)} Amazon records have matches in Google Products")

# Define user instruction
user_instruction = (
    f"Compare the above two representing products with the following column names and determine if they represent the same product.\n\n"
    f"Row 1 columns: {'title', 'description', 'manufacturer'}\n"
    f"Row 2 columns: {'title', 'description', 'manufacturer'}\n\n"
)

# Run the test with the layered approach
result_df = sem_intersect(
    table1=df_left,
    table2=df_right,
    columns1=["title", "description", "manufacturer"],
    columns2=["title", "description", "manufacturer"],
    user_instruction=user_instruction,
    sim_upper_threshold=0.6, 
    sim_lower_threshold=0.4, 
    embedding_model=rm,  # Pass the embedding model directly
    safe_mode=False,
    show_progress_bar=True
)

# Save results
result_df.to_csv("/Users/shangyuntao/Downloads/lotus-main/intersect_test_results/amazon_google_sem_intersect_results.csv", index=False)

# Evaluation
print("\nEvaluating accuracy using perfect mapping ground truth")

try:
    # Check how many of our results are in the ground truth
    our_result_ids = set(result_df['id'].astype(str).tolist())
    ground_truth_ids = set(ground_truth_amazon['id'].astype(str).tolist())
    
    # Count metrics
    true_positives = len(our_result_ids.intersection(ground_truth_ids))
    false_positives = len(our_result_ids - ground_truth_ids)
    false_negatives = len(ground_truth_ids - our_result_ids)
    
    # Calculate precision, recall, and F1
    precision = true_positives / len(our_result_ids) if our_result_ids else 0
    recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Ground truth Amazon records with matches: {len(ground_truth_ids)}")
    print(f"Our method found Amazon records: {len(our_result_ids)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
except Exception as e:
    print(f"Error in evaluation: {e}") 