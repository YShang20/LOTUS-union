import pandas as pd
import numpy as np
import re
from collections import defaultdict

import lotus
from lotus.models import SentenceTransformersRM
from lotus.sem_ops.sem_union import exact_match

# Configure retrieval model for embedding
rm = SentenceTransformersRM(model="sentence-transformers/all-mpnet-base-v2")

# Load the datasets
df_left = pd.read_csv('Amazon-GoogleProducts/Amazon.csv', encoding='ISO-8859-1', quotechar='"', escapechar='\\')
df_right = pd.read_csv('Amazon-GoogleProducts/GoogleProducts.csv', encoding='ISO-8859-1', quotechar='"', escapechar='\\')

# Add source information
df_left['source'] = 'amazon'
df_right['source'] = 'google'

# Rename columns if needed to have consistent naming
df_right = df_right.rename(columns={'name': 'title'})

# Define columns to use for matching
use_columns = ['title', 'description', 'manufacturer', 'price']

# Convert price to numeric format (handling missing values and currency symbols)
def convert_price(price):
    if pd.isna(price):
        return np.nan
    try:
        if isinstance(price, str):
            # Remove currency symbols, commas and convert to float
            price = price.replace('$', '').replace(',', '').strip()
            return float(price) if price else np.nan
        return float(price)
    except (ValueError, TypeError):
        return np.nan

# Apply price conversion to both dataframes
if 'price' in df_left.columns:
    df_left['price'] = df_left['price'].apply(convert_price)
if 'price' in df_right.columns:
    df_right['price'] = df_right['price'].apply(convert_price)

# Load ground truth
ground_truth_df = pd.read_csv("Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv")
print(f"Loaded perfect mapping file with {len(ground_truth_df)} record pairs")

# Print column information
print("\nColumn names in the dataframes:")
print("Amazon columns:", df_left.columns.tolist())
print("Google columns:", df_right.columns.tolist())
print("Ground truth columns:", ground_truth_df.columns.tolist())

# Run exact matching
match_matrix = exact_match(
    table1=df_left,
    table2=df_right,
    columns1=use_columns,
    columns2=use_columns
)

n_left = len(df_left)
n_right = len(df_right)
print(f"Generated match matrix with shape {match_matrix.shape}")
print(f"Amazon products: {n_left}, Google products: {n_right}")

pd.DataFrame(match_matrix).to_csv('exact_match_results.csv', index=False)

# Step 1: Create ID-to-Index mappings for both datasets
amazon_id_to_index = {}
for idx, row in df_left.iterrows():
    amazon_id_to_index[str(row['id']).lower()] = idx

google_id_to_index = {}
google_numeric_id_to_index = {}
for idx, row in df_right.iterrows():
    google_id = str(row['id'])
    google_id_to_index[google_id] = idx
    
    # Extract numeric ID from Google product ID
    match = re.search(r'(\d+)$', google_id)
    if match:
        numeric_id = match.group(1)
        google_numeric_id_to_index[numeric_id] = idx

print(f"Created mappings for {len(amazon_id_to_index)} Amazon IDs and {len(google_numeric_id_to_index)} Google numeric IDs")

# Step 2: Convert match_matrix to product ID pairs
# The match matrix is structured as:
# [  Amazon-Amazon  |  Amazon-Google  ]
# [  Google-Amazon  |  Google-Google  ]
# We need to extract matches from the Amazon-Google quadrant (top-right)

predicted_id_pairs = set()

# Extract matches from the top-right quadrant (Amazon-Google)
for i in range(n_left+n_right):
    for j in range(n_right+n_right):
        j_idx = j + n_left  # Adjust index for right quadrant
        if match_matrix[i, j_idx] == 1:  # If there's a match between an Amazon and a Google product
            amazon_id = str(df_left.iloc[i]['id']).lower()
            google_id = str(df_right.iloc[j]['id'])
            
            # Extract numeric part of Google ID for consistency with ground truth
            google_match = re.search(r'(\d+)$', google_id)
            if google_match:
                google_numeric_id = google_match.group(1)
                predicted_id_pairs.add((amazon_id, google_numeric_id))

print(f"Found {len(predicted_id_pairs)} predicted matching ID pairs")
print(f"Sample pairs (first 5): {list(predicted_id_pairs)[:5]}")

# Step 3: Build a matching index directly from the matrix structure
matches_count = 0
for i in range(n_left):
    for j in range(n_right):
        j_idx = j + n_left
        if match_matrix[i, j_idx] == 1:
            matches_count += 1

print(f"Direct matrix counting found {matches_count} matches between Amazon and Google products")

# Step 4: Convert ground truth to product ID pairs
ground_truth_id_pairs = set()

for _, row in ground_truth_df.iterrows():
    amazon_id = str(row['idAmazon']).lower()
    google_id = row['idGoogleBase']
    
    # Extract numeric ID from Google product URL
    google_match = re.search(r'(\d+)$', google_id)
    if google_match:
        google_numeric_id = google_match.group(1)
        ground_truth_id_pairs.add((amazon_id, google_numeric_id))

print(f"Found {len(ground_truth_id_pairs)} ground truth ID pairs")
print(f"Sample ground truth pairs (first 5): {list(ground_truth_id_pairs)[:5]}")

# Step 5: Calculate evaluation metrics
true_positives = predicted_id_pairs.intersection(ground_truth_id_pairs)
false_positives = predicted_id_pairs - ground_truth_id_pairs
false_negatives = ground_truth_id_pairs - predicted_id_pairs

precision = len(true_positives) / len(predicted_id_pairs) if predicted_id_pairs else 0
recall = len(true_positives) / len(ground_truth_id_pairs) if ground_truth_id_pairs else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\nEvaluation Results (ID-based):")
print(f"True Positives: {len(true_positives)}")
print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Step 6: Detailed analysis of true positives
if len(true_positives) > 0:
    print("\nSample True Positive matches (first 5):")
    for i, (amazon_id, google_id) in enumerate(list(true_positives)[:5]):
        # Find the corresponding rows in the original dataframes
        amazon_idx = next((idx for id_val, idx in amazon_id_to_index.items() if id_val == amazon_id), None)
        google_idx = next((idx for id_val, idx in google_numeric_id_to_index.items() if id_val == google_id), None)
        
        if amazon_idx is not None and google_idx is not None:
            amazon_row = df_left.iloc[amazon_idx]
            google_row = df_right.iloc[google_idx]
            
            print(f"\nMatch {i+1}:")
            print(f"Amazon ID: {amazon_id}")
            print(f"Google ID: {google_id}")
            print(f"Amazon data: {amazon_row[use_columns].to_dict()}")
            print(f"Google data: {google_row[use_columns].to_dict()}")

# Step 7: Check for any missed mappings that could have been caught
# Look at a few false negatives to understand why they weren't matched
if len(false_negatives) > 0:
    print("\nAnalysis of False Negatives (first 5):")
    for i, (amazon_id, google_id) in enumerate(list(false_negatives)[:5]):
        amazon_idx = next((idx for id_val, idx in amazon_id_to_index.items() if id_val == amazon_id), None)
        google_idx = next((idx for id_val, idx in google_numeric_id_to_index.items() if id_val == google_id), None)
        
        if amazon_idx is not None and google_idx is not None:
            amazon_row = df_left.iloc[amazon_idx]
            google_row = df_right.iloc[google_idx]
            
            print(f"\nFalse Negative {i+1}:")
            print(f"Amazon ID: {amazon_id}")
            print(f"Google ID: {google_id}")
            print(f"Amazon data: {amazon_row[use_columns].to_dict()}")
            print(f"Google data: {google_row[use_columns].to_dict()}")
            
            # Check if any of the fields match exactly
            exact_matches = []
            for col in use_columns:
                if not pd.isna(amazon_row[col]) and not pd.isna(google_row[col]):
                    if str(amazon_row[col]).lower() == str(google_row[col]).lower():
                        exact_matches.append(col)
            
            if exact_matches:
                print(f"Exactly matching fields: {exact_matches}")
            else:
                print(f"No exact field matches found between these products") 