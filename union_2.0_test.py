import pandas as pd
import numpy as np
from collections import defaultdict

import lotus
from lotus.models import SentenceTransformersRM, LM
from lotus.sem_ops.sem_union import find_semantic_groups, build_hnsw_index, exact_match
# Configure models for LOTUS
#lm = LM(model="gpt-4o-mini", max_batch_size = 10)
#lm = LM(model="deepseek/deepseek-chat")
#lm = LM(model="ollama/llama3.2")
rm = SentenceTransformersRM(model="sentence-transformers/all-mpnet-base-v2")
#rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
#rm = SentenceTransformersRM(model="all-minilm-l6-v2")

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
# # 2. Merge them and keep track of the 'source'
df_left = pd.read_csv('Amazon-GoogleProducts/Amazon.csv', encoding='ISO-8859-1',quotechar='"', escapechar='\\')
df_right = pd.read_csv('Amazon-GoogleProducts/GoogleProducts.csv', encoding='ISO-8859-1',quotechar='"', escapechar='\\')

# df_left = pd.read_csv('/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus-584/tests/msd_only.csv',quotechar='"', escapechar='\\')
# df_right = pd.read_csv('/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus-584/tests/mxm_only.csv',  quotechar='"', escapechar='\\')

df_left['source'] = 'left'
df_right['source'] = 'right'

# Handle price column - convert to numeric with proper error handling
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

# 3. Perform index-based grouping "id","title","description","manufacturer","price"
df_right = df_right.rename(columns={'name': 'title'})
use_columns = ['title', 'description', 'manufacturer', 'price']
#use_columns = [  'lyrics']
#use_columns = ["author","date","description","length","price","publisher","title"]

# Create a function to build a similarity matrix from embeddings
def build_similarity_matrix(table1, table2, columns, rm, full_matrix=False):
    """
    Build a similarity matrix where each entry (i,j) is the similarity score between row i and row j.
    
    Parameters:
    - table1, table2: DataFrames to compare
    - columns: Columns to use for comparison
    - rm: Retrieval model for embedding
    - full_matrix: If True, compute the full n×n matrix; if False, only compute nearest neighbors 
                  (more memory efficient)
    
    Returns:
    - similarity_matrix: n×n numpy array of similarity scores
    - df_stacked: The combined DataFrame
    """
    # Step 1: Combine tables and create text representation
    df_stacked = pd.concat([table1, table2], ignore_index=True)
    n_rows = len(df_stacked)
    
    # Create text representation for each row
    df_result = df_stacked.apply(
        lambda row: " ".join([str(row[col]) for col in columns if pd.notna(row[col])]), 
        axis=1
    ).tolist()
    
    # Step 2: Embed the texts
    embeddings = rm._embed(df_result)
    
    # Step 3: Build HNSW index for efficient similarity computation
    index = build_hnsw_index(embeddings, space='l2', ef_construction=200, M=16)
    
    # Step 4: Create the similarity matrix
    if full_matrix:
        # Compute full n×n matrix (may be memory-intensive for large datasets)
        similarity_matrix = np.zeros((n_rows, n_rows))
        
        # Query each point against all others (we use a large k value)
        labels, distances = index.knn_query(embeddings, k=n_rows)
        
        # Fill the similarity matrix
        for i in range(n_rows):
            for j_idx, j in enumerate(labels[i]):
                # Use 1 - distance as similarity score (since lower distance means higher similarity)
                # For L2 distance, we might want to apply e^(-distance) for better scaling
                if i == j:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = 1.0 - min(1.0, distances[i][j_idx])
    else:
        # Use a sparse approach - only compute for k nearest neighbors
        k = min(50, n_rows-1)  # Set a reasonable k value
        similarity_matrix = np.zeros((n_rows, n_rows))
        
        # Query the index for top-k nearest neighbors for each point
        labels, distances = index.knn_query(embeddings, k=k)
        
        # Fill the similarity matrix with known values
        for i in range(n_rows):
            for j_idx, j in enumerate(labels[i]):
                # Convert distance to similarity score (1 - distance) and clip to [0,1]
                similarity_matrix[i, j] = 1.0 - min(1.0, distances[i][j_idx])
    
    return similarity_matrix, df_stacked

# Run the similarity matrix computation
# Use full_matrix=False for larger datasets to save memory
# similarity_matrix, df_stacked = build_similarity_matrix(
#     table1=df_left,
#     table2=df_right,
#     columns=use_columns,
#     rm=rm,
#     full_matrix=True  # Set to True for full matrix, False for sparse approximation
# )

# # Print some statistics about the similarity matrix
# print(f"Similarity matrix shape: {similarity_matrix.shape}")
# print(f"Median similarity: {np.median(similarity_matrix)}")
# print(f"Max similarity: {np.max(similarity_matrix)}")
# print(f"Min similarity: {np.min(similarity_matrix)}")

# # Get distribution of values in bins of 0.1
# bins = np.arange(0, 1.1, 0.1)  # Creates bins [0.0, 0.1, 0.2, ..., 1.0]
# hist, bin_edges = np.histogram(similarity_matrix.flatten(), bins=bins)

# # Print the distribution
# print("\nDistribution of similarity scores:")
# for i in range(len(hist)):
#     print(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {hist[i]} values ({hist[i]/similarity_matrix.size*100:.2f}%)")



# # # Optionally, save the matrix to a file for later use
# # np.save("similarity_matrix.npy", similarity_matrix)

# # You can also create a CSV version, though it might be large
# similarity_df = pd.DataFrame(
#     similarity_matrix,
#     index=df_stacked.index, 
#     columns=df_stacked.index
# )
# # Save a sample part to CSV (first 100×100 entries)
# similarity_df.to_csv("similarity_matrix_sample.csv")




final_df = find_semantic_groups(
    table1=df_left,
    table2=df_right,
    columns=use_columns,
    rm=rm,
    k_neighbors=3,
    skip_same_side=False  # or True
)

print("Representatives of each connected semantic group:")
print(final_df)
print(len(final_df))

final_df.to_csv("tests/knn3_amazon_L6_l2_0.4.csv", index=False)


ground_truth_df = pd.read_csv("processed_result_3290.csv")  

# Extract record_ids
# ground_truth_ids = set(ground_truth_df['record_id'])
# retrieved_ids = set(final_df['record_id'])

ground_truth_ids = set(ground_truth_df['id'])
retrieved_ids = set(final_df['id'])

# Compute how many ground truth records are correctly retrieved
true_positives = ground_truth_ids.intersection(retrieved_ids)
recall = len(true_positives) / len(ground_truth_ids)
precision = len(true_positives) / len(retrieved_ids)

print(f"Recall: {recall:.4f}")
print(f"Correctly retrieved: {len(true_positives)} / {len(ground_truth_ids)}")
print(f"Precision: {precision:.4f}")
print(f"Correctly retrieved: {len(true_positives)} / {len(retrieved_ids)}")


