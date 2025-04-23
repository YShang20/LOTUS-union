import pandas as pd
import numpy as np
from typing import List, Dict, Set, Any
from tqdm import tqdm

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticFilterOutput
from lotus.utils import show_safe_mode
from .postprocessors import filter_postprocess
from sentence_transformers import SentenceTransformer
import hnswlib
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from lotus.types import CascadeArgs
from lotus.utils import show_safe_mode
from .cascade_utils import importance_sampling, learn_cascade_thresholds

from sklearn.preprocessing import normalize



def build_hnsw_index(embeddings: np.ndarray, space='cosine', ef_construction=200, M=16) -> hnswlib.Index:
    """
    Build an HNSWlib index given 2D embeddings: shape [num_rows, embedding_dim].
    """
    dim = embeddings.shape[1]
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
    index.add_items(embeddings, np.arange(len(embeddings)))
    # At query-time, setting ef higher helps accuracy
    index.set_ef(50)
    return index

def compute_embedding(df_in: pd.DataFrame):
    rows, cols, data = [], [], []

    for row_idx, bow_str in enumerate(df_in['bow']):
        # Ensure the string is clean
        bow_str = str(bow_str).strip("[] ")
        if not bow_str:
            continue  # skip empty rows
        for item in bow_str.split(','):
            try:
                cleaned = item.strip().strip("'").strip('"')  # Remove extra quotes
                if ':' not in cleaned:
                    print(f"Skipping malformed item '{item}' in row {row_idx}")
                    continue
                idx, count = cleaned.split(':')
                rows.append(row_idx)
                cols.append(int(idx.strip()) - 1)
                data.append(int(count.strip()))
            except Exception as e:
                print(f"Skipping malformed item '{item}' in row {row_idx} due to error: {e}")
                continue

    if not data:
        raise ValueError("No valid bow data parsed.")

    num_docs = df_in.shape[0]
    vocab_size = max(cols) + 1 if cols else 0

    X_tf = csr_matrix((data, (rows, cols)), shape=(num_docs, vocab_size))

    # Apply TF-IDF
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_tf)
    return X_tfidf.toarray()


def find_semantic_groups(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    columns: list[str],
    rm: SentenceTransformer = None,
    k_neighbors: int = 5,
    skip_same_side: bool = False,
) -> pd.DataFrame:
    """
    1) Creates embeddings for each row in merged_df.
    2) Builds a single HNSW index.
    3) For each row, queries k_neighbors from the index.
    4) Builds a graph of mutually-similar rows, then returns one representative per group.

    If skip_same_side=True, we ignore neighbors from the same 'source' column.
    """
    # Step A: embed
    embed_model = rm

    df_stacked = pd.concat([table1, table2], ignore_index=True)
    df_stacked.to_feather("metadata.feather")

    #df_result = df_stacked.apply(lambda row: " ".join([str(row[col]) for col in columns if pd.notna(row[col])]), axis=1).tolist()
    df_result = df_stacked.apply(
    lambda row: " | ".join([f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]),
    axis=1).tolist()
    embeddings = embed_model._embed(df_result)
    #embeddings_df = pd.read_parquet('/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus-584/tests/tfidf_sparse.parquet')
    # embeddings_df = pd.concat([embeddings_df,embeddings_df])
    # embeddings = embeddings_df.to_numpy(dtype=np.float32)
   #  embeddings = compute_embedding(df_stacked)
    index = build_hnsw_index(embeddings, space='l2', ef_construction=200, M=16)

    adjacency_list = defaultdict(set)
    n_rows = len(df_stacked)

    labels, distances = index.knn_query(embeddings, k=k_neighbors)


    filtered_neighbors = []
    for neighbors, dists in zip(labels, distances):
        filtered = [n for n, d in zip(neighbors, dists) if d <=0.4] 
        filtered_neighbors.append(filtered)
        #print(filtered)


    for row_idx in range(n_rows):
        row_source = df_stacked.loc[row_idx, 'source']
        neighs = filtered_neighbors[row_idx]
        dists = distances[row_idx]

        for i, neighbor_idx in enumerate(neighs):
            if neighbor_idx == row_idx:
                # skip self-match, or we can keep it if we want
                continue
            if skip_same_side:
                # skip if both come from the same side
                if df_stacked.loc[neighbor_idx, 'source'] == row_source:
                    continue
            adjacency_list[row_idx].add(neighbor_idx)
            adjacency_list[neighbor_idx].add(row_idx)  # undirected

    # Step D: find connected components via DFS
    visited = set()
    groups = []

    def dfs(node, group):
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.add(current)
            for neighbor in adjacency_list[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

    for node in range(n_rows):
        if node not in visited:
            comp = set()
            dfs(node, comp)
            groups.append(comp)

    # Step E: pick a representative row from each group
    # we'll pick the row with smallest index in that group
    # but you might want a different logic
    rep_indices = [min(g) for g in groups]

    # Step F: return them as a DataFrame
    rep_rows = [df_stacked.iloc[idx].copy() for idx in rep_indices]
    return pd.DataFrame(rep_rows)

def build_similarity_matrix(table1, table2, columns, rm, full_matrix=False, is_stacked=False):
    """
    Build a similarity matrix where each entry (i,j) is the similarity score between row i and row j.
    
    Parameters:
    - table1, table2: DataFrames to compare
    - columns: Columns to use for comparison
    - rm: Retrieval model for embedding
    - full_matrix: If True, compute the full n×n matrix; if False, only compute nearest neighbors 
                  (more memory efficient)
    - is_stacked: If True, assumes table1 is already the stacked result of the two tables
                 and table2 is ignored.
    
    Returns:
    - similarity_matrix: n×n numpy array of similarity scores
    - df_stacked: The combined DataFrame
    """
    # Step 1: Combine tables and create text representation
    if not is_stacked:
        df_stacked = pd.concat([table1, table2], ignore_index=True)
    else:
        df_stacked = table1  # Already stacked
        
    n_rows = len(df_stacked)
    
    # Create text representation for each row
    df_result = df_stacked.apply(
        lambda row: " ".join([str(row[col]) for col in columns if pd.notna(row[col])]), 
        axis=1
    ).tolist()
    
    
    # Step 2: Embed the texts
    embeddings = rm._embed(df_result)

    embeddings = normalize(embeddings, norm='l2', axis=1)
    # Step 3: Build HNSW index for efficient similarity computation
    index = build_hnsw_index(embeddings, space='l2', ef_construction=500, M=64)
    
    # Step 4: Create the similarity matrix
    pairs = []  
    scores = [] 
    if full_matrix:
        similarity_matrix = np.zeros((n_rows, n_rows))
        
        try:
            # Try to query each point against all others with large k value
            labels, distances = index.knn_query(embeddings, k=n_rows)
            
            # Fill the similarity matrix
            for i in range(n_rows):
                for j_idx, j in enumerate(labels[i]):
                    if i == j:
                        similarity_matrix[i, j] = 0
                    else:
                        #  similarity_matrix[i, j] = 1.0 - min(1.0, distances[i][j_idx])
                        similarity_matrix[i, j] = np.exp(-distances[i][j_idx]**2 / (2 * 0.5**2))

        except RuntimeError as e:
            # Fallback to pairwise cosine similarity if KNN fails
            print("KNN query failed. Falling back to pairwise cosine similarity calculation.")
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            
            # Compute pairwise cosine similarity
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            # Set diagonal to zero (no self-matches)
            np.fill_diagonal(similarity_matrix, 0)
    else:
        # Use a sparse approach - only compute for k nearest neighbors
        k = min(50, n_rows-1)  # Set a reasonable k value
        similarity_matrix = np.zeros((n_rows, n_rows))

        try:
            # Query the index for top-k nearest neighbors for each point
            labels, distances = index.knn_query(embeddings, k=k)
            
            # Fill the similarity matrix with known values
            for i in range(n_rows):
                for j_idx, j in enumerate(labels[i]):
                    if j >= n_rows:
                        continue
                    # Convert distance to similarity score (1 - distance) and clip to [0,1]
                    similarity_matrix[i, j] = 1.0 - min(1.0, distances[i][j_idx])
        except RuntimeError as e:
            # Fallback to pairwise cosine similarity if KNN fails
            print("KNN query failed. Falling back to pairwise cosine similarity calculation.")
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            
            # Compute pairwise cosine similarity
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            # Set diagonal to zero (no self-matches)
            np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix, df_stacked, pairs, scores
def print_similarity_stats(similarity_matrix):
    """
    Calculate and print statistics about the similarity score distribution.
    
    Parameters:
    - similarity_matrix: A square matrix of similarity scores
    
    Returns:
    - None (prints statistics to console)
    """
    # Extract scores from upper triangle (excluding diagonal) to avoid counting pairs twice
    n = similarity_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    similarity_scores = similarity_matrix[upper_triangle_indices]
    
    # Calculate basic statistics
    mean_score = np.mean(similarity_scores)
    median_score = np.median(similarity_scores)
    min_score = np.min(similarity_scores)
    max_score = np.max(similarity_scores)
    std_score = np.std(similarity_scores)
    
    # Calculate percentiles
    percentiles = np.percentile(similarity_scores, [10, 25, 75, 90, 95, 99])
    
    # Count scores in different ranges
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(similarity_scores, bins=bins)
    # Print the statistics
    print("\nSimilarity Score Distribution Statistics:")
    print(f"Number of pairwise comparisons: {len(similarity_scores)}")
    print(f"Mean similarity: {mean_score:.4f}")
    print(f"Median similarity: {median_score:.4f}")
    print(f"Min similarity: {min_score:.4f}")
    print(f"Max similarity: {max_score:.4f}")
    print(f"Standard deviation: {std_score:.4f}")
    
    print("\nHistogram (bins of 0.1):")
    for i, count in enumerate(hist):
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}: {count} ({count/len(similarity_scores)*100:.2f}%)")
        

def exact_match(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    columns1: List[str] = None,
    columns2: List[str] = None,
    is_stacked: bool = False,
) -> np.ndarray:
    """
    Implements an efficient exact-match algorithm using sort-merge join approach.
    
    Args:
        table1 (pd.DataFrame): First table to compare
        table2 (pd.DataFrame): Second table to compare
        columns1 (List[str]): Columns from table1 to use for matching. If None, all columns are used.
        columns2 (List[str]): Columns from table2 to use for matching. If None, uses columns1.
        is_stacked (bool): If True, assumes table1 is already the stacked result of the two tables
                          and table2 is ignored.
    
    Returns:
        np.ndarray: An nxn boolean matrix where n is the total number of rows in the combined tables.
                   Matrix[i,j] = 1 if rows i and j match exactly, 0 otherwise.
                   Diagonal elements (i=j) are always 0.
    """
    # Default to all columns if none specified
    if columns1 is None:
        columns1 = list(table1.columns)
    if columns2 is None:
        columns2 = columns1.copy()
    
    # Validate column existence
    for col in columns1:
        if col not in table1.columns:
            raise ValueError(f"Column '{col}' not found in table1")
    
    if not is_stacked:
        for col in columns2:
            if col not in table2.columns:
                raise ValueError(f"Column '{col}' not found in table2")
        
        # Stack the tables
        df_stacked = pd.concat([table1, table2], ignore_index=True)
        n_rows_t1 = len(table1)
    else:
        # If already stacked, just use table1
        df_stacked = table1
        n_rows_t1 = len(table1) // 2
    
    # Create concatenated strings for sorting
    concat_values = []
    for idx, row in df_stacked.iterrows():
        # Determine which set of columns to use based on whether the row came from table1 or table2
        cols = columns1 if idx < n_rows_t1 else columns2
        # Convert row values to strings and concatenate
        row_str = "|".join(str(row[col]) for col in cols)
        concat_values.append(row_str)
    
    df_stacked['_concat_key'] = concat_values
    
    # Sort by the concatenated string
    df_sorted = df_stacked.sort_values('_concat_key')
    sorted_indices = df_sorted.index.values
    
    # Initialize result matrix (n x n) with zeros
    n = len(df_stacked)
    result_matrix = np.zeros((n, n), dtype=int)
    
    # Iterate through sorted data to find matching rows efficiently
    i = 0
    while i < len(df_sorted) - 1:
        j = i + 1
        # Check if current row matches the next row
        while j < len(df_sorted) and df_sorted.iloc[i]['_concat_key'] == df_sorted.iloc[j]['_concat_key']:
            # Get original indices
            orig_i = sorted_indices[i]
            orig_j = sorted_indices[j]
            # Set match in result matrix (1 for match)
            result_matrix[orig_i, orig_j] = 1
            result_matrix[orig_j, orig_i] = 1  # Symmetric matrix
            j += 1
        i += 1
    
    # Ensure diagonal is 0 (no self-matches)
    np.fill_diagonal(result_matrix, 0)
    
    # Delete the temporary concat key column
    if '_concat_key' in df_stacked.columns:
        df_stacked.drop('_concat_key', axis=1, inplace=True)
    
    return result_matrix

def learn_union_thresholds(sim_matrix: np.ndarray,
                           pairs_in: np.ndarray, # this is only useful if using the dense approach, remember to also pass in scores_in
                         combined_df: pd.DataFrame,
                         columns1: List[str],
                         columns2: List[str],
                         user_instr: str,
                         cascade_args: CascadeArgs,
                         examples_mm=None, examples_ans=None,
                         cot=None, strategy=None, default=True):
    """
    Return (tau_pos, tau_neg) learned from a sampled subset of pairwise
    similarities.
    """
    n = len(sim_matrix)
    pairs   = [(i, j) for i in range(n) for j in range(i+1, n) if sim_matrix[i, j] >= 0.2]
    scores  = [sim_matrix[i, j] for i, j in pairs if sim_matrix[i, j] >= 0.2]

    samp_idx, corr = importance_sampling(scores, cascade_args)
    print("sampidex:", len(samp_idx))
    
    samp_pairs  = [pairs[k] for k in samp_idx]
    print('sampairs:', len(samp_pairs))
    samp_scores = [scores[k] for k in samp_idx]
    samp_corr   = corr[samp_idx]
    
    # ----- build LLM docs only for the sample -----
    docs = []
    for (i, j) in samp_pairs:
        r1 = combined_df.iloc[i][columns1].tolist()
        r2 = combined_df.iloc[j][columns2].tolist()
        docs.append({"text": f"Row1: {r1} | Row2: {r2}"})

    lm_prompts = [
        task_instructions.union_formatter(
            d, user_instr, examples_mm,
            ["True" if x else "False" for x in examples_ans] if examples_ans else None,
            cot, strategy)
        for d in docs
    ]
    llm_call_count = len(samp_pairs)
    print ("LLM calls:", llm_call_count)
    #sys.exit()

    out = lotus.settings.lm(
        lm_prompts, show_progress_bar=True,
        progress_bar_desc="Oracle for threshold learning"
    )
    oracle_bool = filter_postprocess(out.outputs, default=False).outputs

    (pos_threshold, neg_threshold), _ = learn_cascade_thresholds(
        proxy_scores=samp_scores,
        oracle_outputs=oracle_bool,
        sample_correction_factors=samp_corr,
        cascade_args=cascade_args
    )
    return pos_threshold, neg_threshold



from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

def detect_valley(
        scores,
        default_lower,
        default_upper,
        bins='auto',
        smooth_sigma=1.5,
        valley_prominence=0.01      # ignore negligible dips
    ):
    """
    Unsupervised threshold detection by valley‑finding on a smoothed histogram.

    Returns (lower_threshold, upper_threshold).  If the distribution is unimodal
    or no prominent valleys appear, falls back to defaults.
    """
    scores = np.asarray(scores, dtype=float)


    # keep only finite scores
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return default_lower, default_upper
    # normalise to [0,1] if necessary
    if scores.min() < 0 or scores.max() > 1:
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # histogram + smoothing ---------------------------------------------------
    hist, edges = np.histogram(scores, bins=bins, range=(0, 1), density=True)
    pdf = gaussian_filter1d(hist, sigma=smooth_sigma)

    # valleys = local minima whose depth is significant ----------------------
    minima = argrelextrema(pdf, np.less)[0]
    # keep only valleys with enough prominence
    prominences = pdf[minima]
    minima = minima[prominences < pdf.max() * (1 - valley_prominence)]

    if minima.size == 0:            # unimodal → use defaults / quantiles
        return default_lower, default_upper
    elif minima.size == 1:          # one clear valley
        t = (edges[minima[0]] + edges[minima[0] + 1]) / 2.0
        return  t, default_upper
    elif minima.size > 2:                           # > 2 valleys → take first & second to last
        t1 = (edges[minima[0]]     + edges[minima[0] + 1]) / 2.0
        t2 = (edges[minima[-2]]    + edges[minima[-2] + 1]) / 2.0
        # if max(t1, t2) > default_upper:
        #     return min(t1, t2), default_upper
        return min(t1, t2), max(t1, t2)
    else:                           # = 2 valleys → take first & last
        t1 = (edges[minima[0]]     + edges[minima[0] + 1]) / 2.0
        t2 = (edges[minima[-1]]    + edges[minima[-1] + 1]) / 2.0
        return min(t1, t2), max(t1, t2)

#----

def gold_union(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    columns1: List[str],
    columns2: List[str],
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Union comparisons",
    additional_cot_instructions: str = "",
) -> pd.DataFrame:
    """
    Implements the semantic union operator using LLM for all row comparisons.
    
    For each row in table1 and each row in table2, a document is created (containing the two rows' values)
    and passed to the LLM using a union_formatter with the provided user_instruction.
    The model is expected to answer simply "True" (if the rows match) or "False".
    
    The resulting boolean outputs are used to build a bipartite match graph between rows of table1 and table2.
    A DFS then finds connected components (groups of mutually matching rows). From each group, a representative
    row (the one with the smallest index) is selected and the final result is returned as a DataFrame.
    
    Args:
        table1 (pd.DataFrame): Left table.
        table2 (pd.DataFrame): Right table.
        columns1 (List[str]): Columns from table1 to compare.
        columns2 (List[str]): Columns from table2 to compare.
        user_instruction (str): Instruction for the LLM; e.g. "Do these rows match exactly? Answer True or False."
        default (bool): Default value for filtering in case of parsing errors.
        examples_multimodal_data (list[dict[str, Any]] | None): Example documents, if any.
        examples_answers (list[bool] | None): Example answers, if any.
        cot_reasoning (list[str] | None): Chain-of-thought reasoning examples.
        strategy (str | None): Reasoning strategy.
        safe_mode (bool): If True, display cost estimates.
        show_progress_bar (bool): Whether to display a progress bar.
        progress_bar_desc (str): Description for the progress bar.
        additional_cot_instructions (str): Extra instructions for the LLM.
    
    Returns:
        pd.DataFrame: DataFrame of representative rows from each matched group.
    """
    combined_table = pd.concat([table1, table2], ignore_index=True)
    table1 = combined_table
    table2 = combined_table

    # Build the list of comparison docs and record mapping of (i, j) for each comparison.
    docs: List[dict[str, Any]] = []
    mapping: List[tuple[int, int]] = []
    for i, row1 in table1[columns1].iterrows():
        for j, row2 in table2[columns2].iterrows():
            # Skip comparing a row with itself
            if i == j:
                continue
            # Create a document that contains both row values
            doc = {
                "text": f"Row1: {row1.tolist()} | Row2: {row2.tolist()}"
            }
            docs.append(doc)
            mapping.append((i, j))
    
    # Convert boolean example answers to strings ("True" / "False") for the union formatter.
    ex_answer_strs = None
    if examples_answers is not None:
        ex_answer_strs = ["True" if ans else "False" for ans in examples_answers]
    
    # Generate prompts using the union_formatter instead of filter_formatter.
    inputs: List[List[dict[str, str]]] = []
    for doc in docs:
        prompt = task_instructions.union_formatter(
            doc,
            user_instruction,
            examples_multimodal_data,
            ex_answer_strs,
            cot_reasoning,
            strategy,
            reasoning_instructions=additional_cot_instructions,
        )
        lotus.logger.debug(f"LLM prompt: {prompt}")
        inputs.append(prompt)
    
    kwargs: dict[str, Any] = {"logprobs": False}
    if safe_mode:
        estimated_total_calls = len(inputs)
        estimated_total_cost = sum(lotus.settings.lm.count_tokens(inp) for inp in inputs)
        show_safe_mode(estimated_total_cost, estimated_total_calls)
    
    # Call the LLM (lotus.settings.lm) with all the prompts.
    lm_output: LMOutput = lotus.settings.lm(
        inputs,
        show_progress_bar=show_progress_bar,
        progress_bar_desc=progress_bar_desc,
        **kwargs
    )
    
    # Postprocess outputs using the same postprocessor as sem_filter.
    postprocess_output: SemanticFilterOutput = filter_postprocess(lm_output.outputs, default=False)
    outputs_bool: List[bool] = postprocess_output.outputs  # Expecting a list of booleans ("True" -> True, "False" -> False)
    
    # Build match lists for table1 and table2.
    n_rows_t1 = table1.shape[0]
    n_rows_t2 = table1.shape[0]  # Since they're the same table
    matches_t1: List[List[int]] = [[] for _ in range(n_rows_t1)]
    matches_t2: List[List[int]] = [[] for _ in range(n_rows_t2)]
    
    for idx, result in enumerate(outputs_bool):
        if result:
            i, j = mapping[idx]
            matches_t1[i].append(j)
            matches_t2[j].append(i)
    
    matches = []

    # Combine matches from both sides while removing duplicates
    for i in range(len(matches_t1)):
        elements1 = matches_t1[i]
        elements2 = matches_t2[i]
        
        combined = []
        
        # Add elements from matches_t1
        for element in elements1:
            if element not in combined:
                combined.append(element)
                
        # Add elements from matches_t2
        for element in elements2:
            if element not in combined:
                combined.append(element)
        
        matches.append(combined)

    # Build a bipartite graph for connected components
    graph: Dict[int, Set[int]] = {}
    
    # Initialize the graph with all nodes
    for i in range(n_rows_t1):
        graph.setdefault(i, set())
    
    # Add edges based on matches
    for i in range(len(matches)):
        for j in matches[i]:
            graph[i].add(j)
            graph[j].add(i)
    
    # Use DFS to find connected components (groups of mutually matching rows)
    visited: Set[int] = set()
    groups: List[Set[int]] = []
    
    def dfs(node: int, group: Set[int]):
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.add(current)
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    for node in graph.keys():
        if node not in visited:
            group = set()
            dfs(node, group)
            groups.append(group)
    
    # From each group, choose the representative row as the one with the smallest index
    rep_indices = []
    for group in groups:
        rep = min(group)
        rep_indices.append(rep)
    
    # Build the final result dataframe using only the representative indices
    final_rows = []
    for idx in sorted(rep_indices):
        row = combined_table.iloc[idx].copy()
        final_rows.append(row)
    
    result_df = pd.DataFrame(final_rows)
    return result_df


def sem_union(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    columns1: List[str],
    columns2: List[str],
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Union comparisons",
    additional_cot_instructions: str = "",
    sim_upper_threshold: float = 0.8,  # High similarity threshold
    sim_lower_threshold: float = 0.3,  # Low similarity threshold
    embedding_model: SentenceTransformer = None,
    auto_threshold: str = "None",  # "None", "Valley", "Oracle"
) -> pd.DataFrame:
    """
    Implements a three-level semantic union operator with progressive matching strategies:
    1. Exact match: Find rows that match exactly
    2. Embedding similarity: Match rows with high similarity or filter out low similarity pairs
    3. LLM-based matching: Use LLM for the remaining undecided pairs
    
    Args:
        table1 (pd.DataFrame): Left table.
        table2 (pd.DataFrame): Right table.
        columns1 (List[str]): Columns from table1 to compare.
        columns2 (List[str]): Columns from table2 to compare.
        user_instruction (str): Instruction for the LLM
        default (bool): Default value for filtering in case of parsing errors.
        examples_multimodal_data (list[dict[str, Any]] | None): Example documents.
        examples_answers (list[bool] | None): Example answers.
        cot_reasoning (list[str] | None): Chain-of-thought reasoning examples.
        strategy (str | None): Reasoning strategy.
        safe_mode (bool): Show cost estimates.
        show_progress_bar (bool): Show progress bar.
        progress_bar_desc (str): Progress bar description.
        additional_cot_instructions (str): Additional instructions for LLM.
        sim_upper_threshold (float): Similarity threshold above which rows are considered a match.
        sim_lower_threshold (float): Similarity threshold below which rows are considered not a match.
        embedding_model (SentenceTransformer): Model for creating embeddings.
        auto_threshold (bool): Whether to automatically determine thresholds.
    Returns:
        pd.DataFrame: DataFrame of representative rows from each matched group.
    """
    cascade_args = CascadeArgs()
    # Stack the tables together
    combined_table = pd.concat([table1, table2], ignore_index=True)
    n_rows_total = len(combined_table)
    n_rows_t1 = len(table1)
    
    # Initialize the match matrix with -1 (unknown match status)
    # -1: not yet determined, 0: not a match, 1: a match
    match_matrix = np.full((n_rows_total, n_rows_total), -1, dtype=np.int8)
    np.fill_diagonal(match_matrix, 0)  # Set diagonal to non-match (don't match with self)
    
    # Step 1: Exact matching
    if show_progress_bar:
        print("Step 1: Running exact matching...")
    
    exact_match_matrix = exact_match(
        table1=combined_table, 
        table2=None, 
        columns1=columns1, 
        columns2=columns2,
        is_stacked=True 
    )
    
    # Count exact matches
    exact_match_count = 0
    exact_match_pairs = []
    
    # Update match matrix with exact matches (1)
    match_indices = np.where(exact_match_matrix == 1)
    for i, j in zip(match_indices[0], match_indices[1]):
        # Make sure indices are valid
        if i < n_rows_total and j < n_rows_total:
            if i < j:  # Only count each pair once
                exact_match_count += 1
                exact_match_pairs.append((i, j))
            match_matrix[i, j] = 1
    
    if show_progress_bar:
        print(f"Layer 1 (Exact Match) - Found {exact_match_count} matching pairs")
    
    # Step 2: Embedding similarity matching
    if show_progress_bar:
        print("\nStep 2: Running embedding similarity matching...")
    
    # If no embedding model provided, create one
    if embedding_model is None:
        # Try to use the default model
        try:
            embedding_model = lotus.settings.retrieval_model
        except AttributeError:
            # Fall back to a standard sentence transformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate similarity matrix
    similarity_matrix, _, dense_pairs, dense_scores = build_similarity_matrix(
        table1=combined_table,
        table2=None,  # Not used when is_stacked=True
        columns=columns1,  # Use all columns for comparison
        rm=embedding_model,
        full_matrix=True,  # We need the full matrix
        is_stacked=True  # Let the function know we've already stacked the tables
    )
    # sim_upper_threshold, sim_lower_threshold = learn_union_thresholds(similarity_matrix, combined_table, columns1, columns2, user_instruction, cascade_args)
    # print("sim_upper_threshold, sim_lower_threshold", sim_upper_threshold, sim_lower_threshold)
    print_similarity_stats(similarity_matrix)

    # Auto-determine thresholds if requested
    if auto_threshold == "Valley":
        # Extract upper triangle of similarity matrix (avoiding diagonal)
        similarity_scores = []
        for i in range(n_rows_total):
            for j in range(i + 1, n_rows_total):
                # Skip if already matched exactly
                if match_matrix[i, j] != 1 and similarity_matrix[i, j] > 0.1:
                    similarity_scores.append(similarity_matrix[i, j])
                    
        if len(similarity_scores) > 10:
            if show_progress_bar:
                print("Automatically determining similarity thresholds based on score distribution...")
            
            lower_threshold, upper_threshold = detect_valley(
                similarity_scores, 
                default_lower=sim_lower_threshold,
                default_upper=sim_upper_threshold
            )
            
            if show_progress_bar:
                print(f"Auto-detected thresholds - Lower: {lower_threshold:.4f}, Upper: {upper_threshold:.4f}")
            
            # Update the thresholds
            sim_lower_threshold = lower_threshold
            sim_upper_threshold = upper_threshold
    elif auto_threshold == "Oracle":
            # this is how you would call it when only limiting k = 50 closest neighbors to save LLM calls
            #sim_upper_threshold, sim_lower_threshold = learn_union_thresholds(dense_scores, dense_pairs, combined_table, columns1, columns2, user_instruction, cascade_args)
            sim_upper_threshold, sim_lower_threshold = learn_union_thresholds(similarity_matrix, dense_pairs, combined_table, columns1, columns2, user_instruction, cascade_args)

            
    
    # Count similarity-based matches/non-matches
    high_sim_count = 0
    low_sim_count = 0
    undecided_count = 0
    high_sim_pairs = []
    low_sim_pairs = []
    
    # Update match matrix based on similarity thresholds
    # High similarity -> match (1)
    # Low similarity -> non-match (0)
    # In-between -> still unknown (-1)
    for i in range(n_rows_total):
        for j in range(i+1, n_rows_total):  # Only process upper triangle to avoid duplicates
            if i >= n_rows_total or j >= n_rows_total:
                continue  # Skip invalid indices
                
            if match_matrix[i, j] != -1:
                continue  # Skip already determined pairs
            
            similarity = similarity_matrix[i, j]
            
            if similarity >= sim_upper_threshold:
                match_matrix[i, j] = 1  # High similarity -> match
                match_matrix[j, i] = 1  # Keep matrix symmetric
                high_sim_count += 1
                high_sim_pairs.append((i, j, similarity))
            elif similarity <= sim_lower_threshold:
                match_matrix[i, j] = 0  # Low similarity -> non-match
                match_matrix[j, i] = 0  # Keep matrix symmetric
                low_sim_count += 1
                low_sim_pairs.append((i, j, similarity))
            else:
                undecided_count += 1
    
    if show_progress_bar:
        print(f"Layer 2 (Embedding Similarity) - Found {high_sim_count} high-similarity matching pairs")
        print(f"Layer 2 (Embedding Similarity) - Found {low_sim_count} low-similarity non-matching pairs")
        print(f"Layer 2 (Embedding Similarity) - Left {undecided_count} pairs undecided")
        
    
    # Step 3: LLM matching for remaining uncertain pairs
    if show_progress_bar:
        if undecided_count > 0:
            print(f"\nStep 3: Running LLM matching for {undecided_count} undecided pairs...")
        else:
            print("\nStep 3: No undecided pairs, skipping LLM matching.")
    
    # Create mapping and docs list for LLM comparison
    docs = []
    mapping = []
    
    for i in range(n_rows_total):
        for j in range(i + 1, n_rows_total):  # Only need to check one side due to symmetry
            if match_matrix[i, j] == -1:  # Only process undecided pairs
                row1 = combined_table.iloc[i]
                row2 = combined_table.iloc[j]
                
                # Get the similarity score for this pair
                similarity = similarity_matrix[i, j]
                
                # Create a document for LLM comparison that includes similarity score
                doc = {
                    "text": f"Row1: {row1[columns1].tolist()} | Row2: {row2[columns2].tolist()} | Similarity Score: {similarity:.4f}"
                }
                docs.append(doc)
                mapping.append((i, j))
    
    # Skip LLM call if no undecided pairs
    llm_match_count = 0
    llm_non_match_count = 0
    llm_match_pairs = []
    if docs:
        # Convert boolean example answers to strings
        ex_answer_strs = None
        if examples_answers is not None:
            ex_answer_strs = ["True" if ans else "False" for ans in examples_answers]
        
        # Generate prompts
        inputs = []
        for doc in docs:
            prompt = task_instructions.union_formatter(
                doc,
                user_instruction,
                examples_multimodal_data,
                ex_answer_strs,
                cot_reasoning,
                strategy,
                reasoning_instructions=additional_cot_instructions,
            )
            lotus.logger.debug(f"LLM prompt: {prompt}")
            inputs.append(prompt)
        
        # Cost estimate if in safe mode
        if safe_mode:
            estimated_total_calls = len(inputs)
            estimated_total_cost = sum(lotus.settings.lm.count_tokens(inp) for inp in inputs)
            show_safe_mode(estimated_total_cost, estimated_total_calls)
        
        # Call LLM
        lm_output = lotus.settings.lm(
            inputs,
            show_progress_bar=show_progress_bar,
            progress_bar_desc=progress_bar_desc,
            logprobs=False
        )
        
        # Process LLM outputs
        postprocess_output = filter_postprocess(lm_output.outputs, default=False)
        outputs_bool = postprocess_output.outputs
        
        # Update match matrix with LLM results
        for idx, result in enumerate(outputs_bool):
            i, j = mapping[idx]
            if i < n_rows_total and j < n_rows_total:  # Ensure indices are valid
                if result:
                    match_matrix[i, j] = 1
                    match_matrix[j, i] = 1  # Keep the matrix symmetric
                    llm_match_count += 1
                    llm_match_pairs.append((i, j))
                else:
                    match_matrix[i, j] = 0
                    match_matrix[j, i] = 0  # Keep the matrix symmetric
                    llm_non_match_count += 1
                
        if show_progress_bar and len(llm_match_pairs) > 0:
            print(f"Layer 3 (LLM) - Found {llm_match_count} matching pairs and {llm_non_match_count} non-matching pairs")
            print("\nLLM matching pairs (first 5):")
            for i, (idx1, idx2) in enumerate(llm_match_pairs[:5]):
                row1 = combined_table.iloc[idx1]
                row2 = combined_table.iloc[idx2]
                print(f"Pair {i+1}: Row {idx1} and Row {idx2}")
                print(f"  Title 1: {row1['title'] if 'title' in row1 else 'N/A'}")
                print(f"  Title 2: {row2['title'] if 'title' in row2 else 'N/A'}")
    
    if show_progress_bar:
        print("\nSummary Statistics:")
        total_comparisons = (n_rows_total * (n_rows_total-1)) // 2
        print(f"Total number of pairwise comparisons: {total_comparisons}")
        print(f"Layer 1 (Exact Match): {exact_match_count} matches ({exact_match_count / total_comparisons * 100:.2f}%)")
        print(f"Layer 2 (Embedding Similarity): {high_sim_count} matches, {low_sim_count} non-matches")
        print(f"Layer 3 (LLM): {llm_match_count} matches, {llm_non_match_count} non-matches")
    
    # Build the graph for connected components
    graph = defaultdict(set)
    
    # Add edges based on match matrix
    for i in range(n_rows_total):
        for j in range(n_rows_total):
            if match_matrix[i, j] == 1:  # If it's a match
                graph[i].add(j)
                graph[j].add(i)
    
    # DFS to find connected components
    visited = set()
    groups = []
    
    def dfs(node, group):
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    for node in range(n_rows_total):
        if node not in visited:
            group = set()
            dfs(node, group)
            groups.append(group)
    
    # Select representative row from each group
    rep_indices = []
    for group in groups:
        rep = min(group)  # Choose the row with the smallest index
        rep_indices.append(rep)
    
    # Build final result DataFrame
    final_rows = []
    for idx in sorted(rep_indices):
        row = combined_table.iloc[idx].copy()
        final_rows.append(row)
    
    result_df = pd.DataFrame(final_rows)
    return result_df

@pd.api.extensions.register_dataframe_accessor("sem_union")
class SemUnionDataFrame:
    """
    DataFrame accessor for sem_union operator.
    
    Enables using sem_union as a method on DataFrames.
    For example:
        result_df = df_left.sem_union(df_right, join_instruction="A:left, B:left, X:right, Y:right")
    
    The join_instruction string is parsed to determine which columns from the left and right tables are to be compared.
    """
    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("sem_union accessor can only be used with DataFrames.")

    @operator_cache
    def __call__(
        self,
        other: pd.DataFrame | pd.Series,
        join_instruction: str,
        safe_mode: bool = False,
        show_progress_bar: bool = True,
        sim_upper_threshold: float = 0.8,
        sim_lower_threshold: float = 0.3,
        embedding_model: SentenceTransformer = None,
    ) -> pd.DataFrame:
        """
        Applies sem_union between this DataFrame and another DataFrame/Series.
        
        Args:
            other (pd.DataFrame | pd.Series): The other table to union with.
            join_instruction (str): A string specifying which columns to compare.
                For example, "A:left, B:left, X:right, Y:right" indicates that columns A and B in the left DataFrame
                are to be compared with columns X and Y in the right DataFrame.
            safe_mode (bool): Whether to show cost estimates.
            show_progress_bar (bool): Whether to display a progress bar.
            sim_upper_threshold (float): Similarity threshold above which rows are considered a match.
            sim_lower_threshold (float): Similarity threshold below which rows are considered not a match.
            embedding_model (SentenceTransformer): Model for creating embeddings.
        
        Returns:
            pd.DataFrame: A DataFrame containing the representative rows from each connected match group.
        """
        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})
        
        # Parse the join_instruction string.
        cols = lotus.nl_expression.parse_cols(join_instruction)
        left_columns: List[str] = []
        right_columns: List[str] = []
        for col in cols:
            if ":left" in col:
                left_columns.append(col.split(":left")[0])
            elif ":right" in col:
                right_columns.append(col.split(":right")[0])
            else:
                if col in self._obj.columns:
                    left_columns.append(col)
                elif col in other.columns:
                    right_columns.append(col)
        
        if not left_columns or not right_columns:
            raise ValueError("Both left and right columns must be specified in join_instruction for sem_union operator.")
        
        # Create a user instruction for row comparison.
        user_instruction = (
            f"Compare these two rows with the following columns and determine if they represent the same entity or information.\n\n"
            f"Row 1 columns: {left_columns}\n"
            f"Row 2 columns: {right_columns}\n\n"
        )
        result = sem_union(
            table1=self._obj,
            table2=other,
            columns1=left_columns,
            columns2=right_columns,
            user_instruction=user_instruction,
            safe_mode=safe_mode,
            show_progress_bar=show_progress_bar,
            sim_upper_threshold=sim_upper_threshold,
            sim_lower_threshold=sim_lower_threshold,
            embedding_model=embedding_model,
        )
        return result
