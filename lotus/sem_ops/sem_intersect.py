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
from .sem_union import build_hnsw_index, compute_embedding, build_similarity_matrix, exact_match

from sklearn.preprocessing import normalize


def sem_intersect(
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
    progress_bar_desc: str = "Intersect comparisons",
    additional_cot_instructions: str = "",
    sim_upper_threshold: float = 0.8,  # High similarity threshold
    sim_lower_threshold: float = 0.3,  # Low similarity threshold
    embedding_model: SentenceTransformer = None,
) -> pd.DataFrame:
    """
    Implements a three-level semantic intersection operator with progressive matching strategies:
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
    
    Returns:
        pd.DataFrame: DataFrame of rows from table1 that have matching rows in table2.
    """
    cascade_args = CascadeArgs()
    n_rows_t1 = len(table1)
    n_rows_t2 = len(table2)
    
    # Initialize the match matrix with -1 (unknown match status)
    # -1: not yet determined, 0: not a match, 1: a match
    match_matrix = np.full((n_rows_t1, n_rows_t2), -1, dtype=np.int8)
    
    # Step 1: Exact matching using sort-merge join approach for efficiency
    if show_progress_bar:
        print("Step 1: Running exact matching...")
    
    # Create concatenated strings for sorting for each table
    concat_values_t1 = []
    for idx, row in table1.iterrows():
        # Convert row values to strings and concatenate
        row_str = "|".join(str(row[col]) for col in columns1)
        concat_values_t1.append((idx, row_str))
    
    concat_values_t2 = []
    for idx, row in table2.iterrows():
        # Convert row values to strings and concatenate
        row_str = "|".join(str(row[col]) for col in columns2)
        concat_values_t2.append((idx, row_str))
    
    # Sort by the concatenated string
    sorted_t1 = sorted(concat_values_t1, key=lambda x: x[1])
    sorted_t2 = sorted(concat_values_t2, key=lambda x: x[1])
    
    # Use merge-join approach to find matching rows efficiently
    exact_match_count = 0
    exact_match_pairs = []
    
    i = 0
    j = 0
    
    # Merge-join on sorted arrays
    while i < len(sorted_t1) and j < len(sorted_t2):
        idx1, val1 = sorted_t1[i]
        idx2, val2 = sorted_t2[j]
        
        if val1 == val2:
            # Found a match
            match_matrix[idx1, idx2] = 1
            exact_match_count += 1
            exact_match_pairs.append((idx1, idx2))
            
            # Check for more matches with the same value in table2
            j_next = j + 1
            while j_next < len(sorted_t2) and sorted_t2[j_next][1] == val1:
                idx2_next = sorted_t2[j_next][0]
                match_matrix[idx1, idx2_next] = 1
                exact_match_count += 1
                exact_match_pairs.append((idx1, idx2_next))
                j_next += 1
            
            # Check for more matches with the same value in table1
            i_next = i + 1
            while i_next < len(sorted_t1) and sorted_t1[i_next][1] == val1:
                idx1_next = sorted_t1[i_next][0]
                # Match with all matching entries from table2
                j_check = j
                while j_check < len(sorted_t2) and sorted_t2[j_check][1] == val1:
                    idx2_check = sorted_t2[j_check][0]
                    match_matrix[idx1_next, idx2_check] = 1
                    exact_match_count += 1
                    exact_match_pairs.append((idx1_next, idx2_check))
                    j_check += 1
                i_next += 1
            
            # Move past all matches
            while i < len(sorted_t1) and sorted_t1[i][1] == val1:
                i += 1
            while j < len(sorted_t2) and sorted_t2[j][1] == val1:
                j += 1
        elif val1 < val2:
            i += 1
        else:
            j += 1
    
    if show_progress_bar:
        print(f"Layer 1 (Exact Match) - Found {exact_match_count} matching pairs")

    
    # Step 2: Embedding similarity matching
    if show_progress_bar:
        print("\nStep 2: Running embedding similarity matching...")
    
    # Create text representation for each row
    df_result1 = table1.apply(
        lambda row: " ".join([str(row[col]) for col in columns1 if pd.notna(row[col])]), 
        axis=1
    ).tolist()
    
    df_result2 = table2.apply(
        lambda row: " ".join([str(row[col]) for col in columns2 if pd.notna(row[col])]), 
        axis=1
    ).tolist()
    
    # Generate embeddings
    embeddings1 = embedding_model._embed(df_result1)
    embeddings2 = embedding_model._embed(df_result2)
    embeddings1 = normalize(embeddings1, norm='l2', axis=1)
    embeddings2 = normalize(embeddings2, norm='l2', axis=1)
    # Build HNSW index for efficient similarity search
    index = build_hnsw_index(embeddings2, space='l2', ef_construction=500, M=64)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_rows_t1, n_rows_t2))
    
    # Use the HNSW index to find similar rows in table2 for each row in table1
    labels, distances = index.knn_query(embeddings1, k=n_rows_t2)
    
    # Convert distances to similarities and populate the similarity matrix
    for i in range(n_rows_t1):
        for j_idx, j in enumerate(labels[i]):
            if j < n_rows_t2:  # Ensure index is valid
                similarity_matrix[i, j] = np.exp(-distances[i][j_idx]**2 / (2 * 0.5**2))
    
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
    for i in range(n_rows_t1):
        for j in range(n_rows_t2):
            if match_matrix[i, j] != -1:
                continue  # Skip already determined pairs
            
            similarity = similarity_matrix[i, j]
            
            if similarity >= sim_upper_threshold:
                match_matrix[i, j] = 1  # High similarity -> match
                high_sim_count += 1
                high_sim_pairs.append((i, j, similarity))
            elif similarity <= sim_lower_threshold:
                match_matrix[i, j] = 0  # Low similarity -> non-match
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
    
    for i in range(n_rows_t1):
        for j in range(n_rows_t2):
            if match_matrix[i, j] == -1:  # Only process undecided pairs
                row1 = table1.iloc[i]
                row2 = table2.iloc[j]
                
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
            if result:
                match_matrix[i, j] = 1
                llm_match_count += 1
                llm_match_pairs.append((i, j))
            else:
                match_matrix[i, j] = 0
                llm_non_match_count += 1
                
        if show_progress_bar and len(llm_match_pairs) > 0:
            print(f"Layer 3 (LLM) - Found {llm_match_count} matching pairs and {llm_non_match_count} non-matching pairs")
    
    if show_progress_bar:
        print("\nSummary Statistics:")
        total_comparisons = n_rows_t1 * n_rows_t2
        print(f"Total number of pairwise comparisons: {total_comparisons}")
        print(f"Layer 1 (Exact Match): {exact_match_count} matches ({exact_match_count / total_comparisons * 100:.2f}%)")
        print(f"Layer 2 (Embedding Similarity): {high_sim_count} matches, {low_sim_count} non-matches")
        print(f"Layer 3 (LLM): {llm_match_count} matches, {llm_non_match_count} non-matches")
    
    # Find rows from table1 that have at least one match in table2
    matching_rows_indices = set()
    for i in range(n_rows_t1):
        if any(match_matrix[i, j] == 1 for j in range(n_rows_t2)):
            matching_rows_indices.add(i)
    
    # Build final result DataFrame
    final_rows = []
    for idx in sorted(list(matching_rows_indices)):
        row = table1.iloc[idx].copy()
        final_rows.append(row)
    
    result_df = pd.DataFrame(final_rows)
    return result_df

@pd.api.extensions.register_dataframe_accessor("sem_intersect")
class SemIntersectDataFrame:
    """
    DataFrame accessor for sem_intersect operator.
    
    Enables using sem_intersect as a method on DataFrames.
    For example:
        result_df = df_left.sem_intersect(df_right, join_instruction="A:left, B:left, X:right, Y:right")
    
    The join_instruction string is parsed to determine which columns from the left and right tables are to be compared.
    """
    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("sem_intersect accessor can only be used with DataFrames.")

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
        Applies sem_intersect between this DataFrame and another DataFrame/Series.
        
        Args:
            other (pd.DataFrame | pd.Series): The other table to intersect with.
            join_instruction (str): A string specifying which columns to compare.
                For example, "A:left, B:left, X:right, Y:right" indicates that columns A and B in the left DataFrame
                are to be compared with columns X and Y in the right DataFrame.
            safe_mode (bool): Whether to show cost estimates.
            show_progress_bar (bool): Whether to display a progress bar.
            sim_upper_threshold (float): Similarity threshold above which rows are considered a match.
            sim_lower_threshold (float): Similarity threshold below which rows are considered not a match.
            embedding_model (SentenceTransformer): Model for creating embeddings.
        
        Returns:
            pd.DataFrame: A DataFrame containing rows from this DataFrame that have matching rows in the other DataFrame.
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
            raise ValueError("Both left and right columns must be specified in join_instruction for sem_intersect operator.")
        
        # Create a user instruction for row comparison.
        user_instruction = (
            f"Compare these two rows with the following columns and determine if they represent the same entity or information.\n\n"
            f"Row 1 columns: {left_columns}\n"
            f"Row 2 columns: {right_columns}\n\n"
        )
        result = sem_intersect(
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