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


def exact_match(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    columns1: List[str] = None,
    columns2: List[str] = None,
) -> np.ndarray:
    """
    Implements an efficient exact-match algorithm using sort-merge join approach.
    
    Args:
        table1 (pd.DataFrame): First table to compare
        table2 (pd.DataFrame): Second table to compare
        columns1 (List[str]): Columns from table1 to use for matching. If None, all columns are used.
        columns2 (List[str]): Columns from table2 to use for matching. If None, uses columns1.
    
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
    for col in columns2:
        if col not in table2.columns:
            raise ValueError(f"Column '{col}' not found in table2")
    
    # Stack the tables
    df_stacked = pd.concat([table1, table2], ignore_index=True)

    
    # Create concatenated strings for sorting
    if len(columns1) == len(columns2):
        # For each row, create a string by concatenating values from specified columns
        concat_values = []
        for idx, row in df_stacked.iterrows():
            # Determine which set of columns to use based on whether the row came from table1 or table2
            cols = columns1 if idx < len(table1) else columns2
            # Convert row values to strings and concatenate
            row_str = "|".join(str(row[col]) for col in cols)
            concat_values.append(row_str)
        
        df_stacked['_concat_key'] = concat_values
    else:
        # If column lengths don't match, we need a different approach
        # For table1 rows, use columns1
        for idx in range(len(table1)):
            row = df_stacked.iloc[idx]
            df_stacked.at[idx, '_concat_key'] = "|".join(str(row[col]) for col in columns1)
        
        # For table2 rows, use columns2
        for idx in range(len(table1), len(df_stacked)):
            row = df_stacked.iloc[idx]
            df_stacked.at[idx, '_concat_key'] = "|".join(str(row[col]) for col in columns2)
    
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
    
    return result_matrix


#----

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
) -> pd.DataFrame:
    """
    Implements the sem_union operator by comparing selected columns between two tables.
    
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
    print(lm_output.outputs)
    # Postprocess outputs using the same postprocessor as sem_filter.
    postprocess_output: SemanticFilterOutput = filter_postprocess(lm_output.outputs, default=False)
    outputs_bool: List[bool] = postprocess_output.outputs  # Expecting a list of booleans ("True" -> True, "False" -> False)
    print(outputs_bool)
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
    print(matches_t2)
    print(matches_t1)
    # matches_t2 = [[], [2, 440], [1, 441], [], [443], [], [445], [446], [9], [8, 33, 448], [449], [], [35, 36], [32], [], [], [], [18, 19, 456], [], [458], [50, 51, 52, 53, 54, 459], [50, 51, 52, 53, 54], [50, 51, 52, 53, 54, 461], [50, 51, 52, 53, 54, 462], [469], [30, 464], [30], [28, 29], [29], [28], [466], [], [], [], [], [], [], [476], [], [], [479], [480], [505], [73, 74, 75, 487], [73, 74, 75], [], [], [482, 486], [482, 486], [], [], [], [], [], [], [], [86, 87], [496], [497], [498], [], [491], [92, 93, 94], [92, 93, 94], [503], [504], [], [], [514], [508], [509], [510], [], [], [511], [], [], [516], [], [], [], [], [], [], [523], [], [525], [117, 118, 119, 120, 121, 122], [117, 118, 119, 120, 121, 122], [117, 118, 119, 120, 121, 122], [117, 118, 119, 120, 121, 122], [117, 118, 119, 120, 121, 122, 536], [530, 532, 533], [], [], [], [], [], [128, 129], [], [136, 539, 540], [539, 540], [451], [451], [135, 136, 451], [135, 136], [], [], [547], [139, 140], [], [141, 142, 143, 550], [141, 142, 143, 551], [551], [553], [554], [], [], [], [], [559], [151, 152], [], [562], [562], [], [], [], [], [], [], [162, 163, 570], [], [], [], [], [577], [], [579], [580], [172, 173, 581], [], [583], [584], [585], [587], [587], [588], [187], [186], [], [], [], [], [], [193, 194, 195, 196, 197], [193, 194, 195, 196, 197], [193, 194, 195, 196, 197, 598], [193, 194, 195, 196, 197], [], [601], [602], [602], [], [202, 203], [], [], [608], [], [], [], [612], [210, 211], [], [], [616], [], [], [], [620], [621], [622], [], [], [], [], [675], [628], [629, 630], [630], [631], [229], [526], [], [210, 211, 635], [635], [637], [235, 236], [639], [640], [], [], [643], [], [], [646], [], [], [], [], [248, 249, 250, 251], [248, 249, 250, 251], [248, 249, 250, 251, 653], [], [655], [656, 713], [657], [], [], [], [], [259, 260, 662], [], [], [], [], [], [], [266, 267, 268, 269, 270, 271, 673], [266, 267, 268, 269, 270, 271, 674], [266, 267, 268, 269, 270, 271, 676], [266, 267, 268, 269, 270, 271, 676, 677], [266, 267, 268, 269, 270, 271, 678, 787], [], [767], [690, 709, 767], [690, 709], [275, 276, 683], [684], [], [686], [], [], [281, 282, 689], [], [], [], [], [], [287, 288], [], [702], [290, 291, 292, 293, 294, 698, 702], [290, 291, 292, 293, 294], [290, 291, 292, 293, 294, 697, 701], [290, 291, 292, 293, 294, 701], [701], [], [], [297, 298, 705], [682, 704], [659], [300, 301, 302], [300, 301, 302], [], [303, 304, 641], [], [], [], [307, 308, 715], [716, 718], [309, 310, 311, 312, 717], [309, 310, 311, 312, 716, 718], [309, 310, 311, 312], [], [721], [722], [], [318, 724], [316, 725], [], [], [], [586], [731], [], [], [734], [326, 327, 328, 736], [326, 327, 328, 736], [737], [], [739], [740], [665], [665, 742], [335, 336, 743], [], [], [], [], [747, 748], [749], [], [], [344, 345], [], [], [], [], [757], [350, 351, 352, 353, 354, 355, 356], [350, 351, 352, 353, 354, 355, 356, 759], [350, 351, 352, 353, 354, 355, 356], [350, 351, 352, 353, 354, 355, 356, 761], [350, 351, 352, 353, 354, 355, 356], [350, 351, 352, 353, 354, 355, 356, 763], [], [765], [766], [], [121], [], [], [122], [], [], [774], [367, 368, 369, 370, 371], [367, 368, 369, 370, 371], [367, 368, 369, 370, 371], [367, 368, 369, 370, 371], [], [], [], [], [], [], [], [783], [680], [380, 381, 382], [380, 381, 382], [790], [425], [384, 385, 792], [793], [794], [], [], [], [], [799], [], [], [], [703], [], [], [], [], [], [473], [436], [], [], [], [], [409], [407], [], [], [], [], [], [], [], [], [], [], [], [727], [], [], [], [], [383], [596], [], [], [703], [], [735], [], [], [], [402], [], [597, 710], [], [521], [], [], [3], [], [], [7], [668], [9], [], [40], [34, 41, 517], [12], [13, 43], [], [15, 45, 542], [46], [], [48], [20, 49], [21], [22, 141, 142, 143], [23], [], [], [], [56], [], [58], [59], [], [61], [], [], [64, 401, 613], [687], [69], [], [], [63], [], [], [], [], [], [], [76], [], [], [79], [80], [], [86, 87], [], [], [], [], [82], [], [688], [], [100], [], [], [], [], [96], [97], [98], [], [], [], [], [103], [104, 735], [81, 708], [], [], [108, 450], [109, 595], [110, 597], [113, 541, 675], [439], [], [107], [631], [116], [], [231], [], [], [532, 533], [530], [123, 530, 532, 533], [], [125], [], [694, 712], [], [], [131, 539, 540, 595], [136, 520], [131], [131, 454], [], [134, 135, 136], [130], [], [138], [139, 140], [139, 140], [], [], [], [], [], [], [147], [], [], [150], [151, 152], [], [153, 154], [], [], [], [], [159], [], [], [162, 163], [578], [], [], [], [], [], [573], [], [724], [], [756], [], [], [], [179], [180], [324], [184, 186, 187, 727], [183, 188, 735], [708], [], [], [], [], [520, 541], [428], [439, 521], [], [198], [], [], [201], [202, 203], [], [], [205], [], [], [208], [], [], [], [212, 475], [], [], [], [], [], [], [], [220], [], [222], [], [675], [], [629, 630], [], [228], [], [229, 230, 526], [231], [], [645, 682, 704], [], [235, 236], [], [], [], [239], [240, 796], [306, 636], [], [243, 706], [], [], [246], [], [], [], [], [697, 701], [], [253, 798], [], [255], [256, 736], [257], [728], [300, 301], [751], [], [], [], [335, 336], [], [], [452], [], [], [], [783], [631], [273], [274, 526, 704], [231], [275, 276], [277], [278, 641], [386], [], [767], [306, 709, 780], [770, 795], [284], [711], [286, 481, 712], [505], [], [289, 701, 746], [281, 282, 702], [], [697], [543, 659, 698, 746], [], [682], [81], [300, 301], [297, 298, 659], [735], [690, 767], [300, 301, 302, 641], [297, 402, 436, 693], [651, 694], [306], [305], [], [], [718], [281, 282, 444, 716], [], [543], [], [], [], [], [316, 318], [317, 750], [316, 318], [], [], [321, 586, 667], [322], [], [324, 596], [325], [428, 708], [], [], [], [331], [], [], [334, 665], [], [335, 336], [337], [702], [339, 698, 747, 748], [340], [726], [342], [343], [], [669], [], [297, 298, 348, 782], [348], [349], [], [], [], [], [589], [], [], [357], [], [359], [690, 709], [], [795], [363, 692, 775], [], [365], [366, 771], [], [281, 282], [], [], [], [], [691, 755], [374], [], [680], [], [378], [], [], [], [], [386], [], [384, 385], [770], [], [279, 692], [], [390], [], [392], [393], [649], [], [662], [], [], [262, 399], [], [], [], [402], [], [], [], [], [407], [407], [], [410], [411], [412], [], [], [], [], [], [], [419], [420], [421], [422], [], [], [], [], [], [], [], [430], [], [], [433], [], [], [], [437]]
    # matches_t1 = [[], [2], [1], [403], [], [], [], [406], [9], [8, 408], [], [], [412], [413], [], [415], [], [], [17], [17], [419], [420], [421], [422], [], [], [], [], [27, 29], [27, 28], [25, 26], [], [13], [9], [411], [12], [12], [], [], [], [410], [411], [], [413], [], [415], [416], [], [418], [419], [20, 21, 22, 23], [20, 21, 22, 23], [20, 21, 22, 23], [20, 21, 22, 23], [20, 21, 22, 23], [], [426], [], [428], [429], [], [431], [], [439], [434], [], [], [], [], [436], [], [], [], [43, 44], [43, 44], [43, 44], [446], [], [], [449], [450], [475, 658], [457], [], [], [], [56, 452], [56, 452], [], [], [], [], [62, 63], [62, 63], [62, 63], [], [466], [467], [468], [], [461], [], [], [473], [474], [], [], [484], [478], [479], [480], [], [], [481], [], [], [486], [87, 88, 89, 90, 91], [87, 88, 89, 90, 91], [87, 88, 89, 90, 91], [87, 88, 89, 90, 91], [87, 88, 89, 90, 91, 321], [87, 88, 89, 90, 91, 324], [493], [], [495], [], [], [98], [98], [506], [500, 502, 503], [], [], [505], [104, 105, 505], [100, 104, 105, 501, 505], [], [508], [109, 509, 510], [109, 509, 510], [111, 112, 421], [111, 112, 421], [111, 112, 421], [], [], [], [517], [], [], [520], [121, 521], [121, 521], [523], [523], [], [], [], [], [528], [], [], [131, 531], [131, 531], [], [], [], [], [], [], [], [], [140], [140], [], [], [], [], [], [546], [547], [], [], [550], [549], [], [149, 549], [148, 549], [550], [], [], [], [], [155, 156, 157, 158], [155, 156, 157, 158], [155, 156, 157, 158], [155, 156, 157, 158], [155, 156, 157, 158], [560], [], [], [563], [164, 564], [164, 564], [], [567], [], [], [570], [], [172, 194], [172, 194], [574], [], [], [], [], [], [], [], [582], [], [584], [], [], [], [], [], [590], [191, 592], [592], [488, 593, 637], [], [], [], [197, 597], [197, 597], [], [], [601], [602], [], [], [605], [], [], [608], [], [210, 211, 212], [210, 211, 212], [210, 211, 212], [210, 211, 212], [], [615], [], [617], [618], [619], [], [221], [221], [], [760], [], [], [], [228, 229, 230, 231, 232], [228, 229, 230, 231, 232], [228, 229, 230, 231, 232], [228, 229, 230, 231, 232], [228, 229, 230, 231, 232], [228, 229, 230, 231, 232], [], [635], [636], [237, 638], [237, 638], [639], [640], [749], [], [243, 652, 671, 729], [243, 652, 671, 729], [], [646], [], [648], [249], [249], [651], [252, 253, 254, 255], [252, 253, 254, 255], [252, 253, 254, 255], [252, 253, 254, 255], [252, 253, 254, 255], [], [], [259, 660, 664, 708], [259, 660, 708], [], [262, 263, 621, 659, 663], [262, 263, 621, 659, 663], [262, 263, 663], [265], [265], [667], [603, 644, 666], [269], [269], [271, 272, 273], [271, 272, 273], [271, 272, 273], [271, 272, 273], [], [], [], [279, 678, 680], [679], [278, 678, 680], [], [], [683], [684], [], [548, 686], [687], [288, 289], [288, 289], [288, 289], [], [], [692], [], [], [695], [296, 626, 697], [296, 626, 697], [698], [], [700], [701], [], [703], [704], [305], [305], [], [], [708, 709], [710], [311, 312, 313, 314, 315, 316], [311, 312, 313, 314, 315, 316], [311, 312, 313, 314, 315, 316], [311, 312, 313, 314, 315, 316], [311, 312, 313, 314, 315, 316], [311, 312, 313, 314, 315, 316], [311, 312, 313, 314, 315, 316], [718], [], [720], [], [], [], [724], [], [726], [727], [328, 329, 330, 331], [328, 329, 330, 331], [328, 329, 330, 331], [328, 329, 330, 331], [328, 329, 330, 331], [], [], [735], [], [], [], [739], [], [341, 342], [341, 342], [341, 342], [386], [345, 746], [345, 746], [641, 744], [], [], [], [751], [], [753], [754], [], [], [], [], [], [760], [], [434], [396, 664, 764], [], [], [], [], [369, 769, 770], [], [368], [772], [773], [774], [], [], [], [], [], [], [781], [782], [783], [784], [], [], [344], [], [], [557, 688], [], [792], [], [], [795], [], [], [363, 664], [799], [], [482, 558], [1], [2], [], [4], [671], [6], [7], [], [9], [10], [478], [102, 103, 104], [629], [], [503], [], [17], [], [19], [20], [], [22], [23], [], [25], [], [30], [], [], [24], [], [], [], [362], [], [574], [37], [], [], [40], [41], [648], [47, 48], [], [], [], [47, 48], [43], [], [], [], [61], [], [], [], [], [57], [58], [59], [], [], [], [], [64], [65], [42, 649], [], [], [69], [70], [71], [74], [], [], [68], [], [77], [411], [], [], [501, 556], [400, 558], [], [84], [], [86], [192, 592, 636], [], [], [], [92, 492, 493], [], [92, 491, 493], [92, 491, 493], [], [], [91], [], [], [100, 101, 500], [100, 101, 500], [481, 556], [415], [655, 673], [], [], [], [108], [], [], [111], [112, 113], [], [114], [115], [], [], [], [], [120], [], [], [123, 124], [], [], [], [], [], [], [], [131], [], [], [538], [], [], [], [136], [532], [138], [139], [140], [], [142], [143], [144], [283, 683], [145, 146], [147], [715], [], [], [], [], [], [479, 500], [387, 686], [398, 480], [157], [], [], [160], [161, 162], [], [], [], [], [], [167], [], [], [], [171], [434], [], [], [175], [], [], [], [179], [180], [181], [], [], [], [], [], [187], [188, 588], [188, 189, 588], [190, 485, 634], [], [], [], [194, 195], [603], [196], [], [198], [199], [265, 640, 663], [], [202], [], [595], [205], [], [], [755], [], [665], [], [212], [], [214], [215], [216], [], [261, 655, 660], [], [], [221, 757], [], [], [294, 295, 695], [], [683], [407], [706], [], [], [], [228], [229], [186, 481, 586], [230, 231], [231], [232], [], [340, 737], [], [260, 595, 657], [237], [238], [], [240], [435], [459], [243], [235, 236, 662, 721], [734], [724, 749], [664], [497, 665], [], [], [254, 613, 654], [252, 655, 700], [], [], [254, 255, 256, 613, 651], [251, 252, 652, 699], [356, 390], [260, 595, 636], [259], [605], [], [475, 551, 688], [235, 236, 644, 721], [398], [647], [497, 648], [215], [], [269], [270, 272, 671], [271], [270, 272, 670], [], [], [275], [276], [], [278, 540], [279], [702], [381, 549], [620], [], [], [284], [], [], [287], [392, 474, 550, 661], [288, 289, 618], [290], [], [292], [293], [], [295], [296], [], [], [651, 655], [301, 700], [301, 700], [302], [679], [622], [], [], [], [734], [542], [310], [], [312], [], [314], [], [316], [], [318], [319], [234, 235, 643, 662], [], [], [645, 747], [727], [], [], [327], [724], [], [], [], [], [644], [], [708], [339, 633], [], [], [], [232], [], [], [343], [], [345], [346], [347], [645, 723], [602], [], [615], [352]]

    matches = []

    # Assuming both lists have the same length (n_rows_t1)
    for i in range(len(matches_t1)):
        # Get elements from each list at current index
        elements1 = matches_t1[i]
        elements2 = matches_t2[i]
        
        # Combine elements while removing duplicates but preserving order
        combined = []
        
        # Add elements from matches_t1
        for element in elements1:
            if element not in combined:
                combined.append(element)
                
        # Add elements from matches_t2
        for element in elements2:
            if element not in combined:
                combined.append(element)
        
        # Add the combined elements to our result
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
        show_progress_bar: bool = True
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
            f"Compare these two rows and determine if they represent the same entity or information.\n\n"
            f"Row 1 columns: {left_columns}\n"
            f"Row 2 columns: {right_columns}\n\n"
            f"Consider semantic meaning, not just exact string matches. For example, 'NY' and 'New York' represent the same state.\n"
            f"Focus on identifying attributes that establish identity, even if formatting or minor details differ.\n"
            f"If the rows likely represent the same underlying entity or information, answer True. Otherwise, answer False. Do not provide any explanation or code."
        )
        result = sem_union(
            table1=self._obj,
            table2=other,
            columns1=left_columns,
            columns2=right_columns,
            user_instruction=user_instruction,
            safe_mode=safe_mode,
            show_progress_bar=show_progress_bar,
        )
        return result
