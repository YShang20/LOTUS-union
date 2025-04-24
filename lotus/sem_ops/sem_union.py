import pandas as pd
import numpy as np
from typing import List, Dict, Set, Any
from tqdm import tqdm
import sys
import scipy

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
from typing import Tuple

from .cascade_utils import importance_sampling, learn_cascade_thresholds


from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize

def reps_from_graph(graph, df):
    visited, reps = set(), []
    def dfs(v, comp):
        st = [v]
        while st:
            cur = st.pop()
            if cur in visited: continue
            visited.add(cur); comp.add(cur); st.extend(graph[cur])
    for node in range(len(df)):
        if node not in visited:
            comp=set(); dfs(node, comp); reps.append(min(comp))
    return df.iloc[sorted(reps)].reset_index(drop=True)

def build_hnsw_index_dense(emb: np.ndarray,
                           space: str = "l2",
                           ef_construction: int = 200,
                           M: int = 16,
                           ef_query: int = 50) -> hnswlib.Index:
    """Return a ready‑to‑query HNSW index."""
    dim = emb.shape[1]
    idx = hnswlib.Index(space=space, dim=dim)
    idx.init_index(max_elements=len(emb), ef_construction=ef_construction, M=M)
    idx.add_items(emb, np.arange(len(emb)))
    idx.set_ef(ef_query)
    return idx


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

    if minima.size == 0 :            # unimodal → use defaults / quantiles
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
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=200)
    X_svd = svd.fit_transform(X_tfidf)
    return X_svd.astype("float32")



def build_similarity_map(emb: np.ndarray,
                      k: int = 5_000,
                      space: str = "l2") -> Dict[int, List[Tuple[int, float]]]:
    """
    Return a dict {row_id: [(nbr_id, distance), …]}
    containing *k* neighbours per row (self‑match removed).
    """
    idx = build_hnsw_index_dense(emb, space=space, ef_construction= 500, M = 64)
    #labels, dists = idx.knn_query(emb, k=min(k, len(emb)))
    labels, dists = idx.knn_query(emb, k)
    sim_map = defaultdict(list)

    for i in range(len(labels)):
        for nbr, dist in zip(labels[i], dists[i]):
            if nbr == i:          # skip self
                continue
            sim_map[i].append((nbr, np.exp(-dist**2 / (2 * 0.5 **2))))
            
    # norms = np.linalg.norm(emb, axis = 1, keepdims = True)
    # normalized_emb = emb / norms

    return sim_map



def learn_union_thresholds_dense(
        neighbor_map: Dict[int, List[Tuple[int, float]]],
        combined_df,
        columns1,
        columns2,
        user_instr: str,
        cascade_args: CascadeArgs,
        *,
        examples_mm=None,
        examples_ans=None,
        cot=None,
        strategy=None,
        default=True,
) -> Tuple[float, float]:
    """
    Learn (tau_pos, tau_neg) from a sparse neighbor map.

    Parameters
    ----------
    neighbor_map : Dict[int, List[Tuple[int, float]]]
        Output of top‑k HNSW query:  for each row i ⇒ [(j, distance), …].
    sim_floor : float
        Ignore pairs with similarity < sim_floor (saves LLM budget).
    dist_to_sim : Callable
        Converts HNSW distance to similarity (default: 1 – d).
    """

    # --------- flatten the map to (pair, score) lists -------------
    pairs, scores = [], []
    for i, lst in neighbor_map.items():
        for j, sim in lst:
            if i < j:                                  # keep unique direction
                
                if sim >= 0.2:                   # pre‑filter
                    pairs.append((i, j))
                    scores.append(sim)

    if not pairs:
        raise ValueError("No pairs above sim_floor; cannot learn thresholds.")

    # --------- importance‑sample the sparse list ------------------
    samp_idx, corr = importance_sampling(scores, cascade_args)
    samp_pairs     = [pairs[k]  for k in samp_idx]
    samp_scores    = [scores[k] for k in samp_idx]
    samp_corr      = corr[samp_idx]

    # --------- build LLM prompts only for sampled pairs -----------
    docs = []
    for (i, j) in samp_pairs:
        r1 = combined_df.iloc[i][columns1].tolist()
        r2 = combined_df.iloc[j][columns2].tolist()
        docs.append({"text": f"Row1: {r1} | Row2: {r2}"})

    print('2nd layer LLM calls:', len(docs))
    prompts = [
        task_instructions.union_formatter(
            d, user_instr, examples_mm,
            ["True" if x else "False" for x in examples_ans] if examples_ans else None,
            cot, strategy
        )
        for d in docs
    ]

    out = lotus.settings.lm(
        prompts,
        show_progress_bar=True,
        progress_bar_desc="Oracle for threshold learning"
    )
    oracle_bool = filter_postprocess(out.outputs, default=default).outputs

    # --------- learn the two thresholds ---------------------------
    (tau_pos, tau_neg), _ = learn_cascade_thresholds(
        proxy_scores=samp_scores,
        oracle_outputs=oracle_bool,
        sample_correction_factors=samp_corr,
        cascade_args=cascade_args
    )
    return tau_pos, tau_neg



# ────────────────────────────────────────────────────────────────
# utilities_sparse.py  (put next to sem_union.py)
# ────────────────────────────────────────────────────────────────
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
def exact_match_sparse(
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
    match_map: Dict[int, List[int]] = defaultdict(list)
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
            match_map[orig_i].append(orig_j)
            #match_map[orig_j].append(orig_i)
            j += 1
        i += 1
    
    # Ensure diagonal is 0 (no self-matches)
    
    
    # Delete the temporary concat key column
    if '_concat_key' in df_stacked.columns:
        df_stacked.drop('_concat_key', axis=1, inplace=True)
    
    return match_map
# ░░░ 1.  Exact‑match as sparse map ░░░
# def exact_match_sparse(
#         table1: pd.DataFrame,
#         table2: pd.DataFrame | None = None,
#         columns1: List[str] | None = None,
#         columns2: List[str] | None = None,
#         is_stacked: bool = False
# ) -> Dict[int, List[int]]:
#     """
#     Return  {row_id : [exact‑match row_ids]}  without allocating an n×n matrix.
#     """
#     if columns1 is None:
#         columns1 = list(table1.columns)
#     if columns2 is None:
#         columns2 = columns1.copy()

#     # ── stack once ───────────────────────────────────────────────
#     if not is_stacked:
#         df_stacked = pd.concat([table1, table2], ignore_index=True)
#         n_rows_t1  = len(table1)
#     else:
#         df_stacked = table1
#         n_rows_t1  = len(table1) // 2

#     # ── build concatenated key for every row ─────────────────────
#     def make_key(idx, row):
#         cols = columns1 if idx < n_rows_t1 else columns2
#         return "|".join(str(row[c]) for c in cols)

#     keys = df_stacked.apply(lambda r: make_key(r.name, r), axis=1)
#     df_stacked["_k"] = keys

#     # ── group identical keys → exact matches ─────────────────────
#     groups = df_stacked.groupby("_k").indices
#     match_map: Dict[int, List[int]] = defaultdict(list)

#     for idx_list in groups.values():
#         idx_list = list(idx_list)
#         if len(idx_list) < 2:
#             continue
#         for i in idx_list:
#             for j in idx_list:
#                 if i != j:
#                     match_map[i].append(j)

#     df_stacked.drop(columns=["_k"], inplace=True)
#     return match_map           # Dict[int, List[int]]



def llm_union_sparse(
        neighbor_map:  Dict[int, List[tuple[int, float]]],
        combined: List[str],
        user_instruction: str,
        *,
        exact_match_map: Dict[int, List[int]],
        default: bool = True,
        examples_multimodal_data: list[dict[str, Any]] | None = None,
        examples_answers: List[bool] | None = None,
        cot_reasoning: List[str] | None = None,
        strategy: str | None = None,
        safe_mode: bool = False,
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Union comparisons",
        additional_cot_instructions: str = ""
) -> pd.DataFrame:
    """
    Same logic as before, but all pair bookkeeping is sparse:
      * exact matches already in `exact_match_map`
      * undecided pairs queried with the LLM
    """
   

    # ── 1.  build adjacency from exact matches ───────────────────
    graph: Dict[int, Set[int]] = defaultdict(set)
    for i, lst in exact_match_map.items():
        for j in lst:
            graph[i].add(j)
            graph[j].add(i)

    # ── 2.  decide remaining pairs with the LLM  (only i<j) ──────
    docs, mapping = [], []
    for i in range(len(neighbor_map)):
        for j , sim_score in neighbor_map[i]:
            if j in graph[i]:              # already matched
                continue
            row1 = combined[i]
            row2 = combined[j]
            docs.append({"text": f"Row1: {row1} | Row2: {row2}"})
            mapping.append((i, j))

    if docs:
        ex_answer_strs = ["True" if a else "False" for a in examples_answers] if examples_answers else None
        prompts = [
            task_instructions.union_formatter(
                d, user_instruction,
                examples_multimodal_data,
                ex_answer_strs,
                cot_reasoning,
                strategy,
                reasoning_instructions=additional_cot_instructions,
            )
            for d in docs
        ]

        lm_out: LMOutput = lotus.settings.lm(
            prompts,
            show_progress_bar=show_progress_bar,
            progress_bar_desc=progress_bar_desc,
            logprobs=False
        )
        llm_bool: List[bool] = filter_postprocess(lm_out.outputs,
                                                  default=default).outputs

        # ── add positive matches to graph ────────────────────────
        for (i, j), is_match in zip(mapping, llm_bool):
            if is_match:
                graph[i].add(j)
                graph[j].add(i)

    # ── 3.  connected‑components → representative rows ───────────
    visited, groups = set(), []

    def dfs(v, acc):
        stack = [v]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            acc.add(cur)
            stack.extend(graph[cur])

    for node in range(n):
        if node not in visited:
            comp = set()
            dfs(node, comp)
            groups.append(comp)

    reps = sorted(min(g) for g in groups)
    return combined.iloc[reps].reset_index(drop=True)

def sem_union(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    columns1: List[str],
    columns2: List[str],
    user_instruction: str,
    *,
    k_neighbors: int = 50,
    run_llm:     bool  = True,
    auto_threshold = "Default",
    default:     bool  = True,
    examples_multimodal_data=None,
    examples_answers=None,
    cot_reasoning=None,
    strategy=None,
    safe_mode=False,
    show_progress_bar=True,
    progress_bar_desc="Union comparisons",
    additional_cot_instructions="",
    embedding_model: SentenceTransformer | None = None,
    cascade_args: CascadeArgs = CascadeArgs(),
) -> pd.DataFrame:

    # 0 ────────── stack once ────────────────────────────────────
    combined = pd.concat([table1, table2], ignore_index=True)
    n_rows   = len(combined)
    if show_progress_bar:
        print(f"[∪] Total stacked rows: {n_rows}")
        print(f"[∪] Combined table memory usage: {combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 1 ────────── exact‑match (sparse) ──────────────────────────
    if show_progress_bar:
        print("[∪] Step 1 – exact string match …")
    exact_map = exact_match_sparse(
        table1 = combined,
        table2 = None,
        columns1=columns1,
        columns2=columns2,
        is_stacked=True
    )
    exact_pairs = sum(len(v) for v in exact_map.values()) 
    if show_progress_bar:
        print(f"    ↳ {exact_pairs:,} exact‑match pairs")
        print(f"    ↳ Exact match map size: {len(exact_map)} keys")

    # 2A ─────────── embeddings + neighbor map ───────────────────
    if embedding_model is None:
        try:
            embedding_model = lotus.settings.retrieval_model
        except AttributeError:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if show_progress_bar:
        print("[∪] Step 2 – building neighbor map …")

    texts = combined[columns1].astype(str).agg(" | ".join, axis=1).tolist()
    print(f"    ↳ Total text entries: {len(texts)}")
    combined_compressed = combined.apply(
    lambda row: " ".join([str(row[col]) for col in columns1 if pd.notna(row[col])]),
    axis=1).tolist()
    #emb = compute_embedding(combined)
    emb = embedding_model._embed(combined_compressed)
    emb = normalize(emb, norm = 'l2', axis = 1)
    print(f"    ↳ Embeddings shape: {emb.shape}, approx memory: {emb.data.nbytes / 1024**2:.2f} MB")

    neighbor_map = build_similarity_map(emb, k=k_neighbors)
    neighbor_counts = [len(v) for v in neighbor_map.values()]
    print(f"    ↳ Neighbor map built: {len(neighbor_map)} entries")
    print(f"    ↳ Avg neighbors per row: {np.mean(neighbor_counts):.2f}, max: {np.max(neighbor_counts)}")

    # 2B ─────────── learn thresholds ─────────────────────────────
    if auto_threshold == "Default":
        tau_pos = 0.8
        tau_neg = 0.3

    if auto_threshold == "Valley":
            # Extract upper triangle of similarity matrix (avoiding diagonal)
            similarity_scores = []
            for i in range(len(neighbor_map)):
                for j, sim_score in  neighbor_map[i]:  # Skip if already matched exactly
                    if j not in exact_map[i]  and j > i and sim_score > 0.1:
                        print("record ", i, " neighbor", j, "score:", sim_score)
                        similarity_scores.append(sim_score)
                        
            if len(similarity_scores) > 10:
                if show_progress_bar:
                    print("Automatically determining similarity thresholds based on score distribution...")
                
                lower_threshold, upper_threshold = detect_valley(
                    similarity_scores, 
                    default_lower=0.3,
                    default_upper=0.8
                )
                
                if show_progress_bar:
                    print(f"Auto-detected thresholds - Lower: {lower_threshold:.4f}, Upper: {upper_threshold:.4f}")
                
                # Update the thresholds
                tau_neg = lower_threshold
                tau_pos = upper_threshold
    
    elif auto_threshold == "Oracle":
        # this is how you would call it when only limiting k = 50 closest neighbors to save LLM calls
        #sim_upper_threshold, sim_lower_threshold = learn_union_thresholds(dense_scores, dense_pairs, combined_table, columns1, columns2, user_instruction, cascade_args)
        
        tau_pos, tau_neg = learn_union_thresholds_dense(
            neighbor_map,
            combined,
            columns1, columns2,
            user_instruction,
            cascade_args,
            examples_mm   = examples_multimodal_data,
            examples_ans  = examples_answers,
            cot           = cot_reasoning,
            strategy      = strategy,
            default       = default,
        )
        if show_progress_bar:
            print(f"[∪] Learned thresholds: τ⁺ = {tau_pos:.4f}, τ⁻ = {tau_neg:.4f}")



    # tau_pos = 0.7
    # tau_neg = 0.3

    # 2C ─────────── Classify neighbor edges ─────────────────────
    if show_progress_bar:
        print("[∪] Classifying high / low similarity edges …")

    graph: Dict[int, Set[int]] = defaultdict(set)
    for i, lst in exact_map.items():
        for j in lst:
            graph[i].add(j)
            graph[j].add(i)

    undecided: List[Tuple[int, int]] = []
    hi_cnt = lo_cnt = 0

    for i, lst in neighbor_map.items():
        for j, sim in lst:
            if i >= j:
                continue
            if sim >= tau_pos:
                graph[i].add(j)
                graph[j].add(i)
                hi_cnt += 1
            elif sim <= tau_neg:
                lo_cnt += 1
            else:
                undecided.append((i, j))

    if show_progress_bar:
        print(f"    ↳ High sim: {hi_cnt}, Low sim: {lo_cnt}, Undecided: {len(undecided)}")
        print(f"    ↳ Graph now has {len(graph)} nodes")

    # 3 ─────────── LLM Step ─────────────────────────────────────
    if run_llm and undecided:
        if show_progress_bar:
            print("[∪] Step 3 – LLM on undecided pairs …")

        result_df = llm_union_sparse(
            neighbor_map=neighbor_map,
            combined= combined_compressed,
            user_instruction=user_instruction,
            exact_match_map=graph,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            safe_mode=safe_mode,
            show_progress_bar=show_progress_bar,
            progress_bar_desc=progress_bar_desc,
            additional_cot_instructions=additional_cot_instructions,
        )
        return result_df

    elif not run_llm:
        print("[∪] Skipping LLM Layer – using graph from layer 2")

    # 4 ─────────── Connected Components ─────────────────────────
    print("[∪] Running DFS to extract connected components …")
    result_df = reps_from_graph(graph, combined)
    print(f"[∪] Final representative rows: {len(result_df)}")

    return result_df
