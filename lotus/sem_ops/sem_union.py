import pandas as pd
from typing import List, Dict, Set, Any
from tqdm import tqdm

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticFilterOutput
from lotus.utils import show_safe_mode
from .postprocessors import filter_postprocess

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
    
    For each row in table1 and each row in table2, a document is created (containing the two rows’ values)
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
        user_instruction (str): Instruction for the LLM; e.g. “Do these rows match exactly? Answer True or False.”
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
            # Create a document that contains both row values.
            # You can adjust the structure as needed.
            if i == j:
                continue
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
    postprocess_output: SemanticFilterOutput = filter_postprocess(lm_output.outputs, default=default)
    outputs_bool: List[bool] = postprocess_output.outputs  # Expecting a list of booleans ("True" -> True, "False" -> False)
    print(outputs_bool)
    # Build match lists for table1 and table2.
    n_rows_t1 = table1.shape[0]
    n_rows_t2 = table2.shape[0]
    matches_t1: List[List[int]] = [[] for _ in range(n_rows_t1)]
    matches_t2: List[List[int]] = [[] for _ in range(n_rows_t2)]
    
    for idx, result in enumerate(outputs_bool):
        if result:
            i, j = mapping[idx]
            matches_t1[i].append(j)
            matches_t2[j].append(i)
    print(matches_t2)
    print(matches_t1)
    # Build a bipartite graph where nodes are ("t1", i) or ("t2", j) and edges represent matches.
    graph: Dict[tuple, Set[tuple]] = {}
    for i in range(n_rows_t1):
        node = ("t1", i)
        graph.setdefault(node, set())
        for j in matches_t1[i]:
            neighbor = ("t2", j)
            graph[node].add(neighbor)
            graph.setdefault(neighbor, set())
            graph[neighbor].add(node)
    
    # Use DFS to find connected components (i.e. groups of mutually matching rows).
    visited: Set[tuple] = set()
    groups: List[Set[tuple]] = []
    
    def dfs(node: tuple, group: Set[tuple]):
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
    # From each group, choose the representative row as the one with the smallest index.
    rep_rows = []
    for group in groups:
        rep = min(group, key=lambda node: node[1])
        rep_rows.append(rep)
    unique_indices = []
    seen = set()
    for tag, idx in rep_rows:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)
    
    # Build the final result dataframe using only the unique indices.
    final_rows = []
    for idx in unique_indices:
        row = table1.iloc[idx].copy()
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
            f"Compare the two rows and answer True if they match semantically on the specified columns "
            f"(left: {left_columns} vs. right: {right_columns}), otherwise answer False. Understand the meaning of each entry first, then only provide simple True or False answer, no explaination or code needed."
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
