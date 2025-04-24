LOTUS SEMANTIC SET OPERATIONS

This document provides an overview of the semantic union and intersection operations available in the Lotus library and their associated test files.

OPERATIONS

Both operations are implemented in the lotus/sem_ops directory:

- sem_union: Performs a semantic union operation, combining data based on semantic similarity.
  - Implementation: lotus/sem_ops/sem_union.py
  - Key Parameters:
    - table1, table2: The input DataFrames.
    - columns1, columns2: Lists of column names to compare in each table.
    - user_instruction: A natural language instruction guiding the LLM for ambiguous cases (e.g., "Determine if the products described in the two rows are the same").
    - embedding_model: The SentenceTransformer model used for semantic similarity comparison (defaults to a pre-configured or standard model).
    - sim_upper_threshold, sim_lower_threshold: Floats between 0 and 1 defining the cutoffs for definite matches/non-matches based on embedding similarity.
    - auto_threshold: Strategy ("None", "Valley", "Oracle") to automatically determine similarity thresholds.
    - examples_multimodal_data, examples_answers, cot_reasoning: Optional lists for providing few-shot examples to the LLM.

- sem_intersect: Performs a semantic intersection operation, finding common elements based on semantic similarity.
  - Implementation: lotus/sem_ops/sem_intersect.py
  - Key Parameters:
    - table1, table2: The input DataFrames.
    - columns1, columns2: Lists of column names to compare in each table.
    - user_instruction: A natural language instruction guiding the LLM for ambiguous cases (similar to sem_union).
    - embedding_model: The SentenceTransformer model used for semantic similarity comparison (defaults to a pre-configured or standard model).
    - sim_upper_threshold, sim_lower_threshold: Floats between 0 and 1 defining the cutoffs for definite matches/non-matches based on embedding similarity.
    - examples_multimodal_data, examples_answers, cot_reasoning: Optional lists for providing few-shot examples to the LLM.

Both operations are also registered as Pandas DataFrame accessors (.sem_union() and .sem_intersect()), which accept similar parameters, often parsing columns1 and columns2 from a join_instruction string (e.g., "A:left, B:left, X:right, Y:right").

TESTING

Test scripts for these operations are located in the root directory:

Semantic Intersection Tests:
- sem_intersect_test.py
- sem_intersect_test_dblp_acm.py

Semantic Union Tests:
- sem_union_final_test_a_g.py
- sem_union_final_test_dblp_acm.py
- sem_union_final_test_ebook.py

Install necessary packages by running: 
pip install -r INSTALL.txt
