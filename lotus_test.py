import pandas as pd

import lotus
from lotus.models import SentenceTransformersRM, LM


# Configure models for LOTUS
#lm = LM(model="gpt-4o-mini", max_batch_size = 10)
lm = LM(model="deepseek/deepseek-chat")
#lm = LM(model="ollama/llama3.2")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

lotus.settings.configure(lm=lm, rm=rm)

# Dataset containing courses and their descriptions/workloads
data = [
    (
        "Probability and Random Processes",
        "Focuses on markov chains and convergence of random processes. The workload is pretty high.",
    ),
    (
        "Deep Learning",
        "Covers both the theory and practical implementation of neural networks. The workload depends on the professor but is usually reasonable.",
    ),
    (
        "Digital Design and Integrated Circuits",
        "Focuses on building RISC-V CPUs in Verilog. Students have said that the workload is VERY high.",
    ),
]
df = pd.DataFrame(data, columns=["Course Name", "Description"])

data2 = [
    (
        "Deep Learning",
        "Fouces on theory and implementation of neural networks. Workload varies by professor but typically isn't terrible.",
    ),
    (
        "Digital Design and Integrated Circuits",
        "Focuses on building RISC-V CPUs in Verilog. Students have said that the workload is VERY high.",
    ),
    (
        "Databases",
        "Focuses on implementation of a RDBMS with NoSQL topics at the end. Most students say the workload is not too high.",
    ),
]
df2 = pd.DataFrame(data2, columns=["Course Name", "Description"])
# # Applies semantic filter followed by semantic aggregation
# ml_df = df.sem_filter("{Description} indicates that the class is relevant for machine learning.")
# print(ml_df)
# tips = ml_df.sem_agg(
#     "Given each {Course Name} and its {Description}, give me a study plan to succeed in my classes."
# )._output[0]
# print(tips)
# top_2_hardest = df.sem_topk("What {Description} indicates the highest workload?", K=2)
# print(top_2_hardest)


# skills_df = pd.DataFrame(
#     [("SQL"), ("Chip Design")], columns=["Skill"]
# )
# classes_for_skills = skills_df.sem_join(
#     df, "Taking {Course Name} will make me better at {Skill}"
# )
# print(classes_for_skills)


# Use the sem_union accessor.
# The join_instruction "A:left, B:left, X:right, Y:right" tells the operator
# to compare columns A and B from df_left with columns X and Y from df_right.
result_df = df.sem_union(
    df2,
    join_instruction="{Course Name:left}, {Description:left}, {Course Name:right}, {Description:right}",
    safe_mode=False
)
print(result_df)