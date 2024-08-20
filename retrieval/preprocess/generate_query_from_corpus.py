import json

import pandas as pd

from llama_index.core.schema import TextNode
from llama_index.core.evaluation import generate_question_context_pairs
import random



# from llama_index.legacy.llms import HuggingFaceLLM
# locally_run = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha")

random.seed(42)

import os
os.environ["DASHSCOPE_API_KEY"] = "sk-6d2fdcd737f04dab91e38fbf5e3369c6"
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_TURBO)

# from llama_index.llms.openai import OpenAI
# open_llm = OpenAI(model="models/Qwen2-0.5B-Instruct", max_retries=5,api_key="EMPTY", api_base="http://localhost:8000/v1")

from llama_index.llms.huggingface import HuggingFaceLLM
locally_run = HuggingFaceLLM(model_name="/mnt/d/PycharmProjects/models/Qwen1.5-14B-Chat-GPTQ-Int4",
                             tokenizer_name="/mnt/d/PycharmProjects/models/Qwen1.5-14B-Chat-GPTQ-Int4")



# Prompt to generate questions
qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a university professor. Your task is to set {num_questions_per_chunk} questions for the upcoming Chinese quiz.
Questions throughout the test should be diverse. Questions should not contain options or start with Q1/Q2.
Questions must be written in Chinese. The expression must be concise and clear. 
It should not exceed 15 Chinese characters. Words such as "这", "那", "根据", "依据" and other punctuation marks 
should not be used. Abbreviations may be used for titles and professional terms.
"""

nodes = []
data_df = pd.read_csv("../data/doc_qa_dataset.csv", encoding="utf-8")
for i, row in data_df.iterrows():
    if len(row["content"]) > 220 and i > 96:
        node = TextNode(text=row["content"])
        node.id_ = f"node_{i + 1}"
        nodes.append(node)

nodes = nodes[:10]

doc_qa_dataset = generate_question_context_pairs(
    nodes, llm=locally_run, num_questions_per_chunk=2, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)
# doc_qa_dataset.save_json("../data/doc_qa_dataset_demo.json")
with open("../data/doc_qa_dataset_demo_qwen1-5_14b.json", "w") as f:
    json.dump(doc_qa_dataset.dict(), f, indent=4, ensure_ascii=False)


doc_qa_dataset = generate_question_context_pairs(
    nodes, llm=dashscope_llm, num_questions_per_chunk=2, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)
with open("../data/doc_qa_dataset_demo_qwen_turbo.json", "w") as f:
    json.dump(doc_qa_dataset.dict(), f, indent=4, ensure_ascii=False)