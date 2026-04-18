from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-base")

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
Give a score out of 100 for this resume based on:
- Skills
- Experience
- Education

Also give explanation.

Resume:
{resume}
"""
)

score_chain = prompt | llm