from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ✅ LOCAL MODEL (NO API NEEDED)
pipe = pipeline("text2text-generation", model="google/flan-t5-base")

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
Extract:
- Skills
- Experience
- Education

Resume:
{resume}
"""
)

extract_chain = prompt | llm