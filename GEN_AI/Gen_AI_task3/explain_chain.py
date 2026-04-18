from langchain_huggingface import HuggingFaceEndpoint
from prompts.explain_prompt import explain_prompt

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    max_new_tokens=512,
    huggingfacehub_api_token="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"
)

explain_chain = explain_prompt | llm