from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Create HuggingFace pipeline (FREE model)
pipe = pipeline("text2text-generation", model="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=pipe)

# 2. Basic LLM
print("Basic LLM:")
print(llm.invoke("What is Artificial Intelligence?"))

# 3. Prompt + Chain
prompt = PromptTemplate.from_template("Explain {topic} simply")
chain = prompt | llm | StrOutputParser()
print("\nChain Output:")
print(chain.invoke({"topic": "Machine Learning"}))

# 4. Memory (latest method)
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
conversation = RunnableWithMessageHistory(
    llm,
    get_session_history
)

print("\nMemory:")
print(conversation.invoke("Hi", config={"configurable": {"session_id": "1"}}))
print(conversation.invoke("What did I just say?", config={"configurable": {"session_id": "1"}}))