from langchain_groq import ChatGroq

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key="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"
   
)
messages = [
    (
        "system",
        "You are a helpful assistant telling about daily news",
    ),
    ("human", "what is the headline in india today"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)