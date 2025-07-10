
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fuzzywuzzy import fuzz








def get_relevant_chunks(faiss_index, question, top_k=1, extend_by=3):
    
    top_docs = faiss_index.similarity_search(question, k=top_k)
    print(f"Top docs found: {len(top_docs)}")

   
    all_chunks = list(faiss_index.docstore._dict.values()) 
    total_chunks = len(all_chunks)

    selected_chunks = []

    for doc in top_docs:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id is not None:
            
            for i in range(chunk_id, min(chunk_id + extend_by + 1, total_chunks)):
                selected_chunks.append(all_chunks[i])

    return selected_chunks

def load_from_faiss(folder_path, index_name):
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
    faiss_index = FAISS.load_local(
        folder_path=folder_path,
        index_name=index_name,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True 
    )
    return faiss_index

faiss_index = load_from_faiss(
    folder_path="D:/intern/database/faiss_db",
    index_name="zipdatasetindex"
  
)



def query_groq(question,top_chunks):
    context = "\n\n".join(chunk.page_content for chunk in top_chunks)
    
    print(context)
    
    prompt_template=ChatPromptTemplate.from_template( '''You are a expert in medical devices and a helpful assistant providing answers based on given content
                                                    When answering, follow these rules:
                                                    ```
                                                        1.Use same words provided in the content to generate the response.
                                                        2...Extract all step-by-step instructions exactly as they are provided in the context , without reordering or summarizing. Preserve multi-line or paragraph instructions under a single point. Do not omit any explanatory sentences that belong to a step.
                                                        3..If table like content is peresent in the context , Dont omit it while generating response,Give table along with the response.
                                                        4..If any Warining content is present , dont skip points due to warning . Instead Generate points along with the warning content.
                                                        5. First check if the question requires combining information from multiple documents
                                                        6. Dont summarize . Give content as it is.
                                                        7. If found across documents:
                                                            - Start with the most relevant document's answer
                                                            - Add "Additional Context:" sections for supporting info from other docs
                                                            - Never omit relevant info just because it's in Document 2-5
                                                        8. Explicitly state when information continues between documents
                                                        9. Resolve any contradictions between documents by:
                                                            - Noting "Document X states... while Document Y suggests..."
                                                            - Preferring more recent/larger documents when applicable
                                                            - Do not answer the question if you do not know the answer.
                                                        10. Use the context provided to answer the question.
                                                        11. context is a collection of documents related to the question.
                                                        12. Do not make up answers.
                                                        13. context is provided to you to answer the question.
                                                        14. If the context does not contain the answer, say "I don't know".
                                                        15.if picture is present in the context matching page number , use it to answer the question.
                                                        16. Understand different phrasings of questions. For example:
                                                                - "How to adjust FLAIR images?"
                                                                - "Tell me the steps for adjusting FLAIR images"
                                                                - "Explain FLAIR image adjustment"
                                                                - Treat them as the same intent and provide complete answers.
                                                        17.If the context has instructions or points like(1.,2.... or bulletin points) gnerate all the points in response.If some points continues in next paragraph in context  ,include those points also in response.
                                                        
 
                    
                                                     
                                                     Answer the question based only on the following 
                                                     context:{context},
                                                    
                                                    Question: {question}
                                                     
                                                     ''')
    llm = ChatGroq(
    api_key="apikey",
     model="llama3-70b-8192")
    
    chain=prompt_template|llm
    
    result=chain.invoke({
        'context':context,
        'question':question
    }
    )
    answer=result.content
        


    return answer

  
 
  

st.markdown("## USER MANUAL FOR MEDICAL DEVICES ")

if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []

    
    
question = st.chat_input("Enter your question  here and press enter ")
    



if question:
                
                with st.spinner("processing your request"):
                    relevant_chunks = get_relevant_chunks(faiss_index,question)

                
                with st.spinner("Getting answer from Groq..."):
                    answer= query_groq(question, relevant_chunks)
                   
                    st.session_state.qa_pairs.append((question, answer))
                    

for q, a in st.session_state.qa_pairs:
    st.info(f"You: {q}")
    st.info(f" llama: {a}")
   
   
        
       


                

   