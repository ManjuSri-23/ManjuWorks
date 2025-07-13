
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def get_relevant_chunks(faiss_index, question, top_k=15, rerank_top_n=5):
  
    initial_docs = faiss_index.similarity_search(question, k=top_k)

    
    query_doc_pairs = [(question, doc.page_content) for doc in initial_docs]
    scores = cross_encoder.predict(query_doc_pairs)

  
    scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:rerank_top_n]]

    all_chunks = list(faiss_index.docstore._dict.values())

    
    chunks_by_pdf = {}
    for doc in all_chunks:
        pdf = doc.metadata["pdf"]
        chunk_id = doc.metadata["chunk_id"]
        chunks_by_pdf.setdefault(pdf, {})[chunk_id] = doc

    
    extended_docs = top_docs.copy()
    seen = set((doc.metadata["chunk_id"], doc.metadata["pdf"]) for doc in extended_docs)

    for doc in top_docs:
        base_chunk_id = doc.metadata["chunk_id"]
        pdf = doc.metadata["pdf"]

        for offset in range(1, 4):  
            next_chunk_id = base_chunk_id + offset
            if next_chunk_id in chunks_by_pdf.get(pdf, {}):
                next_doc = chunks_by_pdf[pdf][next_chunk_id]
                key = (next_chunk_id, pdf)
                if key not in seen:
                    extended_docs.append(next_doc)
                    seen.add(key)

    return extended_docs




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
                                                        2.No matter how short, repeated, or UI-related a name is (like Add, Cancel, Exit), you must include it as a row in the table. Do not skip or truncate the table.
                                                        2. Do not skip or merge rows that have similar or same names. If a row appears twice in context, include both rows.
                                                        3.Include every row even if it represents a button (e.g., Add, Cancel, Exit), or the row title is short (like "TLS", "Exit", "Verify"). All such UI-related rows must be included in the same table. Treat them as part of the table if they follow a name–description format. Do not ignore or exclude based on assumed importance or visual layout.
                                                        4. Preserve exact order of rows as they appear.if the continuation of table is in next chunk present in context dont omit those . please incluse that also in your response.
                                                       
                                                        Even if a key-value pair (e.g., "Associated Storage Service") is not visually inside a table but follows the same tabular structure (key followed by description), **treat it as a table row and include it**.
                                                        5.Extract all the row present in table without omitting a row belonging to the table .
                                                        6.Dont skip any rows in table provided in the content.
                                                        7.If table is present in context give as table not as points.
                                                        8.Extract all step-by-step instructions exactly as they are provided in the context , without reordering or summarizing. Preserve multi-line or paragraph instructions under a single point. Do not omit any explanatory sentences that belong to a step.
                                                        9..If table like content is peresent in the context , Dont omit it while generating response,Give table along with the response.
                                                        10..If any Warining content is present , dont skip points due to warning . Instead Generate points along with the warning content.
                                                        11. First check if the question requires combining information from multiple documents
                                                        12. Dont summarize . Give content as it is.
                                                        13. If found across documents:
                                                            - Start with the most relevant document's answer
                                                            - Add "Additional Context:" sections for supporting info from other docs
                                                            - Never omit relevant info just because it's in Document 2-5
                                                        14. Explicitly state when information continues between documents
                                                        15. Resolve any contradictions between documents by:
                                                            - Noting "Document X states... while Document Y suggests..."
                                                            - Preferring more recent/larger documents when applicable
                                                            - Do not answer the question if you do not know the answer.
                                                        16. Use the context provided to answer the question.
                                                        17. context is a collection of documents related to the question.
                                                        18. Do not make up answers.
                                                        19. context is provided to you to answer the question.
                                                        20. If the context does not contain the answer, say "I don't know".
                                                        21.if picture is present in the context matching page number , use it to answer the question.
                                                        22. Understand different phrasings of questions. For example:
                                                                - "How to adjust FLAIR images?"
                                                                - "Tell me the steps for adjusting FLAIR images"
                                                                - "Explain FLAIR image adjustment"
                                                                - Treat them as the same intent and provide complete answers.
                                                        23.If the context has instructions or points like(1.,2.... or bulletin points) gnerate all the points in response.If some points continues in next paragraph in context  ,include those points also in response.
                                                        
 
                    
                                                     
                                                     Answer the question based only on the following 
                                                     context:{context},
                                                    
                                                    Question: {question}
                                                     
                                                     ''')
    llm = ChatGroq(
    api_key="apikey",
     model="llama3-70b-8192",
     temperature=0.0 )
    
    chain=prompt_template|llm
    
    result=chain.invoke({
        'context':context,
        'question':question
    }
    )
    answer=result.content
        


    return answer
st.set_page_config(page_title="Medical Device QA", layout="wide")
  
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
    }
    .sidebar .sidebar-content {
        background-color: #e0f7fa;
    }
    </style>
""", unsafe_allow_html=True)



st.markdown("## USER MANUAL FOR MEDICAL DEVICES")

if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []

question = st.chat_input("Enter your question here and press enter")


if question:
    with st.spinner("Processing your request..."):
        relevant_chunks = get_relevant_chunks(faiss_index, question)

    with st.spinner("Getting answer from Groq..."):
        answer = query_groq(question, relevant_chunks)

    st.session_state.qa_pairs.append((question, answer))




with st.sidebar:
    st.markdown("### Chat History")

    search_query = st.text_input("Search questions")

    if search_query:
        filtered_pairs = [(q, a) for q, a in st.session_state.qa_pairs if search_query.lower() in q.lower()]
        if filtered_pairs:
            for i, (q, _) in enumerate(reversed(filtered_pairs)):
                st.markdown(f"**Q{i+1}:** {q}")
        else:
            st.info("No matches found.")
    else:
        for i, (q, _) in enumerate(reversed(st.session_state.qa_pairs)):
            st.markdown(f"**Q{i+1}:** {q}")

    
    if st.button(" Clear Chat History"):
        st.session_state.qa_pairs = []
        st.rerun()



for q, a in st.session_state.qa_pairs:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
