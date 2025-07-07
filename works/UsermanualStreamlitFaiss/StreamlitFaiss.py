
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fuzzywuzzy import fuzz





def get_images_by_text_match(answer, faiss_index, score_threshold=70):
    best_score = 0
    best_images = []

    for doc in faiss_index.docstore._dict.values():
        text = doc.page_content
        score = fuzz.partial_ratio(answer, text)

        if score > best_score and score >= score_threshold:
            best_score = score
            best_images = doc.metadata.get("images", [])

    return best_images





def get_relevant_chunks(faiss_index, question, top_k=1, extend_by=2):
    
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
    index_name="myFaissIndex"
  
)
for doc in faiss_index.docstore._dict.values():
    print("Chunk:", doc.page_content[:200], "...")
    break



def query_gemini(question,top_chunks):
    context = "\n\n".join(chunk.page_content for chunk in top_chunks)
    print("======= FULL CONTEXT =======")
    print(context)

    
    prompt_template=ChatPromptTemplate.from_template( '''You are a expert in medical devices and a helpful assistant providing answers based on given content
                                                     instructions:
                                                    When answering, follow these rules:
                                                        1.Use same words provided in the content to generate the response.
                                                        1. First check if the question requires combining information from multiple documents
                                                        2. If found across documents:
                                                            - Start with the most relevant document's answer
                                                            - Add "Additional Context:" sections for supporting info from other docs
                                                            - Never omit relevant info just because it's in Document 2-5
                                                        3. Explicitly state when information continues between documents
                                                        4. Resolve any contradictions between documents by:
                                                            - Noting "Document X states... while Document Y suggests..."
                                                            - Preferring more recent/larger documents when applicable
                                                            - Do not answer the question if you do not know the answer.
                                                        5. Use the context provided to answer the question.
                                                        6. context is a collection of documents related to the question.
                                                        7. Do not make up answers.
                                                        8. context is provided to you to answer the question.
                                                        9. If the context does not contain the answer, say "I don't know".
                                                        10.if picture is present in the context matching page number , use it to answer the question.
                                                        11. Understand different phrasings of questions. For example:
                                                                - "How to adjust FLAIR images?"
                                                                - "Tell me the steps for adjusting FLAIR images"
                                                                - "Explain FLAIR image adjustment"
                                                                - Treat them as the same intent and provide complete answers.
                                                        12.If the content has instructions or points like(1.,2.... or bulletin points) gnerate all the points in response.If some points continues in next paragraph in context  ,include those points also in response.
 
                    
                                                     
                                                     Answer the question based only on the following 
                                                     context:{context},
                                                    
                                                    Question: {question}
                                                     
                                                     ''')
    llm = ChatGoogleGenerativeAI(
    api_key="AIzaSyC2MccWvfnCQNTz3p9jPlGS3sBDcLdEneI",
    model="gemini-2.0-flash")
    
    chain=prompt_template|llm
    
    result=chain.invoke({
        'context':context,
        'question':question
    }
    )
    answer=result.content
        
    img_pth = []

    
    candidate_snippets = [
        " ".join(answer.split()[8:16]),              
        " ".join(answer.split()[:10]),               
        " ".join(answer.split()[:5]),              
        " ".join(answer.split()[-10:])               
    ]

    for snippet in candidate_snippets:
        if snippet.strip():  
            img_pth = get_images_by_text_match(snippet,faiss_index)
            if img_pth:
                break  

    return answer, img_pth
  
 
  

st.markdown("## USER MANUAL FOR MEDICAL DEVICES ")
image=st.image("medical.jpg",width=600)
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []
if image:
    
    
    question = st.chat_input("Enter your question  here and press enter ")
    



    if question:
                
                with st.spinner("processing your request"):
                    relevant_chunks = get_relevant_chunks(faiss_index,question)

                
                with st.spinner("Getting answer from Gemini..."):
                    answer, image_paths= query_gemini(question, relevant_chunks)
                   
                    st.session_state.qa_pairs.append((question, answer, image_paths))
                    

for q, a, imgs in st.session_state.qa_pairs:
    st.info(f"You: {q}")
    st.info(f" Gemini: {a}")
   
    for img_path in imgs:
        if isinstance(img_path, str) and os.path.exists(img_path):
            img_path = img_path.replace("\\", "/")
            img_path = os.path.abspath(img_path)
        
            st.image(img_path, width=400)
            st.balloons()
        else:
            st.warning(f"Image not found: {img_path}")
        
       


                

   