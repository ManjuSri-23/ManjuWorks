import torch
from pymongo import MongoClient
import streamlit as st
from sentence_transformers import SentenceTransformer,util
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pymongo import MongoClient
import os
from datetime import datetime
from difflib import SequenceMatcher
import re
from fuzzywuzzy import fuzz


client = MongoClient("mongodb+srv://manjusri2306:Penquin23@cluster0.alpj0mm.mongodb.net/")
db = client['medical_devices_final']
collection = db['usermanual_conversation_images']


def get_images_by_text_match(answer, score_threshold=70):
    client = MongoClient("mongodb+srv://manjusri2306:Penquin23@cluster0.alpj0mm.mongodb.net/")
    db = client["medical_devices_final"]
    collection = db["medical_devices_with_images_final"]

    best_score = 0
    best_doc = None

    # Loop through all documents to find the best fuzzy match
    for doc in collection.find({}):
        text = doc.get("text", "")
        score = fuzz.partial_ratio(answer, text)

        if score > best_score and score >= score_threshold:
            best_score = score
            best_doc = doc

    # ✅ Return images from the matched document itself (not next page)
    if best_doc:
        return best_doc.get("images", [])

    return []



# def get_images_by_text_match(search_text):
#     client = MongoClient('mongodb+srv://manjusri2306:Northenlights23@cluster0.alpj0mm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
#     db = client["medical_devices"]
#     collection = db["medical_devices_with_images"]
    
    

#     query = { "text": { "$regex": search_text, "$options": "i" } }

#     results = collection.find(query)

#     images_list = []
#     for doc in results:
#         pdf_name = doc.get("pdf")
#         current_page = doc.get("page", 0)  # already adjusted to start at 1

#         # Now we want images from page `current_page + 1`
#         next_page = current_page + 1

#         # Find the next page's document
#         next_page_doc = collection.find_one({ "pdf": pdf_name, "page": next_page })

#         if next_page_doc:
#             next_images = next_page_doc.get("images", [])
#             if next_images:
#                 images_list.extend(next_images)

#     return images_list



    
def get_relevant_chunks(question, chunks, embeddings, top_k=1, extend_by=3):
    model = SentenceTransformer("thenlper/gte-base")
    question_embedding = model.encode(question)
    cos_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()

    added_indices = set()
    all_selected_indices = []

    for idx in top_indices:
        
        for i in range(idx, min(idx + extend_by + 1, len(chunks))):
            if i not in added_indices:
                all_selected_indices.append(i)
                added_indices.add(i)

    top_chunks = [chunks[i] for i in all_selected_indices]
    return top_chunks


def query_gemini(question,top_chunks):
    context = "\n\n".join(chunk.page_content for chunk in top_chunks)

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
            img_pth = get_images_by_text_match(snippet)
            if img_pth:
                break  

    return answer, img_pth
  
 
    
def save_to_mongo(question, answer,image):
   
    
    doc={
        'question':question,'answer':answer,"timestamp": datetime.utcnow()  ,'image_path':image
    }
    
    collection.insert_one(doc)

def load_from_mongo():
    client = MongoClient("mongodb+srv://manjusri2306:Penquin23@cluster0.alpj0mm.mongodb.net/")
    db = client['medical_devices_final']
    collection = db['medical_devices_with_images_final']
    data = list(collection.find())
    texts = data
    embeddings = [doc['embedding'] for doc in data]
    return texts, embeddings
    

texts, embeddings = load_from_mongo()

st.markdown("## USER MANUAL FOR MEDICAL DEVICES ")
image=st.image("medical.jpg",width=600)
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []
if image:
    
    
    question = st.chat_input("Enter your question  here and press enter ")
    



    if question:
                
                with st.spinner("processing your request"):
                    relevant_chunks = get_relevant_chunks(question, texts, embeddings)

                
                with st.spinner("Getting answer from Gemini..."):
                    answer, image_paths= query_gemini(question, relevant_chunks)
                   
                    st.session_state.qa_pairs.append((question, answer, image_paths))
                    save_to_mongo(question, answer,image_paths)

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
        
       


                

    
                   
                    

# '''
#     #                                                  INSTRUCTIONS:
#     #                                                  1.CONTENT WILL BE GIVEN TO YOU .
#     #                                                  2.If the retrieved content or your response contains any of the following:
#     #                                                     - "refer to chapter"
#     #                                                     - "see page"
#     #                                                     - "refer to section"
#     #                                                     - "IHE Radiology Handbook"
#     #                                                     - "users handbook"
                                                        
#     #                                                     or any kind of manual, document, chapter, page,exercise like 2.1.1 or external reference YOU MUST NOT answer. Instead, your ONLY response should be:"I DON'T KNOW".Do NOT include any explanation, suggestions, or partial answers.
#     #                                                     Do not copy or mention about the manual or external document.
                    
#     #                                                     Just reply: "I DON'T KNOW"
                                                        
#     #                                                  3..GIVE ANSWERS ONLY FROM GIVEN CONTENT
#     #                                                  4..DO NOT MAKEUP THE ANSWER ,IF ANSWER IS NOT IN CONTENT
#     #                                                  5..IF QUESTION IS UNRELEVANT TO CONTENT GIVEN  , THEN TELL "I dont know " .
#     #                                                  6..THE ANSWERS YOU GENERATE MUST BE PRESENT IN CONTENT NOT ELSEWHERE .
#     #                                                  7.If the content contains multiple steps or procedures (like using  bullets or instruction sequences), output **all the steps** in correct order. Do not skip any.Make sure last point and fiest point  is present.
#     #                                                  8.If the question is asked without the adverb like ing exactly like the content understand it and give the response accordingly . Dont generate someother response.
#     #                                                  9.Don't start the response with something that contains "described as above ".The first sentence of the response should not contain "as described above" or "as mentioned above" or "as stated above". If it starts like this, then skip that sentence and generate response from next sentence.
           
    



     
        
