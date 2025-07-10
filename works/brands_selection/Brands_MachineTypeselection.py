
import torch
from pymongo import MongoClient
import streamlit as st
from sentence_transformers import SentenceTransformer,util
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
import re
from fuzzywuzzy import fuzz

client = MongoClient("mongodb+srv://manjusri2306:Penquin23@cluster0.alpj0mm.mongodb.net/")  
db = client['medicalscan_brands']
collection = db['medicalscan_convo']

def get_images_by_text_match(answer, score_threshold=70):
    client = MongoClient("mongodb+srv://manjusri2306:Penquin23@cluster0.alpj0mm.mongodb.net/")  
    db = client['medicalscan_brands']
    collection = db['medicalscan_documents']
    

    best_score = 0
    best_doc = None


    for doc in collection.find({}):
        text = doc.get("text", "")
        score = fuzz.partial_ratio(answer, text)

        if score > best_score and score >= score_threshold:
            best_score = score
            best_doc = doc


    if best_doc:
        return best_doc.get("images", [])

    return []

MACHINE_BRAND_MAP = {
    "CT": ["Philips", "GE Healthcare", "Siemens Healthineers","ExploreVista"],
    "MRI": [ "Sy", "Eppeltone", "SDSU"],
    "ECG": ["Biocare", "Gima"],
    "Mammography": ["IHE"],
    "DigitalDiagnost C90":["DigitalDiagnost"]
    
   
}
import re

def detect_machine_type(text):
    text = text.lower().strip()

    
    if "digitaldiagnost c90" in text:
        return "DigitalDiagnost C90"
    elif "mri" in text or "symri" in text:
        return "MRI"
    elif re.search(r"\bct\b", text):
        return "CT"
    elif re.search(r"\becg\b", text):
        return "ECG"
    elif "mammography" in text:
        return "Mammography"
    else:
        return "Unknown"


def detect_brand(text):
    text = text.lower().strip()

   
    if "philips" in text:
        return "Philips"
    elif "siemens" in text or "seimen" in text:
        return "Siemens Healthineers"
    elif "symri" in text or re.search(r"\bsy\b", text):
        return "Sy"
    elif re.search(r"\bge\b", text) or "ge healthcare" in text:
        return "GE Healthcare"
    elif "biocare" in text:
        return "Biocare"
    elif "gima" in text:
        return "Gima"
    elif "eppeltone" in text:
        return "Eppeltone"
    elif "ihe" in text:
        return "IHE"
    elif "sdsu" in text:
        return "SDSU"
    elif "digitaldiagnost" in text:
        return "DigitalDiagnost"
    else:
        return None


def load_from_mongo():
    client = MongoClient("mongodb+srv://manjusri2306:Penquin23@cluster0.alpj0mm.mongodb.net/")  
    db = client['medicalscan_brands']
    collection = db['medicalscan_documents']
    data = list(collection.find())
    texts = data
    embeddings = [doc['embedding'] for doc in data]
    return texts, embeddings


def query_gemini(question,top_chunks):
    context = "\n\n".join(chunk['text'] for chunk in top_chunks)

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
  

def get_relevant_chunks(question, chunks, embeddings, top_k=1, extend_by=3):
    model = SentenceTransformer("thenlper/gte-base")
    question_embedding = model.encode(question)
    cos_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()

    added_indices = set()
    all_selected_indices = []

    for idx in top_indices:
        if idx not in added_indices:
            all_selected_indices.append(idx)
            added_indices.add(idx)
        for offset in range(1, extend_by + 1):
            if idx + offset < len(chunks) and (idx + offset) not in added_indices:
                all_selected_indices.append(idx + offset)
                added_indices.add(idx + offset)
    selected_chunks = [chunks[i] for i in sorted(all_selected_indices)]
    return selected_chunks


def save_to_mongo(question, answer, image_paths):
    collection.insert_one({
        "question": question,
        "answer": answer,
        "image_paths": image_paths
    })


st.title("USER MANUAL")
st.image("medical.jpg", width=600)
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []

question = st.chat_input("Enter your question here and press enter")

if question:
    st.session_state.question = question

    if 'brand_select' in st.session_state:
        del st.session_state['brand_select']
    if 'machine_type_select' in st.session_state:
        del st.session_state['machine_type_select']

if 'question' in st.session_state:
    question = st.session_state.question
    auto_type = detect_machine_type(question)
    auto_brand = detect_brand(question)

    type_options = ["-- Select machine type --"] + list(MACHINE_BRAND_MAP.keys())
    type_index = type_options.index(auto_type) if auto_type in type_options else 0

    if type_index:
        st.success(f"Machine type auto-detected: {auto_type}")
    else:
        st.warning("Machine type not detected. Please select manually.")

    selected_type = st.selectbox("Select Machine Type:", type_options, index=type_index, key="machine_type_select")

    if selected_type == "-- Select machine type --":
        st.stop()

    brand_options = ["-- Select a brand --"] + MACHINE_BRAND_MAP.get(selected_type, [])
    brand_index = brand_options.index(auto_brand) if auto_brand in brand_options else 0

    if brand_index:
        st.success(f"Brand auto-detected: {auto_brand}")
    else:
        st.warning("Brand not detected. Please select manually.")

    brand = st.selectbox("Select Brand:", brand_options, index=brand_index, key="brand_select")

    if brand == "-- Select a brand --":
        st.stop()

    texts, embeddings = load_from_mongo()

    filtered_texts = [doc for doc in texts if doc.get("brand", "").lower() == brand.lower()]
    filtered_embeddings = [doc['embedding'] for doc in filtered_texts]

    if not filtered_texts:
        st.warning(f"No documents found for brand: {brand}")
    else:
        with st.spinner(f"Fetching content for {brand}..."):
            relevant_chunks = get_relevant_chunks(question, filtered_texts, filtered_embeddings)

        with st.spinner("Getting answer from Gemini..."):
            answer, image_paths = query_gemini(question, relevant_chunks)

        if answer and "please specify the brand" not in answer.lower():
            st.session_state.qa_pairs.append((question, answer, image_paths))
            save_to_mongo(question, answer, image_paths)

        del st.session_state.question

for q, a, imgs in st.session_state.qa_pairs:
    st.info(f"You: {q}")
    st.info(f" Gemini: {a}")
   
    for img_path in imgs:
        st.image(img_path, use_container_width=True)
        st.balloons()
