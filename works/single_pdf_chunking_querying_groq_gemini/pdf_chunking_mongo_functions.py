
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
import torch
import google.generativeai as genai
from groq import Groq

def load_pdf_and_split(pdf_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)
    texts = [chunk.page_content for chunk in chunks]
    return chunks, texts

def embed_texts_and_store(texts, collection_name="vr-pdfchunks"):
    model = SentenceTransformer("thenlper/gte-base")
    embeddings = model.encode(texts).tolist()
    client = MongoClient('mongodb+srv://manjusri2306:lWB5kGNj@cluster0.alpj0mm.mongodb.net/')
    db = client['database']
    collection = db[collection_name]
    documents = []
    for i, text in enumerate(texts):
        doc = {'chunk_id': i, 'text': text, 'embedding': embeddings[i]}
        documents.append(doc)
    collection.insert_many(documents)
    return embeddings

def get_relevant_chunks(question, texts, embeddings, top_k=3):
    model = SentenceTransformer("thenlper/gte-base")
    question_embedding = model.encode(question)
    cos_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()
    top_chunks = [texts[i] for i in top_indices]
    return top_chunks

def query_gemini(question, top_chunks):
    context = "\n\n".join(top_chunks)
    prompt = f"""
Answer the user's question using ONLY the content below.

PDF Content:
{context}

Question:
{question}
"""
    genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

def query_groq(question,top_chunks):
    context = "\n\n".join(top_chunks)
    prompt = f"""
    Answer the user's question using ONLY the content below.

    PDF Content:
    {context}

    Question:
    {question}
    """
    client = Groq(api_key="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")


    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",  
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    )

    result=chat_completion.choices[0].message.content
    return result

def ask_question_from_pdf(pdf_path, question):
    chunks, texts = load_pdf_and_split(pdf_path)
    embeddings = embed_texts_and_store(texts)
    relevant_chunks = get_relevant_chunks(question, texts, embeddings)
    answer = query_gemini(question, relevant_chunks)
    return answer


# pdf_path = r"D:\sem6\AR VR\unit 4\VR Technology in Physical Exercises and Games.pdf"
# question = input("Enter your question about the PDF: ")
# response = ask_question_from_pdf(pdf_path, question)
# print(response)

