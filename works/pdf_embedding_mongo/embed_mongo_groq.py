
from groq import Groq
import json
from pymongo import MongoClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer,util
import torch 
from pymongo import MongoClient
import google.generativeai as genai

pdf_path=r"D:\sem6\AR VR\unit 4\VR Technology in Physical Exercises and Games.pdf"
loader=PyPDFLoader(pdf_path)
pages=loader.load()

text_splitter=CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=text_splitter.split_documents(pages)
texts=[chunk.page_content for chunk in chunks]
model=SentenceTransformer("thenlper/gte-base")
embeddings = model.encode(texts).tolist()


client=MongoClient('mongodb+srv://manjusri2306u@cluster0.alpj0mm.mongodb.net/')
db=client['database']
collection=db['vr-pdfchunks']

documents_mongo=[]
for i in range(len(texts)):
    doc={
        'chunk_id':i,
        'text':texts[i],
        'embedding':embeddings[i]
    }
    documents_mongo.append(doc)
inserted_document=collection.insert_many(documents_mongo)


question = input("Enter your question about the PDF: ")


model = SentenceTransformer("thenlper/gte-base")
question_embedding = model.encode(question)


cos_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
top_k = 3
top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()

relevant_chunks = [texts[i] for i in top_indices]

context = "\n\n".join(relevant_chunks)
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
