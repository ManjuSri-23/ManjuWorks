from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from sentence_transformers import SentenceTransformer,util
from pymongo import MongoClient
import os
import torch


def chunk_pdf(folder_path):
    all_chunk=[]
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            loader=PyPDFLoader(file_path)
            pages=loader.load()
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks=text_splitter.split_documents(pages)
            texts=[chunk.page_content for chunk in chunks]
            all_chunk.extend(texts)
    return all_chunk





def embedding_storing(texts):
    
    model=SentenceTransformer("thenlper/gte-base")
    embeddings=model.encode(texts).tolist()
    client=MongoClient('mongodb+srv://manjusri2306:swiss23@cluster0.alpj0mm.mongodb.net/')
    db=client['healthcare']
    collection=db['usermanual']
    collection.delete_many({})  
    documents=[]
    for i , text in enumerate(texts):
        doc={'chunk_id':i,'text':text,'embedding':embeddings[i]}
        documents.append(doc)
    collection.insert_many(documents)
    return embeddings
 

path = r"D:\intern\dataset"
texts = chunk_pdf(path)
embedding_storing(texts)
print("Chunking and embedding stored successfully.")