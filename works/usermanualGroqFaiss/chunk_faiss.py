from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.schema import Document
import os

from langchain_community.embeddings import HuggingFaceEmbeddings

def chunk_pdf(folder_path):
    all_texts = []
    all_metadata = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            pdf_name = os.path.splitext(file)[0]

            loader = PyPDFLoader(file_path)
            pages = loader.load()  

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)

            for chunk in chunks:
                all_texts.append(chunk.page_content)
                all_metadata.append({
                    "pdf": pdf_name,
                    "page": chunk.metadata.get("page", 0)+1
                })

    return all_texts, all_metadata


def embedding_storing(texts, metadata, save_path=r"D:\intern\database\faiss_db", index_name="zipdatasetindex"):
    model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    documents = []

    for i, (text, meta) in enumerate(zip(texts, metadata)):
        pdf_name = meta['pdf']
        page_number = meta['page']

        

        doc_metadata = {
            "chunk_id": i,
            "pdf": pdf_name,
            "page": page_number,
                
            }

        documents.append(Document(page_content=text, metadata=doc_metadata))

  
    faiss_index = FAISS.from_documents(documents, model)

  
    faiss_index.save_local(folder_path=save_path, index_name=index_name)

    return faiss_index

folder_path = r"D:\intern\zip_dataset\PAP-149_attachments"



texts, metadata = chunk_pdf(folder_path)


embedding_storing(texts, metadata)
