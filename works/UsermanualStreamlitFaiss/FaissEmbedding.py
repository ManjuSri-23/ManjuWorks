from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from sentence_transformers import SentenceTransformer,util
from langchain.schema import Document
import os
import torch
import fitz
from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings


def image_extraction(folder_path, output_image_dir="images_extracted_faiss"):
    os.makedirs(output_image_dir, exist_ok=True)
    image_metadata = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            pdf_file = fitz.open(file_path)
            pdf_name = os.path.splitext(file)[0]

            for page_num in range(len(pdf_file)):
                page = pdf_file[page_num]
                images = page.get_images(full=True)

                for i, image in enumerate(images, start=1):
                    xref = image[0]
                    base_image = pdf_file.extract_image(xref)
                    image_bytes = base_image['image']
                    image_ext = base_image['ext']
                    
                    image_filename = f"{pdf_name}_page{page_num+1}_img{i}.{image_ext}"
                    image_path = os.path.join(output_image_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                   
                    image_metadata.append({
                        "pdf": pdf_name,
                        "page": page_num + 1,
                        "image_path": image_path
                    })
    
    return image_metadata

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


def embedding_storing(texts, metadata, image_metadata, save_path=r"D:\intern\database\faiss_db", index_name="myFaissIndex"):
    model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    documents = []

    for i, (text, meta) in enumerate(zip(texts, metadata)):
        pdf_name = meta['pdf']
        page_number = meta['page']

        matched_images = [
            img["image_path"] for img in image_metadata
            if img["pdf"] == pdf_name and img["page"] == page_number
        ]

        doc_metadata = {
            "chunk_id": i,
            "pdf": pdf_name,
            "page": page_number,
            "images": matched_images
        }

        documents.append(Document(page_content=text, metadata=doc_metadata))

    # Create FAISS index
    faiss_index = FAISS.from_documents(documents, model)

    # Save index
    faiss_index.save_local(folder_path=save_path, index_name=index_name)

    return faiss_index

folder_path = r"D:\intern\dataset"

image_metadata = image_extraction(folder_path)

texts, metadata = chunk_pdf(folder_path)


embedding_storing(texts, metadata, image_metadata)

