import os
import fitz  # PyMuPDF
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient


def image_extraction(folder_path, output_image_dir="images_extract_new", min_width=100, min_height=100, max_aspect_ratio=4, min_area=0):
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
                    width, height = image[2], image[3]
                    area = width * height

                    # Filter small images
                    if width < min_width or height < min_height:
                        continue

                    # Filter extreme aspect ratio
                    aspect_ratio = max(width / height, height / width)
                    if aspect_ratio > max_aspect_ratio:
                        continue

                    # Filter small area
                    if area < min_area:
                        continue

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
                        "image_path": image_path,
                        "width": width,
                        "height": height
                    })

    return image_metadata



def infer_brand_from_filename(pdf_name):
    name = pdf_name.lower()
    if "philips" in name:
        return "Philips"
    elif "siemen" in name or "healthineer" in name:
        return "Siemens Healthineers"
    elif "ge" in name:
        return "GE Healthcare"
    elif "gima" in name:
        return "Gima"
    elif "biocare" in name:
        return "Biocare"
    elif "sy" in name:
        return "Sy"
    elif "ihe" in name:
        return "IHE"
    elif "sds" in name:
        return "SDSU"
    elif "digitaldiagnost" in name:
        return "DigitalDiagnost"
    elif "explorevista" in name:
        return "ExploreVista"
    elif "eppeltone" in name:
        return "Eppeltone"
    else:
        return "Unknown"


def infer_machine_type_from_filename(pdf_name):
    name = pdf_name.lower()
    if "ct" in name:
        return "CT"
    elif "ecg" in name:
        return "ECG"
    elif "mri" in name:
        return "MRI"
    elif "mammography" in name:
        return "Mammography"
    else:
        return "Unknown"


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
                    "page": chunk.metadata.get("page", 0) + 1
                })

    return all_texts, all_metadata


def embedding_storing(texts, metadata, image_metadata):
    model = SentenceTransformer("thenlper/gte-base")
    embeddings = model.encode(texts).tolist()

    client = MongoClient("mongodb+srv://manjusri2306:Penquin23@cluster0.alpj0mm.mongodb.net/")  
    db = client['medicalscan_brands']
    collection = db['medicalscan_documents']

    documents = []

    for i, (text, meta) in enumerate(zip(texts, metadata)):
        pdf_name = meta['pdf']
        page_number = meta['page']
        brand = infer_brand_from_filename(pdf_name)
        machine_type = infer_machine_type_from_filename(pdf_name)

        matched_images = [
            img["image_path"] for img in image_metadata
            if img["pdf"] == pdf_name and img["page"] == page_number
        ]

        doc = {
            "chunk_id": i,
            "pdf": pdf_name,
            "page": page_number,
            "brand": brand,
            "machine_type": machine_type,
            "text": text,
            "embedding": embeddings[i],
            "images": matched_images
        }

        documents.append(doc)

    collection.insert_many(documents)
    return embeddings
folder_path = r"D:\intern\dataset"


image_metadata = image_extraction(
    folder_path,
    output_image_dir="images_extract_new",
    min_width=100,
    min_height=100,
    max_aspect_ratio=4,
    min_area=20000  )


texts, metadata = chunk_pdf(folder_path)
embedding_storing(texts, metadata, image_metadata)
