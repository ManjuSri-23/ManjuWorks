from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import CharacterTextSplitter
pdf_path=r"D:\sem6\AR VR\unit 4\VR Technology in Physical Exercises and Games.pdf"
loader=PyPDFLoader(pdf_path)
pages=loader.load()

text_splitter=CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=text_splitter.split_documents(pages)
texts = [chunk.page_content for chunk in chunks]
model = SentenceTransformer("thenlper/gte-base")

embeddings = model.encode(texts).tolist()

#print(embeddings)


client = chromadb.PersistentClient(path="chroma_store")  

collection = client.get_or_create_collection("vr_pdf_chunks")

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(texts))]
)

print("Stored embeddings in ChromaDB")

query = "How VR is used in physical exercise?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)
print(results['documents'][0][0])  