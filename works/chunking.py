from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
pdf_path=r"D:\sem6\AR VR\unit 4\VR Technology in Physical Exercises and Games.pdf"
loader=PyPDFLoader(pdf_path)
pages=loader.load()

#recursive splitting 
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# #fixed size splitting 
# from langchain.text_splitter import CharacterTextSplitter

# # text_splitter = CharacterTextSplitter(
# #     chunk_size=1000,
# #     chunk_overlap=200
# # )

# #token based splitting

# from langchain.text_splitter import TokenTextSplitter

# text_splitter = TokenTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50)

# split_docs=text_splitter.split_documents(pages)


# for i , chunks in enumerate(split_docs):
#     print(f" chunk{i+1}: {chunks.page_content}")
#     break


from langchain_community.document_loaders import UnstructuredWordDocumentLoader


doc_path =r"C:\Users\MANJUSREE\Downloads\MULTIGEN VR FRAMEWORK (1).docx"
loader = UnstructuredWordDocumentLoader(doc_path)
documents = loader.load()
print(documents)