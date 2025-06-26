import streamlit as st

col1,col2=st.columns(2)
with col1:
        st.image("gemini.webp", use_container_width=True)
with col2:
        st.image("groq.png", use_container_width=True)

with st.spinner("AI applications in progress"):
        import tempfile
        import os
        from pdf_chunking_mongo_functions import load_pdf_and_split,embed_texts_and_store,get_relevant_chunks,query_gemini,query_groq

st.title("welcome to AI appliactions")

st.header("Upload a pdf file and get answers at ease ")

pdf_file=st.file_uploader("browse the pdf file ",type=["pdf"])

if pdf_file:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
        with st.spinner("chinking the pdf file "):
                chunks,texts=load_pdf_and_split(tmp_path)

        question=st.text_input("enter your question about the pdf")
        
        with st.spinner("processing your requests"):

                embeddings = embed_texts_and_store(texts)
                relevant_chunks = get_relevant_chunks(question, texts, embeddings)
                answer = query_gemini(question, relevant_chunks)
                result=query_groq(question,relevant_chunks)
        
        gemini=st.button("get answser from gemini")         
        groq=st.button("get answer from groq llama")
        
        with st.spinner("getting answers from gemini"):
        
                if gemini:
                        st.markdown("## Gemini answer is ")
                        st.info(answer)
                        
        with st.spinner("getting answers from groq llama"):
        
                if groq:

                        st.markdown("## Groq answer is ")
                        st.info(result)

        os.remove(tmp_path)

