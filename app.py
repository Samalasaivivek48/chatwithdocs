import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from docx import Document

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
    print("Google API key not found.")
else:
    genai.configure(api_key=api_key)

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_txt_text(txt_file):
    return txt_file.read().decode("utf-8")

def get_docx_text(docx_file):
    text = ""
    doc = Document(docx_file)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert assistant. Answer the question as thoroughly and accurately as possible based on the given context. If the answer is not available in the provided context, say, "The answer is not available in the context." Provide concise, clear, and complete explanations.\n\n
    Context:\n {context}\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def process_files(files):
    raw_text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            raw_text += get_pdf_text(file)
        elif file.name.endswith(".txt"):
            raw_text += get_txt_text(file)
        elif file.name.endswith(".docx"):
            raw_text += get_docx_text(file)
    return raw_text

def main():
    st.set_page_config("Chat with Files")
    st.header("Chat with PDF/TXT/DOCX using Gemini💁")

    user_question = st.text_input("Ask a Question from the Documents")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        files = st.file_uploader("Upload Files (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if files:
                with st.spinner("Processing..."):
                    raw_text = process_files(files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed!")
            else:
                st.error("Please upload at least one file.")

if __name__ == "__main__":
    main()
