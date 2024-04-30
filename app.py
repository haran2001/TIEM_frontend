import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import textwrap
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import requests, json, sys
# from pydantic_settings import BaseSettings
# from constants import CHROMA_SETTINGS
# from streamlit_chat import message
import shutil

from huggingface_hub import login, logout
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(layout="wide")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
persist_directory = "db"

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

#@st.cache_data
#function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages
def display_conversation(history):
    # st.write(history["past"][-1])
    # st.write(history["generated"][-1])
    for i in range(len(history["generated"])-1,-1,-1):
        # message(history["past"][i], is_user=True, key=str(i) + "_user")
        # message(history["generated"][i],key=str(i))
        st.write(history["past"][i])
        st.write(history["generated"][i])

def create_pdf(path: str)->str:
    loader = PyPDFLoader(path)
    data = loader.load()
    pdfLoaderData = ""
    for page in data:
        pdfLoaderData = pdfLoaderData + page.page_content

    # prompt: How to fix UnicodeEncodeError: 'latin-1' codec can't encode character '\u2022' in position 261: ordinal not in range(256)

    pdfLoaderData = pdfLoaderData.encode('utf-8').decode('unicode_escape')
    return pdfLoaderData

def send_pdf(data,name, inference_url):
    payload = {"pdf_name": name, "pdf_string": data}
    r = requests.post(f"{inference_url}/store_pdf", data=json.dumps(payload), headers={"Content-Type": "application/json"})
    print("*******************************")
    print(r)
    print("*******************************")

def add_logo():
    #st.image(".\\assets\\images\\images.jpeg",width=250)
    st.image("assets/images/images.jpeg",width=250)

def main():
    # if "previousFile" not in st.session_state:
    #     st.session_state['previousFile'] = None
    # st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)
    add_logo()
    st.markdown("<h2 style='text-align: center; color:white;'>TI-GPT</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color:white;'>Personal AI Assistant for TIers!</h3>", unsafe_allow_html=True)
    inference_url = st.text_input("",key="urlholder")
    uploaded_file = st.file_uploader("", type=["pdf"])
    if uploaded_file is not None:
        if os.path.isdir("db"):
            shutil.rmtree("db")
        if os.path.isdir("docs"):
            shutil.rmtree("docs")
            os.mkdir("docs")

        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }

        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        if "previousFile" not in st.session_state:
                st.session_state["previousFile"] = uploaded_file.name

        if st.session_state["previousFile"] != uploaded_file.name or ('first' not in st.session_state):
            st.session_state['first'] = True
            data = create_pdf(filepath)
            send_pdf(data=data, name=uploaded_file.name,inference_url=inference_url)
            st.session_state["previousFile"] = uploaded_file.name
            if "generated" in st.session_state:
                st.session_state["generated"] = ["I am ready to help you."]
                st.session_state["past"]      = ["Hi TIer!"]
    if uploaded_file is not None:
        col1, col2= st.columns([1,2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
        #     with st.spinner('Embeddings are in process...'):
        #         ingested_data = data_ingestion()
        #     st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey TIer!"]

            # Search the database for a response based on user input and update session state
            if user_input:
                payload = {'question': f"{user_input}" }
                response = requests.post(f"{inference_url}/answer", data=json.dumps(payload), headers={"Content-Type": "application/json"})
                answer = json.loads(response.text)
                answer = answer['response']
                st.session_state["past"].append(user_input)
                response = answer
                if 'Helpful Answer:' in response:
                    response = response.split('Helpful Answer: ')[1]
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()
