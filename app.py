```python
import streamlit as st
import pandas as pd
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Real Estate Sales RAGBot",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# LOAD PROPERTY DATA
# --------------------------------------------------

df = pd.read_csv("properties.csv")


# --------------------------------------------------
# LOAD HUGGING FACE EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# --------------------------------------------------
# LOAD HUGGING FACE LLM
# --------------------------------------------------

@st.cache_resource
def load_qwen():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    return tokenizer, model


tokenizer, model = load_qwen()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏠 Real Estate Sales RAGBot")

st.write(
    "Find properties using filters and ask questions "
    "about the selected property's brochure using RAG."
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🔍 Property Filters")


builders = ["All"] + sorted(
    df["builder"].unique().tolist()
)

cities = ["All"] + sorted(
    df["city"].unique().tolist()
)

bhk_options = ["All"] + sorted(
    df["bhk"].unique().tolist()
)


selected_builder = st.sidebar.selectbox(
    "Select Builder",
    builders
)


selected_city = st.sidebar.selectbox(
    "Select City",
    cities
)


selected_bhk = st.sidebar.selectbox(
    "Select BHK",
    bhk_options
)


min_price = st.sidebar.number_input(
    "Minimum Price (₹ Lakhs)",
    min_value=0,
    value=0
)


max_price = st.sidebar.number_input(
    "Maximum Price (₹ Lakhs)",
    min_value=1,
    value=150
)


# --------------------------------------------------
# FILTER PROPERTIES
# --------------------------------------------------

result = df.copy()


if selected_builder != "All":

    result = result[
        result["builder"] == selected_builder
    ]


if selected_city != "All":

    result = result[
        result["city"] == selected_city
    ]


if selected_bhk != "All":

    result = result[
        result["bhk"] == int(selected_bhk)
    ]


result = result[
    (result["price_lakh"] >= min_price)
    &
   
