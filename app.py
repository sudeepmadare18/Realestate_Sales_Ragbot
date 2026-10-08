import streamlit as st
import pandas as pd
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(
    page_title="Real Estate Sales RAGBot",
    page_icon="🏠",
    layout="wide"
)


# Load property data
df = pd.read_csv("properties.csv")


# Load embedding model
@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# Load Hugging Face LLM
@st.cache_resource
def load_qwen():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    return tokenizer, model


tokenizer, model = load_qwen()


# Title
st.title("🏠 Real Estate Sales RAGBot")

st.write(
    "Find properties and ask questions about their brochures using RAG."
)


# Sidebar
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


# Filter properties
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
    (result["price_lakh"] <= max_price)
]


# Display properties
st.subheader("🏘️ Available Properties")


if len(result) == 0:

    st.warning(
        "No properties found for the selected filters."
    )

else:

    st.success(
        f"{len(result)} property/properties found."
    )

    property_options = result[
        "property_name"
    ].tolist()

    selected_property = st.selectbox(
        "🏠 Select a Property",
        property_options
    )

    property_data = result[
        result["property_name"] == selected_property
    ].iloc[0]


    # Property details
    st.subheader("📋 Property Details")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.write(
            f"**Property:** "
            f"{property_data['property_name']}"
        )

        st.write(
            f"**Builder:** "
            f"{property_data['builder']}"
        )

        st.write(
            f"**City:** "
            f"{property_data['city']}"
        )


    with col2:

        st.write(
            f"**Location:** "
            f"{property_data['location']}"
        )

        st.write(
            f"**BHK:** "
            f"{property_data['bhk']} BHK"
        )

        st.write(
            f"**Area:** "
            f"{property_data['area_sqft']} sq.ft"
        )


    with col3:

        st.write(
            f"**Price:** "
            f"₹{property_data['price_lakh']} Lakhs"
        )

        st.write(
            f"**Possession:** "
            f"{property_data['possession']}"
        )

        st.write(
            f"**Amenities:** "
            f"{property_data['amenities']}"
        )


    st.write("### 📝 Description")

    st.write(
        property_data["property_details"]
    )


    st.divider()


    # Upload brochure
    st.subheader("📄 Upload Property Brochure")


    uploaded_file = st.file_uploader(
        "Upload the PDF brochure for this property",
        type=["pdf"]
    )


    if uploaded_file is not None:

        pdf_path = "uploaded_brochure.pdf"


        with open(pdf_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )


        st.success(
            f"Brochure uploaded: {uploaded_file.name}"
        )


        # PDF loading
        loader = PyPDFLoader(pdf_path)

        documents = loader.load()


        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = text_splitter.split_documents(
            documents
        )


        st.info(
            f"PDF processed into {len(chunks)} chunks."
        )


        # FAISS vector database
        vectorstore = FAISS.from_documents(
            chunks,
            embedding_model
        )


        st.success(
            "Brochure processed successfully!"
        )


        st.divider()


        # Ask question
        st.subheader("💬 Ask About This Property")


        question = st.text_input(
            "Ask a question about the brochure:"
        )


        if question:

            # Retrieve relevant chunks
            retrieved_docs = vectorstore.similarity_search(
                question,
                k=3
            )


            # Create context
            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )


            # RAG prompt
            prompt = f"""
You are a helpful real estate sales assistant.

Use ONLY the information in the brochure below.

BROCHURE:
{context}

QUESTION:
{question}

Answer the question directly in one or two sentences.

If the answer is not available in the brochure,
say that the information is not available in the brochure.

Do not make up information.
"""


            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            )


            # Generate answer
            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False
                )


            # Decode
            answer = tokenizer.decode(
                outputs[0][
                    inputs["input_ids"].shape[1]:
                ],
                skip_special_tokens=True
            ).strip()


            # Display answer
            st.markdown("### 🤖 RAGBot Answer")

            st.write(answer)
