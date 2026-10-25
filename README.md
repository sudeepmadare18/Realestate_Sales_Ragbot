# Real Estate Sales RAG Bot

A Python-based AI chatbot that answers property-related questions using real estate data and property brochures. The project uses RAG (Retrieval-Augmented Generation), ChromaDB, Sentence Transformers, and Hugging Face models to provide accurate answers.

## Problem Statement

Customers often ask questions related to property price, location, amenities, availability, and project details. Sales teams manually search property brochures and data to answer these questions, which takes time. Therefore, there is a need for an AI-based chatbot that can search property data and brochures and provide quick and accurate answers.

## Project Description

This project is a Real Estate Sales RAG Bot that helps users get information about properties through a chatbot interface. The system loads property data, extracts text from property brochures, converts the text into embeddings, and stores them in ChromaDB. When a user asks a question, the system retrieves the most relevant information and generates an answer using RAG.

The project was first developed and tested in Google Colab using Gradio. Later, a Streamlit web application was created and connected to GitHub for deployment.

## Features

- Answer property-related questions
- Search property data and brochure content
- Extract text from PDF brochures
- Convert text into embeddings
- Store and retrieve information using ChromaDB
- RAG-based question answering
- Chatbot interface using Gradio and Streamlit

## Technologies Used

- Python
- Gradio
- Streamlit
- Pandas
- PyMuPDF
- python-docx
- python-pptx
- Sentence Transformers
- Transformers
- Accelerate
- ChromaDB
- Hugging Face Hub
- Openpyxl
- Google Colab
- GitHub

## Workflow

1. User opens the Real Estate Sales RAG Bot.
2. User enters a property-related question.
3. The system searches property data and brochure content.
4. Relevant information is retrieved from ChromaDB.
5. The RAG model generates an accurate answer.
6. The answer is displayed to the user.

## Implementation

- The project was first developed and tested in Google Colab.
- Required libraries such as Gradio, Pandas, PyMuPDF, Sentence Transformers, ChromaDB, and Hugging Face Hub were installed.
- Property data was loaded and processed using Pandas.
- Text was extracted from property brochures using PyMuPDF.
- Text embeddings were created using Sentence Transformers.
- Embeddings were stored in ChromaDB.
- Property search and RAG question answering were implemented.
- Gradio was used for chatbot output.
- Later, a Streamlit web application was created and connected to GitHub.

## How to Run

1. Clone this repository:
   ```bash
   git clone <your-repository-link>
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Output

- User enters a property-related question.
- The system retrieves relevant property information.
- The chatbot displays an accurate answer based on property data and brochures.

## Live Application

[https://realestatesalesragbot-27vgbdw38srnpk4jta2eda.streamlit.app/]

## Future Scope

- Add voice-based question support
- Add multilingual chatbot support
- Connect live property database
- Add property images and location maps
- Improve answer accuracy using advanced AI models

## Developed By

[Sudeep S]
