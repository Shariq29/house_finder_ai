import pandas as pd
import gradio as gr
import google.generativeai as genai
import os

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Your API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Configure the SDK
genai.configure(api_key=GEMINI_API_KEY)

# Set up local embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# Initialize Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load Data
df = pd.read_excel('house_price.xlsx')

documents = []
for _, row in df.iterrows():
    text = f"""
    Property Details:
    Country: {row['country']}
    City: {row['city']}
    Sqft: {row['sqft']}
    Bedrooms: {row['bedrooms']}
    Price: {row['price']}
    """
    documents.append(Document(text=text))

# Create Index
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)

def query_system(query):
    try:
        print(f"> 1. Searching for: {query}")
        nodes = retriever.retrieve(query)
        context = "\n\n".join([n.text for n in nodes])

        if not context:
            print("> No relevant documents found.")
            return "I couldn't find any properties matching your request."

        print(f"> 2. Context retrieved (first 100 chars): {context[:100]}...")

        prompt = f"""
        You are a real estate assistant.
        Use only the data below to answer.

        Data:
        {context}

        Question : {query}

        Give structured results.
        """

        print("> 3. Calling Gemini API...")
        # Generate content with a specific timeout or simpler call
        response = gemini_model.generate_content(prompt)
        
        if response.text:
            print("> 4. Successfully received response.")
            return response.text
        else:
            return "The AI returned an empty response."

    except Exception as e:
        print(f"> Error occurred during query: {e}")
        return f"Error : {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=query_system,
    inputs=gr.Textbox(placeholder="Ask: properties in India"),
    outputs=gr.Textbox(lines=10),
    title="House Finder AI"
)

demo.launch(share=True)