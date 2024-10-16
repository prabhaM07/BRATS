import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import tensorflow as tf

tf.compat.v1.reset_default_graph()

GOOGLE_API_KEY = "AIzaSyAldrlmJDsWzdwKoHUwFel_T2KlMMh49uM"
genai.configure(api_key=GOOGLE_API_KEY)

pdf_file_path = r"D:\Projects\Brain_Tumour\Brain_tumor_Final\BRATS\brain_tumor_pdf.pdf"  

def page_setup():
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
            }
            .stButton > button {
                background-color: #007bff;
                color: white;
                border-radius: 4px;
                padding: 10px 20px;
            }
            .stTextInput input {
                border-radius: 4px;
                padding: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

    st.header("🧠 Brain Tumor ChatBot")
    st.markdown("Interact with the chatbot to get insights.")

# No need to return model and other LLM properties
def get_llminfo():
    # If needed, you can still log or adjust these values internally
    model = "gemini-1.5-flash"  # Hardcoded model
    temperature = 1.0  # Default temperature
    top_p = 0.94  # Default top_p for nucleus sampling
    max_tokens = 2000  # Maximum number of tokens
    return model, temperature, top_p, max_tokens

def app():
    page_setup()
    
    # Internal model configuration
    model, temperature, top_p, max_tokens = get_llminfo()

    # Extract text from PDF
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Model generation configuration (hardcoded)
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
    )

    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        response = model_instance.generate_content([question, "response the answer similar to the text given:", text])
        st.write(response.text)

# Run the app
if __name__ == '__main__':
    app()
