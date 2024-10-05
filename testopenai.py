from openai import OpenAI
import streamlit as st
from transformers import pipeline

st.title("Hugging Face Demo")
text = st.text_input("Enter text to analyze")
@st.cache_resource()
def get_model():
    return pipeline("sentiment-analysis")
model = get_model()
if text:
    result = model(text)
    st.write("Sentiment:", result[0]["label"])
    st.write("Confidence:", result[0]["score"])
st.title("OpenAI Version")
analyze_button = st.button("Analyze Text")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
if analyze_button:
    messages = [
        {"role": "system", "content": """You are a helpful sentiment analysis assistant.
            You always respond with the sentiment of the text you are given and the confidence of your sentiment analysis with a number between 0 and 1"""},
        {"role": "user", "content": f"Sentiment analysis of the following text: {text}"}
    ]
    MODEL = "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Knock knock."},
            {"role": "assistant", "content": "Who's there?"},
            {"role": "user", "content": "Orange."},
        ],
        temperature=0,
    )
    sentiment = response.choices[0].message['content'].strip()
    st.write(sentiment)