import pickle
import streamlit as st
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load pickled TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict(text):
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_vectorized)[0]

    return prediction

# Streamlit UI
st.title("Free AI detection")

# Text input for user to enter text to classify
text_input = st.text_area("To identify AI-generated content, such as ChatGPT, GPT-4, and Google Gemini, copy and paste your English text below.", "")

# Button to trigger classification
if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text to classify.")
    elif len(text_input.split()) < 100:
        st.warning("Please enter at least 100 words of text to classify.")
    else:
        # Perform prediction
        prediction = predict(text_input)

        # Display prediction result
        if prediction == 0:
            st.success("The text is classified as human-written.")
        else:
            st.success("The text is classified as AI-generated.")

linkedin_id = "https://www.linkedin.com/in/aditya-kumar-saw-8493621a6/"

# Display user's name and LinkedIn ID at the bottom
st.write(f"LinkedIn ID: {linkedin_id}")



