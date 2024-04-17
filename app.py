import pickle
import streamlit as st
import pandas as pd
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
st.title("AI vs. Human Text Classifier")

# Text input for user to enter text to classify
text_input = st.text_area("Enter text to classify", "")

# Button to trigger classification
if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Perform prediction
        prediction = predict(text_input)

        # Display prediction result
        if prediction == 0:
            st.success("The text is classified as human-written.")
        else:
            st.success("The text is classified as AI-generated.")
