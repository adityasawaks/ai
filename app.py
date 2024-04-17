import streamlit as st
import pickle

# Load the TF-IDF vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to predict
def predict(text):
    # Preprocess the text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])
    # Make predictions using the loaded model
    prediction = model.predict(text_vectorized)
    return prediction

# Streamlit UI
st.title('Text Classifier')
input_text = st.text_input('Enter text:')
if st.button('Predict'):
    prediction = predict(input_text)
    st.write('Prediction:', prediction)
