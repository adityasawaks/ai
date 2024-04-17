import streamlit as st
import joblib

# Load the TF-IDF vectorizer and the model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('model.joblib')

# Define a function to preprocess input data and make predictions
def predict(input_text):
    input_text = tfidf_vectorizer.transform([input_text])
    prediction = model.predict(input_text)
    return prediction

# Streamlit UI
st.title('Your Streamlit App')
st.write('Enter your text below:')
user_input = st.text_input('Text:')
if st.button('Predict'):
    prediction = predict(user_input)
    st.write('Prediction:', prediction)
