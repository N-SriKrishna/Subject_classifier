import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

model = load_model("subject_classifier.keras")

max_length = 100
labels = ["Chemistry", "Physics", "Maths"]  

st.title("Subject Classifier")
st.write("Enter a question to classify it into Chemistry, Physics, or Maths")

user_input = st.text_area("Your Question")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a question")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_length, padding="post")
        prediction = model.predict(padded)  
        
        pred_class = np.argmax(prediction[0])  
        pred_prob = prediction[0][pred_class]
        
        st.success(f"Predicted Subject: {labels[pred_class]}")
        st.write(f"Prediction confidence: {pred_prob:.4f}")
