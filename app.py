import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open("tokenizer.pkl","rb") as file:
    tokenizer=pickle.load(file)
model=load_model("topic_classification.keras")

max_length=100
labels = ["Chemistry", "Physics"]

st.title("Subject Classifier")
st.write("enter a question to classify it into Physics or Chemistry")

user_input=st.text_area("Your Question")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("enter a questiion")
    else:
        seq=tokenizer.texts_to_sequences([user_input])
        padded=pad_sequences(seq,maxlen=max_length,padding="post")
        prediction=model.predict(padded)
        pred_class = int(prediction[0] > 0.5)
        st.success(f"Predicted Subject : {labels[pred_class]}")
        st.write("Raw prediction probability:", float(prediction[0]))
