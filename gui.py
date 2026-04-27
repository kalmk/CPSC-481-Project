import streamlit as st

from pipeline import classify_review

st.title("Child Suitability Movie Classifier")

user_input = st.text_area("Enter a movie review:")

if st.button("Classify Review"):
    result = classify_review(user_input)

    st.subheader(result["label"])
    st.write(f"Confidence: {result['confidence']:.2f}")
    st.write(f"Reason: {result['reason']}")