import streamlit as st

from pipeline import classify_review

SUITABLE_LABEL = "Suitable for Children"


def confidence_for_predicted_class(result):
    suitable_probability = result["confidence"]

    if result["label"] == SUITABLE_LABEL:
        return suitable_probability

    return 1 - suitable_probability


st.title("Child Suitability Movie Classifier")

user_input = st.text_area("Enter a movie review:")

if st.button("Classify Review"):
    result = classify_review(user_input)
    displayed_confidence = confidence_for_predicted_class(result)

    st.subheader(result["label"])
    st.write(f"Confidence: {displayed_confidence:.2f}")
    st.write(f"Reason: {result['reason']}")
