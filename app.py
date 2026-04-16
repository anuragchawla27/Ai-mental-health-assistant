import streamlit as st
import joblib

# Load saved model and tools
model = joblib.load("models/lr_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Page config
st.set_page_config(page_title="AI Mental Health Assistant", page_icon="🧠")

# Title
st.title("🧠 AI Mental Health Assistant")
st.write("Type how you're feeling, and the model will analyze your mental state.")

# Input box
user_input = st.text_area("Enter your thoughts here:")

# Response dictionary (based on your dataset)
responses = {
    "Anxiety": "It seems like you're feeling anxious. Try slow breathing and grounding yourself. You're not alone 💙",

    "Depression": "I'm really sorry you're feeling this way. It might help to talk to someone you trust or a professional ❤️",

    "Bipolar": "Your emotions may feel intense or changing. Keeping a routine and talking to a professional can really help 💜",

    "Stress": "You might be under stress. Try taking a short break, relax, and focus on one thing at a time 🌿",

    "Personality Disorder": "It seems like you're dealing with complex emotional patterns. Seeking support from a mental health professional can be very helpful 🤝",

    "Suicidal": "I'm really sorry you're feeling this way. You are not alone. Please reach out to a trusted person or a helpline immediately ❤️",

    "Normal": "You seem to be doing okay 👍 Keep maintaining a healthy routine and positive mindset 😊"
}

# Button
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        input_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vec)
        label = label_encoder.inverse_transform(prediction)[0]

        # Display result
        st.subheader("Prediction:")
        st.success(label)

        # Special alert for critical case
        if label == "Suicidal":
            st.error("⚠️ This seems serious. Please seek immediate help.")
            st.write("📞 Helpline (India): 9152987821")

        # Show response message
        if label in responses:
            st.info(responses[label])

# Footer
st.markdown("---")
st.caption("⚠️ This is not a medical diagnosis tool. Please consult a qualified professional for proper guidance.")