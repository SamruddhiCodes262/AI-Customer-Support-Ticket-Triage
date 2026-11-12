
import streamlit as st
import joblib

# Load saved AI models
models = joblib.load("ticket_triage_models.pkl")

vectorizer = models["vectorizer"]
category_model = models["category_model"]
category_encoder = models["category_encoder"]
urgency_model = models["urgency_model"]
urgency_encoder = models["urgency_encoder"]

# Dashboard title
st.title("AI Customer Support Ticket Triage")

st.write(
    "Enter a customer complaint to predict its category and urgency."
)

# Complaint input
complaint = st.text_area("Enter Customer Complaint")

# Support teams
teams = {
    "Account": "Account Support Team",
    "Billing": "Billing Support Team",
    "Product": "Product Support Team",
    "Technical": "Technical Support Team"
}

# Prediction button
if st.button("Analyze Complaint"):

    if complaint.strip():

        # Convert complaint into numerical form
        complaint_vector = vectorizer.transform([complaint])

        # Predict category and urgency
        category = category_encoder.inverse_transform(
            category_model.predict(complaint_vector)
        )[0]

        urgency = urgency_encoder.inverse_transform(
            urgency_model.predict(complaint_vector)
        )[0]

        # Calculate confidence
        category_confidence = max(
            category_model.predict_proba(complaint_vector)[0]
        ) * 100

        urgency_confidence = max(
            urgency_model.predict_proba(complaint_vector)[0]
        ) * 100

        # Display results
        st.subheader("AI Prediction Results")
        st.write("Category:", category)
        st.write("Urgency:", urgency)
        st.write("Assigned Team:", teams[category])

        st.write(
            "Category Confidence:",
            round(category_confidence, 2), "%"
        )

        st.write(
            "Urgency Confidence:",
            round(urgency_confidence, 2), "%"
        )

        # Human review
        if category_confidence < 60 or urgency_confidence < 60:
            st.warning("Human Review Required")
        else:
            st.success("Automatically Assigned")

    else:
        st.error("Please enter a customer complaint.")
