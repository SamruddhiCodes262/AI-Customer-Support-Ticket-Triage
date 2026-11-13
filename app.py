
import streamlit as st
import joblib
import csv
import os
from datetime import datetime

# Load saved AI models
models = joblib.load("ticket_triage_models.pkl")

vectorizer = models["vectorizer"]
category_model = models["category_model"]
category_encoder = models["category_encoder"]
urgency_model = models["urgency_model"]
urgency_encoder = models["urgency_encoder"]

# Dashboard title
st.title("CircuitGuard Pro")
st.write("AI Customer Support Ticket Triage System")

# Support teams
teams = {
    "Account": "Account Support Team",
    "Billing": "Billing Support Team",
    "Product": "Product Support Team",
    "Technical": "Technical Support Team"
}

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "review_submitted" not in st.session_state:
    st.session_state.review_submitted = False

# Complaint input
complaint = st.text_area("Enter Customer Complaint")

# Analyze complaint
if st.button("Analyze Complaint"):

    if complaint.strip():

        complaint_vector = vectorizer.transform([complaint])

        category = category_encoder.inverse_transform(
            category_model.predict(complaint_vector)
        )[0]

        urgency = urgency_encoder.inverse_transform(
            urgency_model.predict(complaint_vector)
        )[0]

        category_confidence = max(
            category_model.predict_proba(complaint_vector)[0]
        ) * 100

        urgency_confidence = max(
            urgency_model.predict_proba(complaint_vector)[0]
        ) * 100

        # Save prediction in session state
        st.session_state.prediction = {
            "complaint": complaint,
            "category": category,
            "urgency": urgency,
            "category_confidence": category_confidence,
            "urgency_confidence": urgency_confidence
        }

        st.session_state.review_submitted = False

    else:
        st.error("Please enter a customer complaint.")

# Display saved prediction
if st.session_state.prediction:

    result = st.session_state.prediction

    st.subheader("AI Prediction Results")

    st.write("Category:", result["category"])
    st.write("Urgency:", result["urgency"])
    st.write(
        "Assigned Team:",
        teams.get(result["category"], "General Support Team")
    )

    st.write(
        "Category Confidence:",
        round(result["category_confidence"], 2), "%"
    )

    st.write(
        "Urgency Confidence:",
        round(result["urgency_confidence"], 2), "%"
    )

    # Human review system
    if (
        result["category_confidence"] < 60
        or result["urgency_confidence"] < 60
    ):

        st.warning("Human Review Required")

        st.subheader("Human Review Panel")

        if not st.session_state.review_submitted:

            categories = [
                "Account", "Billing", "Product", "Technical"
            ]

            if result["category"] not in categories:
                categories.append(result["category"])

            with st.form("human_review_form"):

                review_decision = st.radio(
                    "Review Decision",
                    ["Approve Prediction", "Reject Prediction"]
                )

                corrected_category = st.selectbox(
                    "Confirm or Correct Category",
                    categories,
                    index=categories.index(result["category"])
                )

                submit_review = st.form_submit_button(
                    "Submit Human Review"
                )

            if submit_review:

                review_status = review_decision

                # Save review record
                file_name = "human_review_records.csv"

                file_exists = os.path.isfile(file_name)

                with open(
                    file_name,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    if not file_exists:
                        writer.writerow([
                            "Date",
                            "Complaint",
                            "Predicted Category",
                            "Urgency",
                            "Review Decision",
                            "Corrected Category"
                        ])

                    writer.writerow([
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        result["complaint"],
                        result["category"],
                        result["urgency"],
                        review_status,
                        corrected_category
                    ])

                st.session_state.review_submitted = True
                st.session_state.review_result = {
                    "decision": review_status,
                    "category": corrected_category
                }

                st.rerun()

        else:

            st.success("Human Review Submitted!")

            st.write(
                "Review Decision:",
                st.session_state.review_result["decision"]
            )

            st.write(
                "Final Category:",
                st.session_state.review_result["category"]
            )

    else:
        st.success("Automatically Assigned")

# Review history download
st.subheader("Human Review History")

if os.path.exists("human_review_records.csv"):

    with open(
        "human_review_records.csv",
        "rb"
    ) as file:

        st.download_button(
            "Download Review Records",
            data=file,
            file_name="human_review_records.csv",
            mime="text/csv"
        )

else:
    st.info("No human review records available yet.")
