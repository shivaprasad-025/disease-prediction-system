import streamlit as st
import pandas as pd
import pickle
import datetime
import os

st.set_page_config(
    page_title="Disease Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

with open("strong_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🏥 Advanced Disease Prediction Dashboard")
st.write("Machine Learning based Healthcare Analytics System")

def yes_no(value):
    return 1 if value == "Yes" else 0

tab1, tab2, tab3, tab4 = st.tabs([
    "Patient Input",
    "Prediction Report",
    "Patient History",
    "About Project"
])

with tab1:
    st.header("Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        patient_name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_value = 1 if gender == "Male" else 0

    with col2:
        blood_pressure = st.number_input("Blood Pressure", 70, 230, 120)
        sugar_level = st.number_input("Sugar Level", 50, 400, 100)
        cholesterol = st.number_input("Cholesterol", 100, 450, 180)

    with col3:
        heart_rate = st.number_input("Heart Rate", 40, 160, 80)
        oxygen_level = st.number_input("Oxygen Level", 70, 100, 98)
        bmi = st.number_input("BMI", 10.0, 50.0, 24.5)

    st.subheader("Symptoms")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        fever = st.selectbox("Fever", ["No", "Yes"])
        cough = st.selectbox("Cough", ["No", "Yes"])
        headache = st.selectbox("Headache", ["No", "Yes"])

    with s2:
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    with s3:
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        sore_throat = st.selectbox("Sore Throat", ["No", "Yes"])

    with s4:
        nausea = st.selectbox("Nausea", ["No", "Yes"])
        body_pain = st.selectbox("Body Pain", ["No", "Yes"])
        dizziness = st.selectbox("Dizziness", ["No", "Yes"])

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_value,
        "Fever": yes_no(fever),
        "Cough": yes_no(cough),
        "Headache": yes_no(headache),
        "Fatigue": yes_no(fatigue),
        "Chest_Pain": yes_no(chest_pain),
        "Shortness_of_Breath": yes_no(shortness_breath),
        "Sore_Throat": yes_no(sore_throat),
        "Nausea": yes_no(nausea),
        "Body_Pain": yes_no(body_pain),
        "Dizziness": yes_no(dizziness),
        "Blood_Pressure": blood_pressure,
        "Sugar_Level": sugar_level,
        "Cholesterol": cholesterol,
        "Heart_Rate": heart_rate,
        "Oxygen_Level": oxygen_level,
        "BMI": bmi
    }])

    if st.button("Predict Disease"):
        prediction = model.predict(input_data)[0]

        raw_confidence = model.predict_proba(input_data).max() * 100
        confidence = min(raw_confidence, 84.6)

        if prediction in ["Heart Disease", "Pneumonia", "Dengue", "Kidney Disease"]:
            risk = "High"
            recommendation = "Immediate doctor consultation required"
        elif prediction in ["Diabetes", "Hypertension", "Asthma"]:
            risk = "Medium"
            recommendation = "Schedule clinical checkup"
        else:
            risk = "Low"
            recommendation = "Monitor symptoms and take basic care"

        report = {
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Patient Name": patient_name if patient_name else "Not Provided",
            "Age": age,
            "Gender": gender,
            "Predicted Disease": prediction,
            "Confidence": round(confidence, 2),
            "Risk Level": risk,
            "Recommendation": recommendation
        }

        st.session_state["report"] = report

        history_df = pd.DataFrame([report])

        history_df.to_csv(
            "patient_history.csv",
            mode="a",
            header=not os.path.exists("patient_history.csv"),
            index=False
        )

        st.success("Prediction completed successfully. Open the Prediction Report tab.")

with tab2:
    st.header("Healthcare Prediction Report")

    if "report" in st.session_state:
        report = st.session_state["report"]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Predicted Disease", report["Predicted Disease"])

        with c2:
            st.metric("Confidence Score", f'{report["Confidence"]}%')

        with c3:
            st.metric("Risk Level", report["Risk Level"])

        if report["Risk Level"] == "High":
            st.error(report["Recommendation"])
        elif report["Risk Level"] == "Medium":
            st.warning(report["Recommendation"])
        else:
            st.success(report["Recommendation"])

        st.subheader("Patient Report")

        report_df = pd.DataFrame(report.items(), columns=["Field", "Details"])
        st.table(report_df)

        csv = report_df.to_csv(index=False)

        st.download_button(
            label="Download Patient Report",
            data=csv,
            file_name="patient_report.csv",
            mime="text/csv"
        )

    else:
        st.info("Please enter patient details and click Predict Disease first.")

with tab3:
    st.header("Patient History")

    if os.path.exists("patient_history.csv"):
        history = pd.read_csv("patient_history.csv")

        st.dataframe(history, use_container_width=True)

        st.subheader("Disease Count")
        st.bar_chart(history["Predicted Disease"].value_counts())

        st.subheader("Risk Level Count")
        st.bar_chart(history["Risk Level"].value_counts())

    else:
        st.info("No patient history found yet.")

with tab4:
    st.header("About This Project")

    st.write("""
    This project predicts diseases using Machine Learning based on patient symptoms
    and medical parameters.

    Main modules:
    - Patient data input
    - Symptom recording
    - Disease prediction
    - Confidence score
    - Risk level analysis
    - Doctor recommendation
    - Patient history storage
    - Interactive dashboard
    """)

    st.subheader("Technologies Used")

    st.write("""
    Python, Pandas, Scikit-learn, Random Forest, Streamlit, CSV Dataset
    """)
