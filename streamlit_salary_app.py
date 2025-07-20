import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from PIL import Image

# Load model and mapping
model = joblib.load("best_model.pkl")
edu_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
edu_mapping = {lvl: idx for idx, lvl in enumerate(edu_levels)}

# Optional: Use streamlit-extras for faster load
try:
    from streamlit_extras.switch_page_button import switch_page
except ImportError:
    pass  # Safe fallback if not installed

# Page Config
st.set_page_config(page_title="SmartPay: Salary Predictor", page_icon="💼", layout="wide")

# Custom dark theme styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0E1117;
        }
        h1, h2, h3, h4, h5 {
            color: #00BFFF;
        }
        .main-title {
            font-size: 32px;
            font-weight: bold;
            color: #00BFFF;
            padding-bottom: 10px;
        }
        .block-container {
            padding-top: 2rem;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation as Buttons (No selectbox)
st.sidebar.title("💼 SmartPay Navigation")
about_btn = st.sidebar.button("🏠 About", use_container_width=True)
predict_btn = st.sidebar.button("📈 Prediction", use_container_width=True)

# Default to About if no button clicked
if "page" not in st.session_state:
    st.session_state.page = "about"
if about_btn:
    st.session_state.page = "about"
if predict_btn:
    st.session_state.page = "predict"

# ---- ABOUT SECTION ----
if st.session_state.page == "about":
    st.markdown('<div class="main-title">🚀 SmartPay: AI-Powered Salary Predictor</div>', unsafe_allow_html=True)

    # Side-by-side layout: Image (left) + Description (right)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("smartpay_banner.png", width=350, caption="SmartPay: Predicting Smarter Salaries 💰")

    with col2:
        st.markdown("""
            <div style="font-size:18px; line-height:1.8;">
                📖 <strong>What is SmartPay?</strong><br><br>
                <strong>SmartPay</strong> is an AI-powered salary prediction tool that estimates an employee’s monthly salary based on:
                <ul>
                    <li>📆 <strong>Years of Experience</strong></li>
                    <li>🎓 <strong>Education Level</strong></li>
                    <li>⏱️ <strong>Weekly Working Hours</strong></li>
                </ul>
                It supports both <strong>single</strong> and <strong>bulk predictions</strong>, and offers illustrative feature importance plots to enhance transparency.<br><br>
                <strong>Tech Used:</strong> Streamlit, Scikit-learn, XGBoost, Pandas, NumPy, Joblib, SHAP (ELI5-style explanations)
            </div>""", unsafe_allow_html=True)


# ---- PREDICTION SECTION ----
elif st.session_state.page == "predict":
    mode = st.sidebar.radio("Choose Prediction Mode", ["Single Prediction", "Bulk Prediction"])

    if mode == "Single Prediction":
        st.header("🧮 Single Salary Prediction")
        st.info("### How to use:\n1. Select Experience, Education & Work Hours\n2. Click 'Predict Salary'\n3. See the estimated monthly salary 💰")

        col1, col2 = st.columns(2)

        with col1:
            experience = st.slider("Years of Experience", 0.0, 40.0, 2.0, step=0.5)
            hours = st.slider("Hours Worked Per Week", 20, 200, 40)

        with col2:
            education = st.selectbox("Education Level", edu_levels)

        edu_encoded = edu_mapping[education]

        if st.button("📊 Predict Salary"):
            with st.spinner("Predicting Salary..."):
                input_data = np.array([[experience, edu_encoded, hours]])
                prediction = model.predict(input_data)[0]
                time.sleep(1)

            st.success(f"💵 Estimated Monthly Salary: ₹{prediction:,.2f}")
            st.subheader("📊 Feature Contribution (Illustration)")
            st.image("plots/gradient_boosting_pdp.png", width = 800,caption="Illustrative Feature Importance Plot")

    elif mode == "Bulk Prediction":
        st.header("📁 Bulk Salary Prediction from CSV")
        st.info("### How to use:\n1. Download sample CSV\n2. Fill in data\n3. Upload CSV to get salary predictions")

        st.write("Expected Columns in CSV: `YearsExperience`, `EducationLevel`, `HoursWorkedPerWeek`")

        sample_df = pd.DataFrame({
            'YearsExperience': [5, 10],
            'EducationLevel': ['Bachelor', 'Master'],
            'HoursWorkedPerWeek': [40, 50]
        })

        st.download_button("📥 Download Sample CSV", data=sample_df.to_csv(index=False), file_name="sample_input.csv", mime='text/csv')

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            try:
                df['EducationLevel'] = df['EducationLevel'].map(edu_mapping)
                preds = model.predict(df)
                df['PredictedSalary'] = preds
                st.success("✅ Predictions completed!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📤 Download Predictions CSV", data=csv, file_name="salary_predictions.csv", mime='text/csv')

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Made by **Srishti Bhatnagar** ")
