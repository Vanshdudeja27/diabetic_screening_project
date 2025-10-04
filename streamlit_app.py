import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

# Load models & scaler once
mlp = joblib.load(r"C:\Users\vansh\OneDrive\Documents\diabetic_screening_project\notebooks\models\pima_mlp_model.pkl")
scaler = joblib.load(r"C:\Users\vansh\OneDrive\Documents\diabetic_screening_project\notebooks\models\pima_scaler.pkl")
cnn_model = load_model(r"C:\Users\vansh\OneDrive\Documents\diabetic_screening_project\notebooks\models\efficientnetb3_aptos.h5")

clinical_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']

def predict_diabetes(clinical_data):
    input_vals = np.array([[clinical_data[col] for col in clinical_cols]], dtype=float)
    input_scaled = scaler.transform(input_vals)
    mlp_prob = mlp.predict_proba(input_scaled)[0,1]
    return round(float(mlp_prob), 4)

def predict_dr_stage(img):
    img = img.resize((224, 224)).convert('RGB')
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = cnn_model.predict(img_array)
    dr_stage = int(np.argmax(preds[0]))
    return dr_stage, {str(i): float(preds[0][i]) for i in range(len(preds[0]))}

st.title("Diabetes and Retinopathy Screening")

# Upload retinal image
uploaded_file = st.file_uploader("Upload Retinal Image (JPG/PNG)", type=['jpg','jpeg','png'])

# Clinical inputs
st.subheader("Enter Clinical Features")
clinical_data = {}
for col in clinical_cols:
    clinical_data[col] = st.number_input(col, min_value=0.0, step=0.1, format="%.2f")

# Predict button
if st.button("Predict"):
    results = {}

    # Clinical prediction
    try:
        if any(clinical_data.values()):  # Check if any clinical input is provided
            diabetes_risk = predict_diabetes(clinical_data)
            # Convert risk probability to binary diagnosis for display (threshold 0.5)
            diabetes_diagnosis = 1 if diabetes_risk > 0.5 else 0
            results['Diabetes Risk (Probability)'] = round(diabetes_risk, 4)
            results['Diabetes Diagnosis '] = f"{diabetes_diagnosis} ({'Diabetes suspected' if diabetes_diagnosis == 1 else 'No diabetes'})"

    except Exception as e:
        st.error(f"Clinical data error: {str(e)}")

    # Image prediction
    if uploaded_file is not None:
        try:
            image_data = Image.open(uploaded_file)
            dr_stage, stage_probs = predict_dr_stage(image_data)

            # Map DR stage to clear description
            dr_stage_desc = {
                0: "No Diabetic Retinopathy",
                1: "Mild Diabetic Retinopathy",
                2: "Moderate Diabetic Retinopathy",
                3: "Severe Diabetic Retinopathy",
                4: "Proliferative Diabetic Retinopathy"
            }

            results['Diabetic Retinopathy Stage'] = f"{dr_stage} ({dr_stage_desc.get(dr_stage, 'Unknown')})"
            results['Stage Probabilities'] = stage_probs
        except Exception as e:
            st.error(f"Image data error: {str(e)}")

    # If no clinical data but DR stage predicted, set diabetes risk based on DR stage
    if not any(clinical_data.values()) and 'Diabetic Retinopathy Stage' in results:
        # Extract numeric stage from the string e.g., "2 (Moderate Diabetic Retinopathy)"
        numeric_stage = int(results['Diabetic Retinopathy Stage'].split()[0])
        if numeric_stage == 0:
            results['Diabetes Risk'] = "0 (No diabetes)"
        else:
            results['Diabetes Risk'] = "1 (Diabetes suspected based on DR stage)"

    # Show results with titles and formatting
    if results:
        st.write("### Prediction Results")
        for key, value in results.items():
            if key == 'Stage Probabilities':
                st.write(f"**{key}:**")
                st.json(value)
            else:
                st.write(f"**{key}:** {value}")
    else:
        st.warning("Please provide clinical data and/or upload an image.")
