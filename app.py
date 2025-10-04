from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)

# Load models and scaler once on startup
mlp = joblib.load(r"C:\Users\vansh\OneDrive\Documents\diabetic_screening_project\notebooks\models/pima_mlp_model.pkl")
scaler = joblib.load(r"C:\Users\vansh\OneDrive\Documents\diabetic_screening_project\notebooks\models/pima_scaler.pkl")
cnn_model =  load_model(r"C:\Users\vansh\OneDrive\Documents\diabetic_screening_project\notebooks\models/efficientnetb3_aptos.h5")   # correct path

clinical_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    results = {}

    # --- Clinical Prediction ---
    clinical = data.get('clinical_features')  # expect dict with keys=clinical_cols
    if clinical:
        try:
            input_vals = np.array([[clinical[col] for col in clinical_cols]], dtype=float)
            input_scaled = scaler.transform(input_vals)
            mlp_prob = mlp.predict_proba(input_scaled)[0,1]
            results['diabetes_risk'] = round(float(mlp_prob), 4)
        except Exception as e:
            results['clinical_error'] = str(e)

    # --- Image Prediction ---
    image_b64 = data.get('retinal_image_base64')  # Expect base64-encoded image string
    if image_b64:
        try:
            import base64
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes)).resize((224, 224))
            img_array = np.array(img)/255.0
            if img_array.shape[-1] == 4:  # remove alpha if present
                img_array = img_array[..., :3]
            img_array = np.expand_dims(img_array, axis=0)
            preds = cnn_model.predict(img_array)
            dr_stage = int(np.argmax(preds[0]))
            results['dr_stage'] = dr_stage
            results['dr_stage_probs'] = {str(i): float(preds[0][i]) for i in range(len(preds[0]))}
        except Exception as e:
            results['image_error'] = str(e)

    # --- Optional referral logic ---
    if 'diabetes_risk' in results and 'dr_stage' in results:
        if results['diabetes_risk'] > 0.5 and results['dr_stage'] > 0:
            results['recommendation'] = "Refer to specialist"
        else:
            results['recommendation'] = "Routine check-up"

    if not results:
        results['error'] = "No valid input data provided"

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
