from flask import Flask, request, jsonify
import pickle
import pandas as pd
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load your model and encoder
try:
    model = pickle.load(open('crop_model.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    print("Model and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    model, label_encoder = None, None

@app.route('/')
def home():
    return "Crop Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the incoming data
        data = request.get_json(force=True)
        print(f"Received data: {data}")  # Log the incoming data

        features = data.get('features')

        # Ensure the features are a list of 7 elements
        if len(features) != 7:
            return jsonify({'message': 'Invalid number of features. Expected 7 features.'}), 400

        # Convert the features into a DataFrame with 1 row and 7 columns
        features = pd.DataFrame([features], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        print(f"Features DataFrame: {features}")  # Log the features DataFrame

        # Ensure the model and encoder are loaded properly
        if model is None or label_encoder is None:
            return jsonify({'message': 'Model or Label Encoder not loaded properly.'}), 500

        # Make the prediction
        prediction = model.predict(features)
        print("predicted",prediction)
        prediction_label = label_encoder.inverse_transform(prediction)
        print(f"Prediction: {prediction_label[0]}")  # Log the prediction

        return jsonify({'prediction': prediction_label[0]})
    
    except Exception as e:
        # Print detailed error logs for debugging
        print(f"Error during prediction: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        return jsonify({'message': 'Error processing the request.', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
