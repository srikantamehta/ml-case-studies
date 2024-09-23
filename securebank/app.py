from flask import Flask, request, jsonify
from pipeline import Pipeline
import logging
import pandas as pd
import os
import json

app = Flask(__name__)

pipeline = Pipeline(version='random_forest_model_1')

# Configure logging
logging.basicConfig(level=logging.INFO)

print(f"New working directory: {os.getcwd()}")

@app.route('/list_models/', methods=['GET'])
def list_models():
    """
    Endpoint to list all available models in the storage/models/artifacts/ directory.
    """
    try:
        models = pipeline.list_models()
        return jsonify({'available_models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/select_model/', methods=['POST'])
def select_model():
    """
    Endpoint to select a specific model version for future predictions.
    Expects a JSON body with a model_version key.
    """
    try:
        # Parse the input JSON
        input_data = request.get_json()

        # Extract the model version from the request
        model_version = input_data.get('model_version')
        if not model_version:
            return jsonify({'error': 'model_version is required'}), 400
        
        # Select the model using the pipeline
        pipeline.select_model(model_version)

        # Log the selection
        logging.info(f"Model version {model_version} selected successfully.")

        # Return a success message
        return jsonify({'message': f'Model version {model_version} selected successfully.'})
    except Exception as e:
        logging.error(f"Error selecting model: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
@app.route('/model_stats/',methods=['GET'])
def get_model_stats():
    """
    Endpoint to return the chosen models stats from training
    """
    try:
        # Retrieved saved model stats 
        model_stats = pipeline.get_model_stats()
        return jsonify({'model_stats': model_stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/predict/', methods=['POST'])
def predict():
    """
    Endpoint to classify a transaction as legitimate or fraudulent.
    Expects a JSON body with the necessary transaction fields.
    """
    try:
        # Parse the input JSON
        input_data = request.get_json()

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)

        # Log the prediction result
        logging.info(f"Prediction result: {prediction}")

        # Check if the prediction is a list (for multiple inputs) or a single value
        if isinstance(prediction, list):
            # Return a list of predictions if multiple data points were passed
            return jsonify({
                'predictions': [int(pred) for pred in prediction]  # Convert boolean predictions to integers
            })
        else:
            # Return a single prediction if only one data point was passed
            return jsonify({
                'prediction': int(prediction)  # Convert boolean prediction to integer
            })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/get_history/', methods=['GET'])
def get_history():
    """
    Endpoint to retrieve prediction history on transaction data.
    """
    try:
        history = pipeline.get_history()
        logging.info(f"History retrieved: {history}")
        return jsonify({'history': history})
    except Exception as e:
        logging.error(f"Error retrieving history: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

