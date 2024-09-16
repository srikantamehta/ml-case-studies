from flask import Flask, request, jsonify
from pipeline import Pipeline
import os

app = Flask(__name__)

pipeline = Pipeline()

print(f"New working directory: {os.getcwd()}")


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

        # Return the result as JSON
        return jsonify({
            'prediction': int(prediction)  # 0: legitimate, 1: fraud
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

