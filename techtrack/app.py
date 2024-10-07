from flask import Flask, request, jsonify
from inference.inference_service import InferenceService

app = Flask(__name__)

@app.route('/run_inference', methods=['POST'])
def run_inference():
    """
    API endpoint to trigger the inference service.
    Expected payload: JSON with 'video_source' and other optional config parameters.
    """
    # Log request for debugging
    print("Request received")

    # Get JSON data from the request
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    config = data.get('config')
    video_source = data.get('video_source')

    if not config or not video_source:
        return jsonify({"error": "Missing config or video source"}), 400

    # Initialize the InferenceService
    inference_service = InferenceService(config)

    # Run inference
    print(f"Running inference on {video_source}")
    results = inference_service.run_inference(video_source)

    # Log results
    print(f"Inference completed: {results}")
    
    # Return results as JSON (the results are still saved locally as images/YOLO format)
    return jsonify(results), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
