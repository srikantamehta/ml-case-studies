from flask import Flask, request, jsonify
from inference.inference_service import InferenceService
import time
import psutil
import numpy as np

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

    # Track start time for system latency measurement
    start_time = time.time()

    # Run inference
    print(f"Running inference on {video_source}")
    results = inference_service.run_inference(video_source)

    # Track end time and calculate latency
    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency for inference: {latency:.2f} seconds")

    # Calculate frame processing rate (FPS)
    total_time = end_time - start_time
    total_frames = len(results)
    fps = total_frames / total_time if total_time > 0 else 0
    print(f"Frame Processing Rate: {fps:.2f} FPS")

    # Get resource utilization metrics
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage}%")

    # Calculate average detection confidence
    confidences = [frame_result['scores'] for frame_result in results if 'scores' in frame_result]
    avg_confidence = np.mean([score for frame_scores in confidences for score in frame_scores]) if confidences else 0
    print(f"Average detection confidence: {avg_confidence:.2f}")

    # Construct a metrics response
    metrics = {
        'latency': round(latency, 2),
        'fps': round(fps, 2),
        'average_confidence': round(avg_confidence, 2),
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'total_frames_processed': total_frames,
    }

    # Return metrics as JSON
    return jsonify(metrics), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
