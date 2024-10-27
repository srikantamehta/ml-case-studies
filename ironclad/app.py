from flask import Flask, request, jsonify
from pipeline import Pipeline
import os
import time
from datetime import datetime

app = Flask(__name__)

# Initialize the pipeline
pipeline = Pipeline(pretrained='vggface2', device='cpu', index_type='IVF')

# Define the gallery and catalog paths
GALLERY_DIR = "ironclad/storage/multi_image_gallery"
FAISS_PATH = "ironclad/storage/catalog/vggface2_IVF.index"
METADATA_PATH = "ironclad/storage/catalog/vggface2_IVF_metadata.pkl"

# Store search history and metrics
search_history = []
query_times = []
start_time = datetime.now()

@app.route('/add', methods=['POST'])
def add_identity():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        save_path = os.path.join(GALLERY_DIR, file.filename)
        file.save(save_path)

        # Encode and add new identity
        embedding_vector = pipeline._Pipeline__encode(save_path).reshape(1, -1)
        metadata = {'name': os.path.basename(save_path), 'filename': file.filename}
        pipeline.index.add_embeddings(embedding_vector, metadata=[metadata])
        
        # Save updated FAISS index and metadata
        pipeline._Pipeline__save_embeddings(faiss_path=FAISS_PATH, metadata_path=METADATA_PATH)

        return jsonify({"message": f"Identity {file.filename} added to the gallery and indexed."}), 200

@app.route('/identify', methods=['POST'])
def identify():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        probe_image_path = os.path.join("ironclad/temp", file.filename)
        if not os.path.exists("ironclad/temp"):
            os.makedirs("ironclad/temp")
        file.save(probe_image_path)

        k = request.args.get('k', default=5, type=int)
        
        # Measure query response time
        start_query_time = time.time()
        results = pipeline.search_gallery(probe_image_path, k=k, faiss_path=FAISS_PATH, metadata_path=METADATA_PATH)
        end_query_time = time.time()
        
        # Calculate and record query time
        query_time = end_query_time - start_query_time
        query_times.append(query_time)

        # Add to search history
        search_history.append({
            "probe_image": file.filename,
            "results": results,
            "query_time": query_time
        })

        return jsonify({"probe_image": file.filename, "results": results, "query_time": query_time}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    # Calculate average response time
    avg_response_time = sum(query_times) / len(query_times) if query_times else 0
    
    # Calculate uptime
    uptime = (datetime.now() - start_time).total_seconds()
    
    # Retrieve gallery size from FAISS index metadata 
    gallery_size = len(pipeline.index.metadata)
    
    metrics_data = {
        "average_query_response_time": avg_response_time,
        "uptime_seconds": uptime,
        "total_queries": len(query_times),
        "gallery_size": gallery_size
    }

    return jsonify(metrics_data), 200

@app.route('/history', methods=['GET'])
def history():
    return jsonify(search_history), 200

if __name__ == '__main__':
    if not os.path.exists(FAISS_PATH):
        print("Precomputing gallery embeddings...")
        pipeline._Pipeline__precompute(GALLERY_DIR)
        pipeline._Pipeline__save_embeddings(faiss_path=FAISS_PATH, metadata_path=METADATA_PATH)
    app.run(host='0.0.0.0', port=5000)
