# Inference Service Container Setup

This project provides an inference service to run object detection on video files or UDP streams using YOLO models. The service is containerized using Docker.

## Project Structure

- `docker-compose.yml` - Docker Compose configuration to set up the container.
- `Dockerfile` - Instructions to create the Docker container.
- `app.py` - Flask API to run inference.
- `inference/` - Contains modules for preprocessing, object detection, and the inference service.
- `rectification/` - Contains modules for hard negative mining, and augmentation.
- `storage/` - Directory for model files, predictions, and outputs.
- `notebooks/` - Contains the analysis notebooks

## Prerequisites

- Docker installed on your machine.
- Docker Compose installed on your machine.

## Setting Up the Container

1. Clone this repository to your local machine.
2. Ensure the required model files are in the `storage/{yolo_model}/` directory. You will need:
- YOLO weights file (.weights)
- YOLO configuration file (.cfg)
- Class names file (.names)

3. Build and start the Docker container using Docker Compose:
    ```
    docker-compose up -d
    ```

This will:
- Build the container image for the inference service.
- Set up necessary networking for accessing the UDP stream and exposing the Flask API.

## Testing the API

### Test with a Video File Source

To run inference on a video file, use the following command:

```bash
curl -X POST http://localhost:5001/run_inference \
  -H "Content-Type: application/json" \
  -d '{
        "video_source": "storage/test_videos/worker-zone-detection.mp4",
        "config": {
          "model_name": "yolo_model_1",
          "drop_rate": 20,
          "score_threshold": 0.5,
          "apply_nms": true,
          "nms_iou_threshold": 0.4,
          "output_dir": "/app/storage/test_output"
        }
      }'
```

### Test with a UDP File Stream

To run inference on a UDP stream, follow these steps:

1. Start the UDP Stream on your local machine, run the following command to stream a test video via UDP:
```
ffmpeg -re -i techtrack/storage/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```
2. Run the Inference in another terminal, use the following curl command to run inference on the UDP stream:

```bash
curl -X POST http://localhost:5001/run_inference \
     -H "Content-Type: application/json" \
     -d '{
           "video_source": "udp://127.0.0.1:23000",
           "config": {
             "model_name": "yolo_model_1",
             "drop_rate": 20,
             "score_threshold": 0.5,
             "apply_nms": true,
             "nms_iou_threshold": 0.4,
             "output_dir": "/app/storage/test_output"
           }
         }'
```

## Troubleshooting

- Port Conflicts: Make sure ports 5001 (for the API) and 23000 (for the UDP stream) are not used by other services on your machine.
- Network Issues: If the UDP stream inference is not working, consider using Docker's host network mode for better compatibility with the local network.