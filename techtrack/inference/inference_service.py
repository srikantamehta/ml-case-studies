import cv2 as cv
import os
from .preprocessing import VideoPreprocessor
from .object_detection import Model
from .nms import NMS

class InferenceService:
    def __init__(self, config):
        """
        Initialize the Inference Service with the given configuration.
        :param config: Configuration dictionary.
        """
        self.config = config
        self.video_preprocessor = VideoPreprocessor()
        self.set_model_paths()  # Set the paths to the model files (weights, config, names)
        self.model = Model(self.weights_file, self.config_file, self.names_file)
        self.nms = NMS(self.config['nms_iou_threshold'])
        self.output_dir = self.config['output_dir']  # Directory to save results

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def set_model_paths(self):
        """
        Automatically set the paths for weights, config, and names files based on the model name.
        Searches by file extensions in the model directory.
        """
        model_name = self.config['model_name']
        base_path = f"/app/storage/{model_name}"  # Docker container path for models

        # Search for files with .weights, .cfg, and .names extensions
        self.weights_file = self.find_file_with_extension(base_path, '.weights')
        self.config_file = self.find_file_with_extension(base_path, '.cfg')
        self.names_file = self.find_file_with_extension(base_path, '.names')

        # Check if all files were found
        if not all([self.weights_file, self.config_file, self.names_file]):
            raise FileNotFoundError(f"Model files (weights, config, or names) for {model_name} not found in {base_path}")

    def find_file_with_extension(self, directory, extension):
        """
        Find a file in a directory with the specified extension.
        :param directory: The directory to search in.
        :param extension: The file extension to search for (e.g., '.weights').
        :return: Full file path if found, else None.
        """
        for file_name in os.listdir(directory):
            if file_name.endswith(extension):
                return os.path.join(directory, file_name)
        return None
    
    def run_inference(self, video_source):
        """
        Run the inference on a given video source (UDP stream or file).
        :param video_source: Path to the video file or UDP stream URL.
        :return: Processed results (bounding boxes, class IDs, scores) for each frame.
        """
        frame_index = 0  # For naming output images
        results = []  # To store results for each frame
        for frame in self.video_preprocessor.capture_video(video_source, self.config['drop_rate']):
            # Run the object detection model on the preprocessed frame
            predictions, original_size = self.model.predict(frame)

            # Post-process the predictions
            bboxes, class_ids, scores = self.model.post_process(predictions, original_size, self.config['score_threshold'])

            # Apply NMS if specified in the configuration
            if self.config['apply_nms']:
                bboxes, class_ids, scores = self.nms.filter(bboxes, class_ids, scores)

            # Prepare results for this frame 
            frame_results = {
                'frame_index': frame_index,
                'bboxes': [list(map(float, bbox)) for bbox in bboxes],  # Ensure bbox is list of floats
                'class_ids': [int(class_id) for class_id in class_ids],  # Convert to int
                'scores': [float(score) for score in scores]  # Convert to float
            }
            results.append(frame_results)

            # Save results (image + bounding boxes + predictions)
            self.save_results(frame, bboxes, class_ids, scores, frame_index)
            frame_index += 1

        return results  # Return the collected results for all frames

    def save_results(self, frame, bboxes, class_ids, scores, frame_index):
        """
        Save the processed results: store the image with detections and save the bounding box data in YOLO format.
        :param frame: Original video frame.
        :param bboxes: List of bounding boxes (in YOLO format, normalized).
        :param class_ids: List of class IDs.
        :param scores: List of confidence scores.
        :param frame_index: Index of the frame being processed (for naming output files).
        """
        # Save image with bounding boxes
        output_image_path = os.path.join(self.output_dir, f"frame_{frame_index}.jpg")
        self.draw_bboxes(frame, bboxes, class_ids, scores)  # Convert YOLO to pixel coordinates for drawing
        cv.imwrite(output_image_path, frame)

        # Save predictions in YOLO format (boxes are already in YOLO format)
        output_predictions_path = os.path.join(self.output_dir, f"frame_{frame_index}.txt")
        with open(output_predictions_path, 'w') as f:
            for bbox, class_id, score in zip(bboxes, class_ids, scores):
                cx, cy, bw, bh = bbox  # YOLO format (normalized)
                # Write to file in YOLO format (class_id, center_x, center_y, width, height, score)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.2f}\n")

    def draw_bboxes(self, frame, bboxes, class_ids, scores):
        """
        Draw bounding boxes on the frame for visualization.
        :param frame: Original video frame.
        :param bboxes: List of bounding boxes in YOLO format (normalized).
        :param class_ids: List of class IDs.
        :param scores: List of scores.
        """
        h, w = frame.shape[:2]  # Get original image size
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            cx, cy, bw, bh = bbox  # YOLO format: (center_x, center_y, width, height)

            # Convert YOLO format to pixel coordinates
            x1 = int((cx - bw / 2) * w)  # Top-left x
            y1 = int((cy - bh / 2) * h)  # Top-left y
            x2 = int((cx + bw / 2) * w)  # Bottom-right x
            y2 = int((cy + bh / 2) * h)  # Bottom-right y

            label = f"{self.model.classes[class_id]}: {score:.2f}"
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
