import cv2 as cv
import numpy as np
import os

class Model:
    def __init__(self, weights_file, config_file, names_file) -> None:
        """
        Initialize the Model class by loading the YOLO model, weights, and class names.
    
        :param weights: Path to the YOLO weights file.
        :param config: Path to the YOLO config file.
        :param names: Path to the file containing class names.
        """
        # Load YOLO model
        self.net = cv.dnn.readNet(weights_file, config_file)
        # Get all layer names 
        self.layer_names = self.net.getLayerNames()
        # Get the output layers
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        # Load class names
        with open(names_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def predict(self, preprocessed_frame):
        """
        Perform object detection on the preprocessed frame using the YOLO model.

        :param preprocessed_frame: Input image frame after processing.
        :return: Raw model output predictions (bounding boxes, scores, and class IDs)
        """
        # Get the original size of the input frame
        original_size = preprocessed_frame.shape[:2]
        # Prepare the frame for input into YOLO
        blob = cv.dnn.blobFromImage(preprocessed_frame,
                                    scalefactor=1/255,
                                    size=(416,416),
                                    mean=(0,0,0),
                                    swapRB=True,
                                    crop=False)
        self.net.setInput(blob)
        # Run forward pass to get predictions
        outputs = self.net.forward(self.output_layers)
        return outputs, original_size
    
    def post_process(self, predict_output, original_size, score_threshold=0.5):
        """
        Post-process the model predictions by filtering out low-confidence detections (below score_threshold).

        :param predict_output: Raw model outputs from the predict function.
        :param score_threshold: Confidence score threshold.
        :return: Tuple of (yolo_bboxes, class_ids, scores) lists, where yolo_bboxes are in YOLO format.
        """
        yolo_bboxes = []
        class_ids = []
        scores = []
        original_height, original_width = original_size
        fixed_size = (416, 416)

        # Iterate over each feature map in the output
        for feature_maps in predict_output:
            for detection in feature_maps:
                # Extract the bounding box coordinates and the objectness score
                boxes = detection[:4]
                score = detection[4]
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]

                if confidence > score_threshold:
                    # Convert bounding box from (center_x, center_y, width, height) to pixel values
                    center_x, center_y, w, h = boxes * np.array([fixed_size[1], fixed_size[0], fixed_size[1], fixed_size[0]])

                    # Convert from the fixed size (416x416) to the original image size
                    center_x = (center_x / fixed_size[1]) * original_width
                    center_y = (center_y / fixed_size[0]) * original_height
                    w = (w / fixed_size[1]) * original_width
                    h = (h / fixed_size[0]) * original_height

                    # Normalize the bounding box with respect to the original image size
                    norm_center_x = center_x / original_width
                    norm_center_y = center_y / original_height
                    norm_width = w / original_width
                    norm_height = h / original_height

                    # Append to results in YOLO format
                    yolo_bboxes.append([norm_center_x, norm_center_y, norm_width, norm_height])
                    class_ids.append(class_id)
                    scores.append(float(confidence))

        return yolo_bboxes, class_ids, scores
    

    def save_predictions(self, outputs, original_size, output_path, image_filename, score_threshold=0):
        """
        Save the predictions in YOLO format to a text file, including class scores.

        :param outputs: Raw model outputs from the predict function.
        :param original_size: Original size of the image (height, width).
        :param score_threshold: Confidence score threshold for filtering predictions.
        :param output_path: Path to the directory where results will be saved.
        :param image_filename: Name of the image (without extension) to use for saving the result file.
        """
        output_file = os.path.join(output_path, f"{image_filename}.txt")
        original_height, original_width = original_size
        fixed_size = (416, 416)

        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Write the results to the text file
        with open(output_file, 'w') as f:
            # Iterate over each feature map in the output
            for feature_maps in outputs:
                for detection in feature_maps:
                    # Extract the bounding box coordinates and the objectness score
                    boxes = detection[:4]
                    score = detection[4]
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]

                    # Filter by score threshold
                    if confidence > score_threshold:
                        # Convert bounding box from (center_x, center_y, width, height) to pixel values
                        center_x, center_y, w, h = boxes * np.array([fixed_size[1], fixed_size[0], fixed_size[1], fixed_size[0]])

                        # Convert from the fixed size (416x416) to the original image size
                        center_x = (center_x / fixed_size[1]) * original_width
                        center_y = (center_y / fixed_size[0]) * original_height
                        w = (w / fixed_size[1]) * original_width
                        h = (h / fixed_size[0]) * original_height

                        # Normalize the bounding box with respect to the original image size
                        norm_center_x = center_x / original_width
                        norm_center_y = center_y / original_height
                        norm_width = w / original_width
                        norm_height = h / original_height

                        # Convert class_scores to a string
                        class_score_str = " ".join(f"{s:.6f}" for s in class_scores)

                        # Write in YOLO format: class_id center_x center_y width height (all normalized) + score + class scores
                        f.write(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f} {confidence:.2f} {class_score_str}\n")
