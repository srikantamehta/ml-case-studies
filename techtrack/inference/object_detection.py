import cv2 as cv
import numpy as np

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
        :retrun: Raw model output predictions (bounding boxes, scores, and class IDs)
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
        :return: Tuple of (bboxes, class_ids, scores) lists.
        """
        bboxes = []
        class_ids = []
        scores = []
        # Original image size before resizing
        original_height, original_width = original_size
        fixed_size = (416,416)
        # Iterate over each feature map in the output
        for feature_maps in predict_output:
            for detection in feature_maps:
                boxes = detection[:4]
                score = detection[4]
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]

                if confidence > score_threshold:
                    # Convert bounding box from (center_x, center_y, width, height) to (x, y, width, height)
                    center_x, center_y, w, h = boxes * np.array([fixed_size[1], fixed_size[0], fixed_size[1], fixed_size[0]])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Scale the bounding box back to the original image size
                    x = int(x * original_width / fixed_size[1])
                    y = int(y * original_height / fixed_size[0])
                    w = int(w * original_width / fixed_size[1])
                    h = int(h * original_height / fixed_size[0])

                    bboxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    scores.append(float(confidence))
        return bboxes, class_ids, scores
    
