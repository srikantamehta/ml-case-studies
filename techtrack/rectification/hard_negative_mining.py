import numpy as np
import os
import cv2 as cv

class HardNegativeMining:
    
    def __init__(self, iou_threshold=0.5, lambda_bb=1, lambda_obj=1, lambda_cls=1, lambda_no_obj=0.5):
        """
        Initialize the Hard Negative Mining module.
        :param iou_threshold: Threshold for IoU to consider a match.
        :param lambda_bb: Weight for bounding box loss.
        :param lambda_obj: Weight for objectness loss.
        :param lambda_cls: Weight for classification loss.
        :param lambda_no_obj: Weight for no-object loss.
        """
        self.iou_threshold = iou_threshold
        self.lambda_bb = lambda_bb
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_no_obj = lambda_no_obj

    # IoU calculation function
    def calculate_iou(self, boxA, boxB):
        """
        Calculate IoU between two bounding boxes in (x_min, y_min, width, height) format.

        :param boxA: First bounding box [x_min, y_min, width, height].
        :param boxB: Second bounding box [x_min, y_min, width, height].
        :return: Intersection over Union (IoU) score.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        # Compute the area of intersection
        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        interArea = inter_width * inter_height

        # Compute the area of both bounding boxes
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        # Compute the Intersection over Union (IoU)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    # Function to convert YOLO format to (x_min, y_min, width, height)
    def convert_yolo_to_bbox(self, yolo_box, image_width, image_height):
        """
        Convert YOLO format (center_x, center_y, width, height) to (x_min, y_min, width, height).
        
        :param yolo_box: YOLO format bounding box [center_x, center_y, width, height] (values are relative to image size).
        :param image_width: Original image width.
        :param image_height: Original image height.
        :return: Bounding box in (x_min, y_min, width, height) format.
        """
        center_x, center_y, bbox_w, bbox_h = yolo_box
        center_x *= image_width
        center_y *= image_height
        bbox_w *= image_width
        bbox_h *= image_height

        x_min = int(center_x - bbox_w / 2)
        y_min = int(center_y - bbox_h / 2)
        return [x_min, y_min, int(bbox_w), int(bbox_h)]

    def read_yolo_labels(self, label_file):
        """
        Read YOLO format labels from a text file.
        :param label_file: Path to the YOLO format text file.
        :return: List of bounding boxes in YOLO format. Supports both ground truth (5 values) and prediction (6+ values) formats.
        """
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) == 5:
                    # Ground truth labels (class_id, x_center, y_center, width, height)
                    class_id, x_center, y_center, width, height = map(float, values)
                    labels.append((int(class_id), x_center, y_center, width, height))
                elif len(values) >= 6:
                    # Prediction labels (class_id, x_center, y_center, width, height, score, [optional class scores])
                    class_id, x_center, y_center, width, height, score = map(float, values[:6])
                    class_scores = list(map(float, values[6:])) if len(values) > 6 else []
                    labels.append((int(class_id), x_center, y_center, width, height, score, class_scores))
                else:
                    raise ValueError(f"Unexpected format in label file: {label_file}. Expected 5 or 6+ values per line.")
        return labels

    def compute_loss(self, predictions, annotations, image_width, image_height):
        """
        Compute the loss between a set of predictions and ground truth annotations for an image.
        
        :param predictions: List of predicted bounding boxes in YOLO format 
                            (class_id, center_x, center_y, width, height, confidence, [class_scores]).
        :param annotations: List of ground truth bounding boxes in YOLO format 
                            (class_id, center_x, center_y, width, height).
        :param image_width: Width of the original image.
        :param image_height: Height of the original image.
        :return: Average total loss for all predictions.
        """
        losses = []
        if len(predictions) > 0:
            for pred_label in predictions:
                pred_box = self.convert_yolo_to_bbox(pred_label[1:5], image_width, image_height)
                pred_class_scores = pred_label[6]  
                
                for gt_label in annotations:
                    gt_box = self.convert_yolo_to_bbox(gt_label[1:], image_width, image_height)
                    iou = self.calculate_iou(gt_box, pred_box)
                    total_pred_loss = 0

                    if iou > self.iou_threshold:
                        # Bounding box loss (mean squared error)
                        tx, ty, tw, th = pred_label[1:5]  # YOLO prediction (center_x, center_y, width, height)
                        gt_tx, gt_ty, gt_tw, gt_th = gt_label[1:]  # Ground truth (center_x, center_y, width, height)
                        bbox_loss = ((tx - gt_tx)**2 + (ty - gt_ty)**2 + (tw - gt_tw)**2 + (th - gt_th)**2)
                        total_pred_loss += self.lambda_bb * bbox_loss

                        # Classification loss: Cross-entropy summed over all classes
                        num_classes = len(pred_class_scores)  # Number of classes inferred from the prediction
                        true_class = np.zeros(num_classes)
                        true_class[int(gt_label[0])] = 1  # One-hot encode the ground truth class
                        
                        predicted_class_probs = np.array(pred_class_scores)
                        cls_loss = -np.sum(true_class * np.log(predicted_class_probs + 1e-9))  # Cross-entropy loss
                        total_pred_loss += self.lambda_cls * cls_loss

                        # Objectness loss: Mean Square Error
                        obj_loss = (1 - pred_label[5])**2  # Object confidence at index 5
                        total_pred_loss += self.lambda_obj * obj_loss

                    else:
                        # No-object loss (when IoU is less than threshold)
                        obj_loss = (0 - pred_label[5])**2
                        total_pred_loss += self.lambda_no_obj * obj_loss

                    losses.append(total_pred_loss)
        if len(predictions)==0 and len(annotations)>0:
            total_pred_loss = 0
            for annotation in annotations:
                no_obj_loss = (1-0)**2
                total_pred_loss+= self.lambda_no_obj*no_obj_loss
            losses.append(total_pred_loss)
            
        # Return the average loss for all predictions
        return np.mean(losses) if len(losses) > 0 else 0
    
    def count_misclassifications_per_class(self, predictions, annotations, image_width, image_height):
        """
        Count the number of misclassifications per class for an image by comparing every prediction 
        with all annotations. If IoU is above threshold and classes don't match, it's counted as a misclassification.

        :param predictions: List of predicted bounding boxes in YOLO format 
                            (class_id, center_x, center_y, width, height, confidence, [class_scores]).
        :param annotations: List of ground truth bounding boxes in YOLO format 
                            (class_id, center_x, center_y, width, height).
        :param image_width: Width of the original image.
        :param image_height: Height of the original image.
        :return: A dictionary with class_ids as keys and misclassification counts as values.
        """
        misclassifications = {}

        # Loop through each prediction
        for pred_label in predictions:
            pred_class_id = pred_label[0]
            pred_box = self.convert_yolo_to_bbox(pred_label[1:5], image_width, image_height)

            # Compare prediction with every annotation
            for gt_label in annotations:
                gt_class_id = gt_label[0]
                gt_box = self.convert_yolo_to_bbox(gt_label[1:], image_width, image_height)

                # Calculate IoU between the prediction and ground truth
                iou = self.calculate_iou(pred_box, gt_box)

                # If IoU is above threshold and the class does not match, record misclassification
                if iou > self.iou_threshold and pred_class_id != gt_class_id:
                    if gt_class_id not in misclassifications:
                        misclassifications[gt_class_id] = 0
                    misclassifications[gt_class_id] += 1

        return misclassifications

    def sample_hard_negatives(self, prediction_dir, annotation_image_dir, num_samples):
        """
        Sample the top N hard negatives from the prediction and annotation directories.
        
        :param prediction_dir: Directory containing predicted YOLO bounding boxes (.txt).
        :param annotation_image_dir: Directory containing both the annotations (.txt) and the images (.jpg).
        :param num_samples: Number of hard negatives to return.
        :return: List of hard negatives (sorted by loss).
        """
        hard_negatives = []

        # Loop through the predictions and annotations
        for pred_file in os.listdir(prediction_dir):
            pred_path = os.path.join(prediction_dir, pred_file)
            ann_path = os.path.join(annotation_image_dir, pred_file)  # Annotations are in the same directory
            img_path = os.path.join(annotation_image_dir, pred_file.replace('.txt', '.jpg'))  # Corresponding image

            # Check if the corresponding annotation and image exist
            if os.path.exists(ann_path) and os.path.exists(img_path):
                # Read predictions and annotations
                predictions = self.read_yolo_labels(pred_path)
                annotations = self.read_yolo_labels(ann_path)

                # Read the image to get its dimensions
                image = cv.imread(img_path)
                image_height, image_width = image.shape[:2]

                # Compute the loss for this image using the whole set of predictions and annotations
                loss = self.compute_loss(predictions, annotations, image_width, image_height)

                # Append the file name and the loss
                hard_negatives.append((pred_file, loss))

        # Sort by loss and return top num_samples hard negatives
        hard_negatives = sorted(hard_negatives, key=lambda x: x[1], reverse=True)
        return hard_negatives[:num_samples]
    
    def sample_misclassifications(self, prediction_dir, annotation_image_dir):
        """
        Sample the top N classes with the most misclassifications from the prediction and annotation directories.
        
        :param prediction_dir: Directory containing predicted YOLO bounding boxes (.txt).
        :param annotation_image_dir: Directory containing both the annotations (.txt) and the images (.jpg).
        :param num_samples: Number of classes with the most misclassifications to return.
        :return: List of tuples (class_id, misclassification_count) sorted by the most misclassifications.
        """
        class_misclassifications = {}

        # Loop through the predictions and annotations
        for pred_file in os.listdir(prediction_dir):
            pred_path = os.path.join(prediction_dir, pred_file)
            ann_path = os.path.join(annotation_image_dir, pred_file)  
            img_path = os.path.join(annotation_image_dir, pred_file.replace('.txt', '.jpg'))  # Corresponding image

            # Check if the corresponding annotation and image exist
            if os.path.exists(ann_path) and os.path.exists(img_path):
                # Read predictions and annotations
                predictions = self.read_yolo_labels(pred_path)
                annotations = self.read_yolo_labels(ann_path)

                # Read the image to get its dimensions
                image = cv.imread(img_path)
                image_height, image_width = image.shape[:2]

                # Get misclassifications for this image
                misclassifications = self.count_misclassifications_per_class(predictions, annotations, image_width, image_height)

                # Update the overall misclassification count per class
                for class_id, count in misclassifications.items():
                    if class_id not in class_misclassifications:
                        class_misclassifications[class_id] = 0
                    class_misclassifications[class_id] += count

        # Sort the classes by misclassification count and return the top N
        sorted_misclassifications = sorted(class_misclassifications.items(), key=lambda x: x[1], reverse=True)
        return sorted_misclassifications
