import cv2
import numpy as np
import os

class Augmentation:
    def __init__(self, output_dir):
        """
        Initialize the Augmentation Class.
        :parap output_dir: Directory where augmented images and their YOLO files will be saved.
        """
        self.output_dir = output_dir
        # Create directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def read_yolo_file(self, yolo_file_path):
        """
        Read YOLO formatted bounding boxes from a text file.
        :param yolo_file_path: Path to the YOLO format text file.
        :return: List of bounding boxes in YOLO format (class_id, center_x, center_y, width, height).
        """
        bboxes = []
        with open(yolo_file_path, 'r') as f:
            for line in f.readlines():
                class_id, cx, cy, bw, bh = map(float, line.strip().split())
                bboxes.append([int(class_id), cx, cy, bw, bh])
        return bboxes
    
    def save_yolo_file(self, bboxes, output_txt_path):
        """
        Save the YOLO formatted bounding boxes to a text file.
        :param bboxes: List of bounding boxes in YOLO format (class_id, center_x, center_y, width, height).
        :param output_txt_path: Path to save the new YOLO format text file.
        """
        with open(output_txt_path, 'w') as f:
            for bbox in bboxes:
                class_id, cx, cy, bw, bh = bbox

                # Write the values
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    
    def horizontal_flip(self, image, bboxes):
        """
        Apply horizontal flip to the input image and adjust bounding boxes (keeping normalized values).
        :param image: Input image (numpy array).
        :param bboxes: List of bounding boxes in YOLO format (class_id, center_x, center_y, width, height).
        :return: Horizontally flipped image and updated bounding boxes.
        """
        flipped_image = cv2.flip(image, 1)
        # Flip bounding boxes horizontally (keeping normalized values)
        flipped_bboxes = []
        for bbox in bboxes:
            class_id, cx, cy, bw, bh = bbox
            flipped_cx = 1 - cx  # Invert the center_x for horizontal flip
            flipped_bboxes.append([class_id, flipped_cx, cy, bw, bh])

        return flipped_image, flipped_bboxes


    def gaussian_blur(self, image, kernel_size=(15, 15)):
        """
        Apply Gaussian blur to the input image.
        :param image: Input image (numpy array).
        :param kernel_size: Kernel size for Gaussian blur.
        :return: Blurred image.
        """
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def resize(self, image, bboxes, new_size=(224, 224)):
        """
        Resize the input image. Bounding boxes remain the same since they are in normalized YOLO format.
        :param image: Input image (numpy array).
        :param bboxes: List of bounding boxes in YOLO format (class_id, center_x, center_y, width, height).
        :param new_size: Desired output size (width, height).
        :return: Resized image and unchanged bounding boxes.
        """
        resized_image = cv2.resize(image, new_size)
        # Bounding boxes are already normalized, so they remain unchanged
        return resized_image, bboxes

        
    def rotate(self, image, bboxes):
        """
        Rotate the input image by 90 degrees clockwise and adjust bounding boxes accordingly.
        :param image: Input image (numpy array).
        :param bboxes: List of bounding boxes in YOLO format (class_id, center_x, center_y, width, height).
        :return: Rotated image and updated bounding boxes (still normalized).
        """
        (h, w) = image.shape[:2]

        # Rotate image by 90 degrees (clockwise)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Adjust bounding boxes for 90-degree rotation
        rotated_bboxes = []
        for bbox in bboxes:
            class_id, cx, cy, bw, bh = bbox

            # Adjust bounding boxes for 90-degree rotation (keeping normalized values)
            rotated_cx = 1 - cy  # New center_x is 1 - old center_y
            rotated_cy = cx      # New center_y is old center_x
            rotated_bw = bh      # Swap width and height
            rotated_bh = bw

            rotated_bboxes.append([class_id, rotated_cx, rotated_cy, rotated_bw, rotated_bh])

        return rotated_image, rotated_bboxes

    def adjust_brightness(self, image, value=30):
        """
        Adjust the brightness of the input image.
        :param image: Input image (numpy array).
        :param value: Amount to adjust brightness.
        :return: Brightness adjusted image.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v + value, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, alpha=1.5):
        """
        Adjust the contrast of the input image.
        :param image: Input image (numpy array).
        :param alpha: Contrast control (1.0-3.0).
        :return: Contrast adjusted image.
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    def apply_augmentations(self, image_filename, yolo_filename):
        """
        Apply all augmentations to the image and its corresponding YOLO file.
        :param image_filename: Path to the original image file.
        :param yolo_filename: Path to the YOLO format text file.
        """
        image = cv2.imread(image_filename)
        bboxes = self.read_yolo_file(yolo_filename)
        h, w = image.shape[:2]  # Get original image size

        augmentations = [
            ('flipped', self.horizontal_flip),
            ('blurred', self.gaussian_blur),
            ('resized', self.resize),
            ('rotated', self.rotate),
            ('brightness', self.adjust_brightness),
            ('contrast', self.adjust_contrast)
        ]

        for aug_name, aug_func in augmentations:
            # Apply augmentation
            if aug_name == 'resized' or aug_name == 'rotated' or aug_name == 'flipped':
                aug_image, aug_bboxes = aug_func(image, bboxes)
            else:
                aug_image = aug_func(image)
                aug_bboxes = bboxes  # Bounding boxes remain unchanged

            # Save augmented image
            output_image_path = os.path.join(self.output_dir, f"{os.path.basename(image_filename).split('.')[0]}_{aug_name}.jpg")
            cv2.imwrite(output_image_path, aug_image)

            # Save YOLO file (whether bounding boxes are altered or not)
            output_txt_path = os.path.join(self.output_dir, f"{os.path.basename(yolo_filename).split('.')[0]}_{aug_name}.txt")
            self.save_yolo_file(aug_bboxes, output_txt_path)
