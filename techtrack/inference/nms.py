import cv2 as cv

class NMS:
    def __init__(self, nms_iou_threshold=0.4) -> None:
        """
        Initialize the NMS class and IoU threshold for filtering overlapping boxes. 

        :param nms_iou_threshold: IoU threshold to filter overlapping boxes.
        """
        self.nms_iou_threshold = nms_iou_threshold

    def filter(self, bboxes, class_ids, scores):
        """
        Apply NMS to filter overlapping bounding boxes.
        
        :param bboxes: List of bounding boxes (x, y, width, height).
        :param class_ids: List of class IDs corresponding to each bounding box.
        :param scores: List of confidence scores for each bounding box.
        :return: Lists of filtered bounding boxes, class IDs, and scores.
        """
        # Apply Non-Maximum Suppression using cv2.dnn.NMSBoxes()
        indices = cv.dnn.NMSBoxes(bboxes,
                                    scores,
                                    score_threshold=0.0,
                                    nms_threshold=self.nms_iou_threshold)
        filtered_bboxes = []
        filtered_class_ids = []
        filtered_scores = []

        if len(indices) > 0:
            for i in indices.flatten():
                filtered_bboxes.append(bboxes[i])
                filtered_class_ids.append(class_ids[i])
                filtered_scores.append(scores[i])

        return filtered_bboxes, filtered_class_ids, filtered_scores