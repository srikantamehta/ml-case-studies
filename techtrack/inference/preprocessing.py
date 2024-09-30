import cv2 as cv

class VideoPreprocessor:

    def __init__(self) -> None:
        """
        Initialize the VideoPreprocessor class
        """
        pass

    def capture_video(self, filename, drop_rate):
        """
        Generator function that captures frames from the video file and yields every Nth frame based on drop_rate.

        :param filename: Path to the video file.
        :param drop_rate: Number of frames to drop (i.e., keep every Nth frame).
        :yield: Frame (numpy array) from the video.
        """
        cap = cv.VideoCapture(filename) # Open the video file
        frame_count = 0 
        if not cap.isOpened(): # Check if video file was opened successfully
            print(f"Error: Cannot open video file {filename}")
            return
        while cap.isOpened(): # Process frames while the video is open
            ret, frame = cap.read()
            if not ret:
                print("Video has ended")
                break  # Exit loop if no more frames
            if frame_count % drop_rate == 0:
                yield frame
            frame_count+=1
        cap.release()