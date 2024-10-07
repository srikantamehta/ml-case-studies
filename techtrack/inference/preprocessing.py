import cv2 as cv

class VideoPreprocessor:

    def __init__(self) -> None:
        """
        Initialize the VideoPreprocessor class
        """
        pass

    def capture_video(self, source, drop_rate):
        """
        Generator function that captures frames from the video or UDP stream and yields every Nth frame based on drop_rate.

        :param source: Path to the UDP stream URL (e.g., 'udp://127.0.0.1:23000').
        :param drop_rate: Number of frames to drop (i.e., keep every Nth frame).
        :yield: Frame (numpy array) from the video or stream.
        """
        cap = cv.VideoCapture(source, cv.CAP_FFMPEG)  # Open the UDP stream
        
        frame_count = 0 
        if not cap.isOpened():  # Check if the video source was opened successfully
            print(f"Error: Cannot open video source {source}")
            return
        while cap.isOpened():  # Process frames while the stream is open
            ret, frame = cap.read()  # Read the next frame
            if not ret:
                print("Stream or video has ended")
                break  # Exit loop if no more frames
            if frame_count % drop_rate == 0:
                yield frame  # Yield the frame every Nth frame based on drop_rate
            frame_count += 1
        cap.release()

    def capture_video_from_file(self, filename, drop_rate):
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