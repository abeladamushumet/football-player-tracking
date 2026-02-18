import os
import cv2


class VideoLoader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def load(self):
        """
        Load the video file and return OpenCV VideoCapture object.
        """
        # Check if file exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"❌ Video not found: {self.video_path}")

        # Open video
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError("❌ Could not open video file.")

        print("✅ Video loaded successfully.")
        self.print_video_info()

        return self.cap

    def print_video_info(self):
        """
        Print basic video properties.
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("📹 Video Information:")
        print(f"   Resolution: {width} x {height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
