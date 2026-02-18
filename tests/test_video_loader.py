from src.preprocessing.video_loader import VideoLoader

# Your video file
video_path = "data/raw/1.mp4"

# Load video
loader = VideoLoader(video_path)
cap = loader.load()

# Release video when done
cap.release()
