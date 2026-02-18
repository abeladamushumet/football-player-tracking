import os
import cv2
from video_loader import VideoLoader  # assuming you are running from src/preprocessing

class FrameExtractor:
    def __init__(self, video_path: str, output_dir: str, skip_frames: int = 1):
        """
        :param video_path: path to input video
        :param output_dir: folder where frames will be saved
        :param skip_frames: save every nth frame (1 = save all)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.skip_frames = skip_frames

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_frames(self):
        # Load video using VideoLoader
        loader = VideoLoader(self.video_path)
        cap = loader.load()

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Save every nth frame
            if frame_count % self.skip_frames == 0:
                frame_filename = os.path.join(
                    self.output_dir, f"frame_{frame_count:05d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

        cap.release()
        print(f"✅ Finished extracting {saved_count} frames to {self.output_dir}")
        return saved_count


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    video_path = "data/raw/1.mp4"  # your video
    output_dir = "data/processed/frames"

    extractor = FrameExtractor(video_path, output_dir, skip_frames=5)  # save every 5th frame
    extractor.extract_frames()
