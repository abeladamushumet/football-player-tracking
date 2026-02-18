import os
import cv2
import json
from ultralytics import YOLO
from src.preprocessing.video_loader import VideoLoader

class PlayerDetectorCPU:
    def __init__(self, model_path: str, output_dir: str, conf_thresh: float = 0.4, skip_frames: int = 5, resize_width: int = 640):
        """
        CPU-friendly YOLOv8 player detector with frame skipping and resizing
        :param model_path: path to YOLOv8 weights
        :param output_dir: folder to save output video and JSON
        :param conf_thresh: detection confidence threshold
        :param skip_frames: process every nth frame
        :param resize_width: width to resize frames (maintains aspect ratio)
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.conf_thresh = conf_thresh
        self.skip_frames = skip_frames
        self.resize_width = resize_width

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print(f"🔹 Loading YOLO model from {self.model_path} (CPU mode)...")
        self.model = YOLO(self.model_path)
        print("✅ YOLO model loaded.")

    def detect_video(self, video_path: str, output_video_name="output.mp4"):
        loader = VideoLoader(video_path)
        cap = loader.load()

        # Original size
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Resize proportionally
        scale = self.resize_width / orig_width
        resize_height = int(orig_height * scale)

        out_path = os.path.join(self.output_dir, output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (self.resize_width, resize_height))

        results_list = []
        frame_count = 0
        saved_count = 0

        print(f"📹 Processing {total_frames} frames with skip={self.skip_frames} and resize={self.resize_width}x{resize_height}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames
            if frame_count % self.skip_frames != 0:
                continue

            # Resize for faster CPU processing
            frame_resized = cv2.resize(frame, (self.resize_width, resize_height))

            # Run YOLO detection
            results = self.model.predict(frame_resized, conf=self.conf_thresh, verbose=False)[0]

            # Draw boxes and save detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Draw rectangle
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_resized,
                    f"{cls_id}:{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # Save detection info
                results_list.append({
                    "frame": frame_count,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

            # Write frame to output video
            out_video.write(frame_resized)
            saved_count += 1

            # Print progress every 100 processed frames
            if saved_count % 100 == 0:
                print(f"Processed {saved_count} frames...")

        cap.release()
        out_video.release()

        # Save JSON
        json_path = os.path.join(self.output_dir, "detections.json")
        with open(json_path, "w") as f:
            json.dump(results_list, f, indent=4)

        print(f"✅ Detection finished. Output video: {out_path}")
        print(f"✅ Detection JSON: {json_path}")


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    model_path = "models/detection/yolov8/yolov8n.pt"  # YOLOv8 nano
    video_path = "data/raw/1.mp4"
    output_dir = "outputs/videos"

    detector = PlayerDetectorCPU(
        model_path=model_path,
        output_dir=output_dir,
        conf_thresh=0.4,
        skip_frames=5,       # process 1 frame every 5
        resize_width=640     # resize frames to 640px width
    )
    detector.detect_video(video_path)
