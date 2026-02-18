import cv2
import os
import json
from ultralytics import YOLO
from src.homography.field_mapping import FieldMapper

class Tracker:
    def __init__(self):
        # Input raw video
        self.video_path = "data/raw/1.mp4"  

        # Output paths
        self.output_video_path = "outputs/videos/tracking_output.avi"  # use .avi on Windows
        self.output_json_path = "outputs/tracking_output.json"
        self.output_field_json = "outputs/tracking_field_coords.json"

        os.makedirs("outputs/videos", exist_ok=True)

        # Load YOLOv8 medium (CPU-friendly)
        self.model = YOLO("models/detection/yolov8/yolov8m.pt")

        # Initialize field mapper
        self.mapper = FieldMapper()

        # TODO: Replace these with actual points from your video
        image_points = [
            (100, 200),     # top-left
            (1800, 220),    # top-right
            (150, 900),     # bottom-left
            (1750, 880)     # bottom-right
        ]

        # Real-world field coordinates in meters (FIFA 105x68)
        field_points = [
            (0, 0),
            (105, 0),
            (0, 68),
            (105, 68)
        ]

        self.mapper.set_correspondences(image_points, field_points)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # fallback if width/height/fps are zero
        if width == 0 or height == 0:
            width, height = 1280, 720
        if fps == 0:
            fps = 25

        print(f"Video info -> width: {width}, height: {height}, fps: {fps}")

        # Use XVID codec for Windows
        out = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width, height)
        )

        tracking_results = []
        field_results = []
        frame_id = 0

        print("📹 Starting tracking...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0]  # only person
            )[0]

            frame_tracks = []
            frame_field = []

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)

                    # Draw bbox + ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame,
                                f"ID {track_id}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,255,0), 2)

                    frame_tracks.append({
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2]
                    })

                    # Map bbox to field coordinates
                    field_x, field_y = self.mapper.map_bbox_to_field([x1, y1, x2, y2])
                    frame_field.append({
                        "track_id": track_id,
                        "field_pos": [field_x, field_y]
                    })

            tracking_results.append({
                "frame_id": frame_id,
                "tracks": frame_tracks
            })

            field_results.append({
                "frame_id": frame_id,
                "field_tracks": frame_field
            })

            out.write(frame)
            frame_id += 1

        cap.release()
        out.release()

        # Save JSON outputs
        with open(self.output_json_path, "w") as f:
            json.dump(tracking_results, f, indent=4)

        with open(self.output_field_json, "w") as f:
            json.dump(field_results, f, indent=4)

        print("✅ Tracking complete!")
        print("🎥 Video saved at:", self.output_video_path)
        print("📄 JSON saved at:", self.output_json_path)
        print("📄 Field coordinates saved at:", self.output_field_json)


if __name__ == "__main__":
    tracker = Tracker()
    tracker.run()
