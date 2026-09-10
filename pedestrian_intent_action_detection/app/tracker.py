from ultralytics import YOLO


class Tracker:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

        self.target_classes = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracks = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in self.target_classes:
                    continue

                # tracking ID가 없는 경우 제외
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                tracks.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": self.target_classes[class_id],
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        return tracks