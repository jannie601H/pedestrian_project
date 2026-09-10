from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

        # COCO class IDs
        self.target_classes = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

    def detect(self, frame):
        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in self.target_classes:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": self.target_classes[class_id],
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections