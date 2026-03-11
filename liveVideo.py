from ultralytics import YOLO
import cv2
from collections import deque, Counter

WINDOW_SIZE = 5
MIN_HITS = 3

# Class IDs:
# 0 = logo
# 1 = fake symbol
# 2 = real symbol

CLASS_COLORS = {
    0: (255, 165, 0),   # Orange – logo
    1: (0, 0, 255),     # Red – fake
    2: (0, 255, 0),     # Green – real
}


model = YOLO("runs/detect/yolo11tr8/weights/best.pt")
cap = cv2.VideoCapture(0)

history = deque(maxlen=WINDOW_SIZE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        conf=0.45,
        iou=0.4,
        imgsz=512,
        device="mps",
        verbose=False
    )

    detections = results[0].boxes
    current_classes = []

    if detections is not None:
        for box in detections:
            current_classes.append(int(box.cls))

    history.append(current_classes)

    flat_history = [c for f in history for c in f]
    counts = Counter(flat_history)

    stable_classes = {
        cls for cls, count in counts.items() if count >= MIN_HITS
    }

    annotated = frame.copy()

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls)
            if cls_id not in stable_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = f"{model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Stable Symbol Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
