from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")
# Initialize tracker
tracker = DeepSort(max_age=30)

# Video path
cap = cv2.VideoCapture(r"C:\Users\HARINI M\Downloads\Guardianeye\traffic.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()
else:
    print("Video loaded successfully")

cv2.namedWindow("GuardianEye Demo", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break

    frame = cv2.resize(frame, (800, 600))

    # YOLO detection
    results = model(frame)

    detections = []
    for r in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, r)
        detections.append(([x1, y1, x2, y2], 0.9, 'vehicle'))

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Risk scoring
    centers = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltwh()

        # Draw bounding box
        cv2.rectangle(frame,
                      (int(l), int(t)),
                      (int(l + w), int(t + h)),
                      (0, 255, 0), 2)

        # Draw ID
        cv2.putText(frame,
                    f"ID: {track_id}",
                    (int(l), int(t - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)

        # Center point
        cx = int(l + w / 2)
        cy = int(t + h / 2)

        centers.append((cx, cy))

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Distance check
    risk = False

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1 = centers[i]
            x2, y2 = centers[j]

            distance = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

            if distance < 60:   # slightly stricter
                risk = True
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Alert display
    if risk:
        cv2.putText(frame, "COLLISION RISK!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

        cv2.putText(frame, "Ambulance Dispatched", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0), 2)

        cv2.putText(frame, "Location: Junction A", (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2)

        cv2.putText(frame, "Hospital Alerted", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

    # Show frame
    cv2.imshow("GuardianEye Demo", frame)

    # Exit
    key = cv2.waitKey(25)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()