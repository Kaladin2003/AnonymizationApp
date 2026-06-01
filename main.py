import cv2
import time
import os
from ultralytics import YOLO

# --- 1. Υποστηρικτικές Συναρτήσεις για Ανωνυμοποίηση ---
def apply_gaussian_blur(image, x, y, w, h):
    h_img, w_img = image.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    roi = image[y:y2, x:x2]
    if roi.size == 0:
        return image
    image[y:y2, x:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
    return image

def apply_pixelation(image, x, y, w, h, blocks=10):
    h_img, w_img = image.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    w_clamped, h_clamped = x2 - x, y2 - y
    roi = image[y:y2, x:x2]
    if roi.size == 0:
        return image
    roi_small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated_roi = cv2.resize(roi_small, (w_clamped, h_clamped), interpolation=cv2.INTER_NEAREST)
    image[y:y2, x:x2] = pixelated_roi
    return image

# --- 2. Αρχικοποίηση Μοντέλων ---

# Haar Cascade για Πρόσωπα (Ενσωματωμένο στο OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# YOLOv8 για Πινακίδες Κυκλοφορίας
model_path = 'best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"'{model_path}' not found. Please place your trained YOLO model in the working directory."
    )
yolo_model = YOLO(model_path)
class_names = yolo_model.names  # {0: 'license_plate'}

# --- 3. Ρύθμιση Κάμερας ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check that it is connected and not in use.")

# Initialize pTime before the loop to avoid division by zero on the first frame
pTime = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Υπολογισμός FPS
    cTime = time.time()
    elapsed = cTime - pTime
    fps = 1 / elapsed if elapsed > 0 else 0
    pTime = cTime

    # --- HAAR CASCADES: Εντοπισμός Προσώπων ---
    # Haar Cascades work on grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        frame = apply_gaussian_blur(frame, x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- YOLOv8: Εντοπισμός Πινακίδων Κυκλοφορίας ---
    # Convert BGR (OpenCV default) → RGB (YOLO expected format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_yolo = yolo_model(frame_rgb, imgsz=640, verbose=False)

    for r in results_yolo:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = class_names.get(cls, "").lower()

            if label == 'license_plate':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w_box, h_box = x2 - x1, y2 - y1
                frame = apply_pixelation(frame, x1, y1, w_box, h_box, blocks=15)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)

    # Εμφάνιση FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Real-Time Anonymization", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()