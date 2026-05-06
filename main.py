import cv2
import time
from ultralytics import YOLO

# --- 1. Υποστηρικτικές Συναρτήσεις για Ανωνυμοποίηση ---
def apply_gaussian_blur(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    if roi.size == 0: return image
    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
    image[y:y+h, x:x+w] = blurred_roi
    return image

def apply_pixelation(image, x, y, w, h, blocks=10):
    roi = image[y:y+h, x:x+w]
    if roi.size == 0: return image
    roi_small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated_roi = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = pixelated_roi
    return image

# --- 2. Αρχικοποίηση Μοντέλων ---

# Haar Cascade για Πρόσωπα (Ενσωματωμένο στο OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# YOLOv8 για Πινακίδες (ή Οχήματα προσωρινά)
yolo_model = YOLO('yolov8n.pt') 

# --- 3. Ρύθμιση Κάμερας ---
cap = cv2.VideoCapture(0)
pTime = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Υπολογισμός FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    # --- HAAR CASCADES: Εντοπισμός Προσώπων ---
    # Τα Haar Cascades δουλεύουν καλύτερα σε ασπρόμαυρη εικόνα
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Εφαρμογή Gaussian Blur στα πρόσωπα
        frame = apply_gaussian_blur(frame, x, y, w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # --- YOLOv8: Εντοπισμός Οχημάτων/Πινακίδων ---
    results_yolo = yolo_model(frame, verbose=False)
    
    for r in results_yolo:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 2: # 2 = "Αυτοκίνητο"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w_box, h_box = x2 - x1, y2 - y1
                # Εφαρμογή Pixelation στα οχήματα
                frame = apply_pixelation(frame, x1, y1, w_box, h_box, blocks=15)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Εμφάνιση FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Real-Time Anonymization", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()