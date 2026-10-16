"""Real-time anonymization of faces and license plates.

Faces are found with an OpenCV Haar cascade and blurred.
License plates are found with a custom-trained YOLOv8 model and pixelated.

Usage:
    python main.py                                  # webcam 0
    python main.py --source street.mp4 --output anonymized.mp4
    python main.py --source photo.jpg --output photo_anon.jpg
"""

import argparse
import os
import sys
import time

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_COLOR = (0, 255, 0)    # green (BGR)
PLATE_COLOR = (255, 0, 0)   # blue (BGR)


# --- 1. Anonymization helpers ---

def clamp_box(image, x, y, w, h):
    """Clip a box to the image borders. Returns (x1, y1, x2, y2)."""
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    return x1, y1, x2, y2


def apply_gaussian_blur(image, x, y, w, h):
    x1, y1, x2, y2 = clamp_box(image, x, y, w, h)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
    return image


def apply_pixelation(image, x, y, w, h, blocks=10):
    x1, y1, x2, y2 = clamp_box(image, x, y, w, h)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    roi_small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    image[y1:y2, x1:x2] = cv2.resize(roi_small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return image


# --- 2. Detection ---

def detect_faces(face_cascade, frame):
    """Return face boxes as (x, y, w, h). Haar cascades work on grayscale."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))


def detect_plates(model, frame, conf):
    """Return license plate boxes as (x, y, w, h).

    Ultralytics expects NumPy frames in BGR order, the same as OpenCV,
    so the frame is passed in unchanged.
    """
    plates = []
    for result in model(frame, imgsz=640, conf=conf, verbose=False):
        for box in result.boxes:
            label = model.names.get(int(box.cls[0]), "").lower()
            if label == "license_plate":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plates.append((x1, y1, x2 - x1, y2 - y1))
    return plates


def anonymize_frame(frame, face_cascade, model, conf=0.25, draw_boxes=True):
    """Detect faces and plates on the clean frame, then anonymize them."""
    faces = detect_faces(face_cascade, frame)
    plates = detect_plates(model, frame, conf)

    for (x, y, w, h) in faces:
        apply_gaussian_blur(frame, x, y, w, h)
        if draw_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_COLOR, 2)

    for (x, y, w, h) in plates:
        apply_pixelation(frame, x, y, w, h, blocks=15)
        if draw_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), PLATE_COLOR, 2)
            cv2.putText(frame, "license_plate", (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, PLATE_COLOR, 2)

    return frame, len(faces), len(plates)


# --- 3. Input / output ---

def load_models(model_path):
    if not os.path.exists(model_path):
        sys.exit(f"Model '{model_path}' not found. Place the trained YOLO model in the working directory.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade, YOLO(model_path)


def process_image(args, face_cascade, model):
    frame = cv2.imread(args.source)
    if frame is None:
        sys.exit(f"Cannot read image '{args.source}'.")
    frame, n_faces, n_plates = anonymize_frame(frame, face_cascade, model, args.conf, not args.no_boxes)
    print(f"Faces: {n_faces}, license plates: {n_plates}")

    if args.output:
        cv2.imwrite(args.output, frame)
        print(f"Saved: {args.output}")
    if not args.no_display:
        cv2.imshow("Anonymized image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_stream(args, face_cascade, model):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source '{args.source}'. Check the camera or file path.")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_time = time.time()
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame, _, _ = anonymize_frame(frame, face_cascade, model, args.conf, not args.no_boxes)

            now = time.time()
            fps = 1 / (now - prev_time) if now > prev_time else 0
            prev_time = now

            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, FACE_COLOR, 3)
                cv2.imshow("Real-Time Anonymization", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Blur faces and pixelate license plates in images, videos or a webcam.")
    parser.add_argument("--source", default="0", help="webcam index, video file or image file (default: 0)")
    parser.add_argument("--output", help="save the anonymized result to this file")
    parser.add_argument("--model", default="best.pt", help="YOLO license plate model (default: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="plate detection confidence threshold (default: 0.25)")
    parser.add_argument("--no-boxes", action="store_true", help="do not draw boxes around anonymized areas")
    parser.add_argument("--no-display", action="store_true", help="do not open a preview window")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    face_cascade, model = load_models(args.model)

    if os.path.splitext(args.source)[1].lower() in IMAGE_EXTENSIONS:
        process_image(args, face_cascade, model)
    else:
        process_stream(args, face_cascade, model)


if __name__ == "__main__":
    main()
