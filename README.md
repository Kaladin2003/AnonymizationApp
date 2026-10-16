# Anonymization App

Real-time anonymization of **faces** and **license plates** in webcam feeds, videos and images, built with **OpenCV** and **YOLOv8**.

Faces are blurred and license plates are pixelated, so footage can be shared without exposing personal data.

## How it works

| Target | Detector | Anonymization |
|---|---|---|
| Faces | OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) | Gaussian blur |
| License plates | Custom-trained YOLOv8n model (`best.pt`, one class: `license_plate`) | Pixelation |

For every frame, both detectors run on the original image first. Only then are the detected areas blurred or pixelated, so one step never hides an object from the other.

## Installation

```bash
git clone https://github.com/Kaladin2003/AnonymizationApp.git
cd AnonymizationApp
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

OpenCV is pinned below version 5, because OpenCV 5 moved the Haar cascade face detector out of the main package.

## Usage

```bash
python main.py                                              # webcam 0, press q to quit
python main.py --source 1                                   # another webcam
python main.py --source street.mp4 --output street_anon.mp4  # anonymize a video
python main.py --source photo.jpg --output photo_anon.jpg    # anonymize an image
```

| Option | Description |
|---|---|
| `--source` | Webcam index, video file or image file (default: `0`) |
| `--output` | Save the anonymized result to this file |
| `--model` | YOLO license plate model (default: `best.pt`) |
| `--conf` | Plate detection confidence threshold (default: `0.25`) |
| `--no-boxes` | Hide the boxes drawn around anonymized areas |
| `--no-display` | Run without a preview window |

## Limitations

- The Haar cascade detects **frontal** faces only. Faces in profile or at strong angles may be missed. A deep-learning face detector (for example a YOLO face model) would be more robust.
- Plate detection quality depends on the training data of `best.pt`. Lower `--conf` to catch more plates, at the cost of more false positives.

## Team

- **Marios Borozan** – application, face anonymization, project setup
- **GlenVokrri** – license plate detection model (`best.pt`)

## License

This project is licensed under the [MIT License](LICENSE).
