import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "models/yolo11n_ncnn_model"
IMAGE_SIZE = 320
CONFIDENCE = 0.35

# Load the exported YOLO model
model = YOLO(MODEL_PATH, task="detect")
print("YOLO model loaded.")

# Configure Raspberry Pi Camera Module
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()
time.sleep(2)

print("Camera started.")
print("Object detection running. Press Q to quit.")

previous_time = time.time()

try:
    while True:
        # Capture frame from CSI camera
        frame_rgb = picam2.capture_array()

        # Convert to OpenCV BGR format
        frame = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        # Perform object detection
        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        output = results[0].plot()

        # Calculate approximate processing FPS
        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        previous_time = current_time

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLO Object Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
