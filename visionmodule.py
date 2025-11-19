import os
import cv2
from ultralytics import YOLO


# Path to your YOLO dataset
YOLO_DATASET_PATH = r"C:\Users\91901\Downloads\animals.v1i.yolov8"

# Use the train images folder
IMAGES_DIR = os.path.join(YOLO_DATASET_PATH, "train", "images")

MODEL_PATH = "yolov8n.pt"  
ALERT_THRESHOLD = 0.6

print("🚀 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"Image folder not found: {IMAGES_DIR}")

print(f"📁 Scanning images in: {IMAGES_DIR}")


for image_name in os.listdir(IMAGES_DIR):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(IMAGES_DIR, image_name)
    print(f"\n🔍 Processing: {image_name}")

    # Run YOLO detection
    results = model(img_path)
    annotated_frame = results[0].plot()  # draws boxes

  
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]

        if conf > ALERT_THRESHOLD:
            print(f"⚠️ ALERT: {label.upper()} detected ({conf:.2f})")
            cv2.putText(
                annotated_frame,
                f"ALERT: {label.upper()} ({conf:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

   
    cv2.imshow("🧠 Vision Alert System", annotated_frame)

    # Press ESC to exit, or any key for next image
    key = cv2.waitKey(0)
    if key == 27:  # ESC key
        print("🛑 Exiting visualization...")
        break

cv2.destroyAllWindows()
print("✅ Completed all detections.")

