import os
import cv2
from ultralytics import YOLO

# ==========================
# CONFIGURATION
# ==========================
# Path to your YOLO dataset
YOLO_DATASET_PATH = r"C:\Users\91901\Downloads\animals.v1i.yolov8"

# Path to YOLO pretrained model (replace with your own if needed)
MODEL_PATH = "yolov8n.pt"
ALERT_THRESHOLD = 0.6

# ==========================
# LOAD MODEL
# ==========================
print("🚀 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# ==========================
# DETECT AVAILABLE FOLDERS
# ==========================
SUBSETS = ["train", "valid", "test"]
found_folders = []

for subset in SUBSETS:
    subset_path = os.path.join(YOLO_DATASET_PATH, subset, "images")
    if os.path.exists(subset_path):
        found_folders.append(subset_path)

if not found_folders:
    raise FileNotFoundError(
        f"No 'train', 'valid', or 'test' image folders found under {YOLO_DATASET_PATH}"
    )

print(f"📁 Found image folders: {found_folders}")

# ==========================
# PROCESS ALL IMAGES
# ==========================
for folder in found_folders:
    print(f"\n📸 Processing dataset folder: {folder}")
    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        print(f"⚠️ No images found in {folder}")
        continue

    for image_name in images:
        img_path = os.path.join(folder, image_name)
        print(f"\n🔍 Processing: {image_name}")

        # Run YOLO inference
        results = model(img_path)
        annotated = results[0].plot()  # Draw bounding boxes

        # ==========================
        # ALERT VISUALIZATION
        # ==========================
        alert_triggered = False
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]

            if conf > ALERT_THRESHOLD:
                alert_triggered = True
                print(f"⚠️ ALERT: {label.upper()} detected ({conf:.2f})")
                cv2.putText(
                    annotated,
                    f"ALERT: {label.upper()} ({conf:.2f})",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

        if not alert_triggered:
            cv2.putText(
                annotated,
                "SAFE - No alerts",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
            )

        # ==========================
        # DISPLAY RESULTS
        # ==========================
        window_name = f"🧠 Vision Alerts ({os.path.basename(folder)})"
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(0)
        if key == 27:  # ESC to stop
            print("🛑 Exiting visualization...")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("✅ Completed detections for all folders.")
