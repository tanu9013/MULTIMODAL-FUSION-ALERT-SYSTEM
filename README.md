# Multimodal Real-Time Hazard Detection & Alert System
### Integrating YOLOv8 Vision + Audio Signal Processing for Traffic Safety

This repository contains a **real-time multimodal safety system** that detects hazards using **computer vision** and **audio processing**. By combining YOLO-based object detection with horn/siren audio analysis and a fusion-based decision engine, the system provides accurate, low-latency alerts to enhance road safety.

---

## 🌟 Key Features

- **YOLOv8-based object detection** for animals, pedestrians, vehicles, and traffic hazards  
- **Real-time horn & siren detection** using RMS and MFCC audio features  
- **Fusion engine** to combine vision + audio for high-confidence decisions  
- **Streamlit dashboard** for real-time monitoring  
- **Instant desktop notifications** for alerts  
- **Supports custom YOLO datasets**  
- **Works on normal CPU systems** (no GPU required)

---

## 🏗 Architecture Overview

The fusion engine merges predictions to reduce false positives and increase reliability in poor lighting or noisy environments.

---

## 🧰 Tech Stack

- **Python 3+**  
- **YOLOv8 (Ultralytics)**  
- **OpenCV**  
- **Librosa**  
- **SoundDevice**  
- **Streamlit**  
- **NumPy & Pandas**  
- **Plyer (system notifications)**

---

## 📁 Repository Structure
finalintegration.py ## streamlit app

multimodal-alert-system/
│
├── src/                           # All source code
│   ├── fusionapp.py               # integrated module
│   ├── newvision.py               # Updated vision detection module
│   ├── visionmodule.py            # YOLO vision module code
│   ├── sound.py                   # Audio detection script
│   ├── soundmodule.py             # Audio processing module
│   ├── visualization.py           # Confusion matrix & visualization
│   └── finalintegration.py        # streamlit interface with alert system
│
├── models/
│   └── yolov8n.pt                 # YOLO model weights
│
├── datasets/
│   ├── HornBase - A Car Horns Dataset #Audio data
│   └── animals.v1i.yolov8 #imagesdataset
│
├── docs/
│   ├── MULTIMODALreport.pdf       # Final project report
│   ├── research paper.pdf         # Research paper
│   ├── POSTER.pdf                 # Project poster
│   ├── report.docx                # Editable report
│   ├── MULTIMODAL.docx            # Documentation
│   └── multimodalfusion.pptx      # Presentation
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_plot.png
│
├── README.md                      # Project documentation (to be added)
├── requirements.txt               # Python dependencies
└── .gitignore                     # Ignore unnecessary files

---

## 📊 Performance Summary

- **YOLO Vision Accuracy:** 60-70%  
- **Audio Detection Accuracy:** ~60%  
- **Multimodal Accuracy:** ~70%  
- **False Positives Reduced:** 30–40% (compared to vision-only)




