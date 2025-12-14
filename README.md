# 🚗 Vehicle Speed Estimation using YOLOv8

This project implements **real-time vehicle speed estimation from traffic videos** using **YOLOv8** and **computer vision** techniques.  
It detects vehicles, tracks them across frames, and estimates their speed based on pixel displacement over time.

The system works on **single-camera (monocular) traffic footage** and is lightweight enough to run on a local machine.

---

## ✨ Features

- Vehicle detection using **YOLOv8 (Ultralytics)**
- Multi-object tracking with persistent IDs
- Speed estimation from frame-to-frame motion
- Speed smoothing for stable readings
- Real-time visualization with bounding boxes and speed labels
- Output video generation

---

## 🧠 How It Works

1. YOLOv8 detects vehicles in each video frame.
2. Each detected vehicle is assigned a unique tracking ID.
3. The center point of each bounding box is tracked across frames.
4. Speed is calculated using:
   - Pixel displacement between frames  
   - Video FPS  
   - Approximate meters-per-pixel scale
5. Estimated speeds are smoothed and displayed on the video in **km/h**.

---

## 📂 Project Structure

vehicle-speed-estimation/
│
├── src/
│ ├── detect.py
│ ├── track.py
│ ├── speed_estimation.py
│ └── speed_smooth.py
│
├── outputs/
│ └── speed_output.mp4
│
├── .gitignore
└── README.md


---

## ⚙️ Requirements

- Python 3.8 or higher
- OpenCV
- Ultralytics YOLOv8
- NumPy

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vamshi671/vehicle-speed-estimation.git
cd vehicle-speed-estimation
2️⃣ Create and Activate Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate
Windows
venv\Scripts\activate
3️⃣ Install Dependencies
pip install ultralytics opencv-python numpy
4️⃣ Run the Project
python src/speed_estimation.py
📤 Output

The processed video will be saved as:

outputs/speed_output.mp4
