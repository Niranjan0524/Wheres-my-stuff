# CSE411 Computer Vision - Project Task 4: Implementation Part 1

**Project Title:** Where Is My Stuff?
**Team Number:** 13
**Names and Roll Numbers:** 
- Saksham Saklani, 2023BCD0049
-  Niranjan Alase, 2023BCD0055
- Abhinav Marlingaplar ,2023BCD0013
- Kedar Vaishnav , 2023BCS0162


---

## 1. Project Overview & Architecture
"Where Is My Stuff?" is a Computer Vision-based tracking and observation system designed to locate and monitor objects within a defined space (e.g., a room). The core architecture involves:
1. **Detection & Tracking Pipeline:** Utilizes YOLOv8 for object detection and ByteTrack for multi-object tracking across frames.
2. **Zone Classification:** A heuristic-based spatial mapping algorithm that identifies which zone (Desk, Floor, Bed, Sofa) an object belongs to based on its coordinates.
3. **Storage:** An SQLite database to log temporal tracking observations.
4. **Backend API:** A FastAPI service acting as a wrapper for a Vision LLM (Google Gemini 2.5 Flash).
5. **Frontend UI:** A Streamlit application that provides an interactive interface for video uploading, processing, visualization, and AI analysis.

## 2. Dataset Collection
For this initial phase, dataset collection consists of acquiring sample continuous video footage of a room environment (`sample_video.mp4`). The videos capture standard household items and their movements across different predefined spatial zones. 
- **Video Specifications:** `.mp4` format videos captured from a static camera viewpoint to ensure consistent spatial coordinate mapping.
- **Future Data Collection:** The dataset will be expanded to include varying lighting conditions, different room layouts, and multiple camera angles to ensure the robustness of the detection and tracking models.

## 3. Pre-processing
The pre-processing steps currently implemented in the pipeline include:
- **Frame Extraction:** Utilizing OpenCV (`cv2.VideoCapture`) to systematically extract frames from the uploaded video file.
- **Image Resizing & Normalization:** Implicitly handled by the YOLOv8 model upon passing the frames for inference.
- **Spatial Coordinate Mapping:** The raw `(x, y)` bounding box coordinates extracted from the tracker are processed to calculate the center point `(cx, cy)` of each object. This center point is then evaluated against the frame dimensions to map the object to a specific quadrant for zone classification.

## 4. Current Implementation (Approx. 25% of Proposed Methodology)
The current prototype successfully implements the foundational core of the project:

- **Object Detection (YOLOv8):** Integrated the `yolov8n.pt` (Nano) model for fast, CPU-efficient frame-by-frame object detection.
- **Multi-Object Tracking (ByteTrack):** Implemented ByteTrack to assign persistent IDs to objects across frames, preventing them from being treated as new objects in every frame.
- **Spatial Zone Classification:** Developed a heuristic function (`classify_zone`) that divides the camera view into four static quadrants (Top-Left=Desk, Bottom-Left=Floor, Top-Right=Bed, Bottom-Right=Sofa) and assigns objects to these zones.
- **Observation Logging (SQLite):** Configured a database schema (`db.py`) to persistently store object metadata (timestamp, frame number, object class, tracking ID, confidence score, and zone).
- **Vision LLM Integration (FastAPI):** Created a backend endpoint (`api.py`) that interfaces with Google's Gemini 2.5 Flash model. This allows the system to send specific frames to the LLM for advanced scene description and semantic object identification.
- **User Interface (Streamlit):** Built a functional frontend (`app.py`) allowing users to upload videos, run the pipeline with a progress bar, download the annotated video, view the SQLite database records in a tabular format, and query the LLM on specific frames.

## 5. Major Tasks/Modules Remaining
To complete the project methodology, the following tasks remain:

1. **Dynamic Zone Mapping:** Replace the static quadrant heuristic with a dynamic polygon-based drawing tool in the UI, allowing users to custom-define zones (e.g., dragging points to outline a specific table) for different camera angles.
2. **Object Re-identification (ReID):** Enhance tracking robustly to handle heavy occlusions and scenarios where objects leave and re-enter the camera frame (ensuring they retain their original tracking ID).
3. **Natural Language Query Engine:** Develop an NLP interface where a user can ask, "Where is my laptop?" The system will parse this, query the SQLite database for the latest known location of the 'laptop' class, and return the corresponding zone and timestamp.
4. **Custom Model Fine-tuning:** If the base YOLOv8 model struggles to detect specific personal items (e.g., keys, wallets), we will need to annotate a custom dataset and fine-tune the model.
5. **Real-time Processing Integration:** Adapt the pipeline (`cv2.VideoCapture(0)`) to support real-time RTSP streams or local webcams instead of just pre-recorded `.mp4` uploads.
6. **Evaluation & Metrics:** Rigorously evaluate the system using standard tracking metrics (MOTA, IDF1) and detection metrics (mAP) to benchmark performance.
