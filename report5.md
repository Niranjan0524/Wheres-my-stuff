# CSE411 Computer Vision - Project Task 5: Implementation Part 2

**Project Title:** Where Is My Stuff?  
**Team Number:** 13  
**Names and Roll Numbers:**  
- Saksham Saklani, 2023BCD0049  
- Niranjan Alase, 2023BCD0055  
- Abhinav Marlingaplar, 2023BCD0013  
- Kedar Vaishnav, 2023BCS0162  

---

## 1. Project Overview & Architecture
"Where Is My Stuff?" is a Computer Vision-based tracking and observation system designed to locate and monitor objects within an indoor space (such as a room) and answer the fundamental user question: *"Where did I last leave this object?"*

Advancing from the initial 25% prototype developed in Task 4, this submission achieves **approximately 50% overall implementation of the proposed methodology**. The core architecture has been upgraded from a static demonstration pipeline to an adaptable, queryable observation system:

1. **Detection & Tracking Pipeline:** Utilizes YOLOv8 for per-frame detection and ByteTrack for multi-object tracking. Bounding box colors are deterministically mapped to identities for visual consistency.
2. **Dynamic Zone Classification (`zones.py`):** Replaces rigid static quadrants with a flexible `ZoneManager` evaluating arbitrary polygonal spatial zones using OpenCV's Point-in-Polygon algorithm (`cv2.pointPolygonTest`). Zones are persisted in JSON format.
3. **Visual Re-Identification & Track Stitching (`reid.py`):** Integrates a `GlobalTracker` that extracts normalized HSV color histograms and aspect-ratio descriptors to re-identify objects and stitch fragmented ByteTrack IDs across occlusions and re-entries.
4. **Persistent Observation Storage (`db.py`):** An extended SQLite database schema logging timestamps, frame numbers, object classes, ByteTrack IDs, detection confidence scores, classified zones, global track IDs, cropped object paths, and keyframe paths.
5. **Natural Language Query Engine (`query_engine.py`):** A natural language interface that parses user queries (e.g., *"Where did I leave my bottle?"*), resolves object classes through exact, alias, and fuzzy matching, and retrieves the latest known location, confidence, and movement trajectory.
6. **Backend API (`api.py`):** An expanded FastAPI service hosting endpoints for natural language database queries (`/query`), zone configurations (`/zones`), tracked object summaries (`/classes`, `/summary`), and Vision-Language Model scene analysis (`/analyze-image` via Google Gemini 2.5 Flash).
7. **Frontend UI (`app.py`):** A multi-tab Streamlit dashboard providing video upload and processing, interactive zone previews, natural language search with visual evidence (object crops and full frames), object movement history timelines, and on-demand Vision LLM analysis.

---

## 2. Dataset Collection
Dataset collection continues using continuous static room footage (`sample_video.mp4`) capturing standard household items (laptop, bottle, backpack, books, chair) moving across different spatial surfaces.
- **Current Video Specifications:** Continuous `.mp4` video filmed from a fixed camera viewpoint, allowing consistent pixel-to-world spatial coordinate mapping.
- **Future Data Expansion:** Planned collection across multiple room layouts, varying ambient and artificial lighting conditions, and diverse camera mounting angles to prepare for model fine-tuning and benchmark evaluations.

---

## 3. Pre-processing
The pre-processing pipeline has been augmented for the 50% milestone:
- **Frame Extraction:** Systematic frame-by-frame extraction via OpenCV (`cv2.VideoCapture`) with in-memory streaming to minimize memory footprint.
- **Spatial Coordinate Mapping:** Midpoint calculation `(cx, cy)` from bounding box coordinates `[x1, y1, x2, y2]`, evaluated against polygonal zone boundaries using `cv2.pointPolygonTest`. Points falling outside all configured polygons are categorized as `"Unknown / In Transit"`.
- **Object Crop Extraction:** High-confidence detected object crops are systematically saved to a dedicated `crops/` directory at keyframe intervals. These crops supply visual evidence for search results and feed the appearance feature extractor for track stitching.
- **Image Resizing & Normalization:** Handled natively by the YOLOv8 letterbox preprocessing pipeline.

---

## 4. Current Implementation (Approx. 50% of Proposed Methodology)
The system now implements approximately 50% of the overall proposed methodology, completing the core functional modules required for spatial reasoning, track persistence, and object retrieval:

- **Dynamic Polygonal Zone Classification (`zones.py`):** Implemented the `ZoneManager` class supporting custom convex and concave polygon zones. Replaced hardcoded quadrants with configurable polygon vertices saved in `zones.json`. Zone boundaries and labels are dynamically rendered as translucent overlays on the annotated video.
- **Visual Re-Identification (`reid.py`):** Built an appearance-based re-identification module using 48-dimensional normalized HSV color histograms and aspect ratios. The `GlobalTracker` compares new detections against recently lost tracks using Bhattacharyya distance, stitching fragmented track IDs into persistent `global_track_id` records.
- **Natural Language Query Engine (`query_engine.py`):** Implemented an NLP query parser that extracts entities from natural questions, matches them against observed classes using aliases (e.g., "mug" → "cup", "phone" → "cell phone") and fuzzy matching, queries SQLite for the latest location, and returns human-friendly responses with movement history.
- **Extended Storage Architecture (`db.py`):** Upgraded the SQLite database schema with `global_track_id`, `crop_path`, and `frame_path` columns. Implemented specialized queries for latest observations, zone transition histories, and session summaries.
- **Multi-Tab Dashboard (`app.py`):** Rebuilt the Streamlit interface into four organized tabs:
  1. *Video Processing & Zones:* Video upload, zone configuration preview, processing with progress feedback, and annotated video download.
  2. *Where Is My Stuff?:* Natural language query bar with quick-select prompt chips and visual evidence display (side-by-side cropped object preview and full frame).
  3. *Object Movement History:* Per-class summary cards and chronological zone-transition timelines with frame snapshots.
  4. *AI Scene Analysis:* Frame selector connected to Gemini 2.5 Flash for deep semantic scene interrogation.
- **Extended REST API (`api.py`):** Expanded the FastAPI backend with `/query`, `/zones` (GET/POST), `/classes`, and `/summary` endpoints alongside the existing `/analyze-image` Vision LLM endpoint.

---

## 5. Integration and Testing Results Obtained
The integrated modules were systematically tested on `sample_video.mp4` with the following results:

- **End-to-End Pipeline Execution:** Successfully processed the video stream, performing frame extraction, YOLOv8n inference, ByteTrack association, appearance-based ReID stitching, polygonal zone evaluation, and SQLite logging.
- **Re-Identification Accuracy:** When tracked objects underwent brief occlusions or intermittent detections, the `GlobalTracker` successfully associated new ByteTrack IDs back to the original `global_track_id` via HSV Bhattacharyya distance matching (verified with test cases achieving distance < 0.55 threshold).
- **Zone Classification Integrity:** Objects placed on the desk, bed, floor, and sofa were reliably classified into their respective polygonal boundaries. Centroid labels and translucent color fills rendered cleanly on `output.mp4`.
- **Query Engine Precision:**
  - *"Where is my laptop?"* → Correctly resolved `laptop`, returning the latest zone (`Desk`), frame number, confidence score, and cropped preview.
  - *"Where did I leave my bottle?"* → Correctly extracted `bottle`, returned latest location and chronological zone history.
  - *"What objects do you see?"* → Returned an aggregated summary of all distinct classes, total detection counts, and last seen zones.
  - *"Find my backpack"* → Correctly mapped to `backpack` with corresponding zone and timestamp.
  - Out-of-vocabulary query (e.g., *"Where is my wallet?"*) → Gracefully informed the user that the item was not observed, providing a list of available tracked items.
- **API and UI Verification:** All six FastAPI endpoints responded with HTTP 200 and expected JSON payloads. The Streamlit dashboard successfully rendered all tabs, video downloads, and visual evidence cards.

---

## 6. Major Tasks/Modules Remaining (Remaining ~50%)
To bring the project to 100% completion for the final submission, the following modules remain:

1. **Interactive Polygon Drawing Canvas:** Upgrade the zone configuration interface with an interactive drawing canvas in Streamlit (e.g., `streamlit-drawable-canvas`), allowing users to click and drag vertices directly on a video freeze-frame.
2. **Deep Learning ReID Model:** Supplement the lightweight HSV appearance tracker with a lightweight deep ReID feature extractor (such as OSNet or MobileNet-ReID) for invariance against sharp lighting shifts and significant viewpoint changes.
3. **Custom Model Fine-Tuning:** Annotate a focused indoor dataset of common misplaced items absent from the standard COCO vocabulary (keys, wallets, spectacles, earphones) and fine-tune YOLOv8 Nano.
4. **Real-Time Camera & RTSP Streaming:** Adapt the pipeline from pre-recorded video batch processing to live streaming input (`cv2.VideoCapture(0)` or RTSP camera streams) with continuous temporal sliding-window logging.
5. **Formal Quantitative Evaluation:** Benchmark the tracking pipeline using standardized Computer Vision metrics (MOTA, IDF1, HOTA) to quantify tracking continuity and ReID impact, alongside detection mAP evaluation.
