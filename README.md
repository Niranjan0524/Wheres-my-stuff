# Where Is My Stuff? - Task 4 Part 1

This is the prototype for the core pipeline of "Where Is My Stuff?"

## Architecture
1. **Pipeline**: YOLOv8 (Detection) -> ByteTrack (Tracking) -> Quadrant Heuristic (Zone Classification) -> SQLite (Storage).
2. **Backend API**: FastAPI endpoint that acts as a wrapper for a Vision LLM (Google Gemini by default).
3. **Frontend UI**: Streamlit application to upload videos, run the pipeline, view annotated results, browse stored observations, and query the Vision LLM for specific frames.

## Setup Instructions

1. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Set Environment Variable (Optional but recommended for AI features)**
   Set your Google Gemini API key (or generic API key if you modify `api.py`).
   ```powershell
   $env:LLM_API_KEY="your_api_key_here"
   ```
   *Note: If you don't set this, the AI analysis will just return a mock response for testing.*

3. **Run the Backend API**
   Open a terminal and start the FastAPI server:
   ```powershell
   python api.py
   ```
   The API will run on `http://localhost:8000`.

4. **Run the Streamlit UI**
   Open a **second** terminal and start the UI:
   ```powershell
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit UI in your browser (usually `http://localhost:8501`).
2. Upload a sample room video (any `.mp4` video with objects will work).
3. Click **Start Processing**.
4. The pipeline will process the video, draw bounding boxes with tracking IDs and zones, and save everything to `observations.db`.
5. Once complete, you can:
   - Play or download the annotated video.
   - View the database of observations.
   - Select an extracted frame from the dropdown and click **Analyze with AI** to send it to the Vision LLM.

## Notes
- Zones are currently statically defined by screen quadrants (Top-Left=Desk, Bottom-Left=Floor, Top-Right=Bed, Bottom-Right=Sofa).
- A YOLOv8 Nano model (`yolov8n.pt`) is used by default for fast CPU inference.
