"""
api.py — FastAPI Backend

Endpoints:
  POST /analyze-image    — Send an image frame to the Vision LLM (Gemini 2.5 Flash)
  POST /query            — Natural-language "where is my …" query against the DB
  GET  /zones            — Retrieve current zone configuration
  POST /zones            — Update zone polygon configuration
  GET  /classes          — List distinct object classes tracked in the DB
  GET  /summary          — Observation summary per class
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from pydantic import BaseModel
import io
from PIL import Image

from query_engine import answer_query
from zones import ZoneManager
from db import get_distinct_tracked_classes, get_observation_summary, init_db

app = FastAPI(title="Where Is My Stuff? — API", version="2.0")


# ── Models ───────────────────────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    description: str
    objects: list[str]
    summary: str


class QueryRequest(BaseModel):
    question: str


class ZoneUpdateRequest(BaseModel):
    zones: list[dict]


# ── Vision LLM ───────────────────────────────────────────────────────────────
def analyze_image_with_llm(image_bytes):
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return {
            "description": "[MOCK] A view of a room with a desk and some objects.",
            "objects": ["bottle", "laptop", "book (mock)"],
            "summary": "Mock summary because LLM_API_KEY is not set."
        }

    try:
        import google.genai as genai
        client = genai.Client(api_key=api_key)

        image = Image.open(io.BytesIO(image_bytes))

        prompt = (
            "Briefly describe this scene and identify the visible objects. "
            "Return a short description."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image]
        )

        text = response.text

        return {
            "description": text,
            "objects": ["Extracted from Gemini response"],
            "summary": "Successfully analyzed with Gemini"
        }
    except Exception as e:
        return {
            "description": f"Error analyzing image: {str(e)}",
            "objects": [],
            "summary": "API call failed."
        }


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/analyze-image", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """Send a frame image to the Vision LLM for scene description."""
    contents = await file.read()
    result = analyze_image_with_llm(contents)
    return result


@app.post("/query")
async def query_objects(req: QueryRequest):
    """Answer a natural-language question about tracked objects."""
    result = answer_query(req.question)
    return JSONResponse(content=result)


@app.get("/zones")
async def get_zones():
    """Return the current zone polygon configuration."""
    zm = ZoneManager()
    return JSONResponse(content={"zones": zm.to_dict_list()})


@app.post("/zones")
async def update_zones(req: ZoneUpdateRequest):
    """Update zone polygon configuration and persist to disk."""
    zm = ZoneManager()
    zm.update_zones(req.zones)
    return JSONResponse(content={"status": "ok", "zones": zm.to_dict_list()})


@app.get("/classes")
async def list_classes():
    """Return distinct object classes tracked in the current database."""
    init_db()
    classes = get_distinct_tracked_classes()
    return JSONResponse(content={"classes": classes})


@app.get("/summary")
async def summary():
    """Return per-class observation summary."""
    init_db()
    data = get_observation_summary()
    return JSONResponse(content={"summary": data})


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
