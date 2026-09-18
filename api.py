from fastapi import FastAPI, UploadFile, File
import os
from pydantic import BaseModel
import io
from PIL import Image

app = FastAPI()

class AnalysisResult(BaseModel):
    description: str
    objects: list[str]
    summary: str

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

@app.post("/analyze-image", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = analyze_image_with_llm(contents)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
