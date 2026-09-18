import streamlit as st
import os
import requests
import pandas as pd
from pipeline import process_video
from db import get_all_observations
import time

st.set_page_config(page_title="Where Is My Stuff?", layout="wide")
st.title("Where Is My Stuff? - Demo (Task 4 Part 1)")

st.write("Core Pipeline: YOLO Detection -> ByteTrack -> Zone Classification -> SQLite Storage")

os.makedirs("uploads", exist_ok=True)
os.makedirs("frames", exist_ok=True)

uploaded_file = st.file_uploader("Upload Room Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = os.path.join("uploads", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")
    
    if st.button("Start Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            if total > 0:
                progress_bar.progress(current / total)
            status_text.text(f"Processing frame {current}/{total}")
            
        with st.spinner("Processing video (YOLO + ByteTrack)..."):
            output_video_path = "output.mp4"
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
                
            output_video_path = process_video(video_path, output_video_path=output_video_path, progress_callback=update_progress)
            
        st.success("Processing Complete!")
        
        st.write("Annotated Video:")
        # Display video. Note: cv2 mp4v codec might not play in all browsers without ffmpeg conversion to h264.
        try:
            st.video(output_video_path)
        except Exception:
            st.warning("Video cannot be played directly in browser. Please download it.")
        
        with open(output_video_path, "rb") as f:
            st.download_button("Download Annotated Video", f, file_name="annotated_output.mp4")

st.markdown("---")
st.header("Observations")
try:
    obs = get_all_observations()
    if obs:
        df = pd.DataFrame(obs)
        st.dataframe(df)
        
        st.markdown("---")
        st.header("Analyze Frame with AI")
        frame_files = sorted([f for f in os.listdir("frames") if f.endswith(".jpg")])
        if frame_files:
            selected_frame = st.selectbox("Select a frame", frame_files)
            frame_path = os.path.join("frames", selected_frame)
            st.image(frame_path, caption=selected_frame, use_container_width=True)
            
            if st.button("Analyze with AI"):
                with st.spinner("Analyzing with Vision LLM..."):
                    try:
                        with open(frame_path, "rb") as f:
                            response = requests.post("http://localhost:8000/analyze-image", files={"file": f})
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.subheader("Analysis Result")
                            st.write("**Description:**", result.get("description"))
                            st.write("**Objects:**", result.get("objects"))
                            st.write("**Summary:**", result.get("summary"))
                        else:
                            st.error(f"API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Failed to connect to API: {e}. Is the FastAPI server running on port 8000?")
    else:
        st.info("No observations found. Process a video first.")
except Exception as e:
    st.info("Database not initialized or no observations yet.")
