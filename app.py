"""
app.py — Streamlit Frontend (Task 5 — Multi-Tab Dashboard)

Tabs:
  1. Video Processing & Zone Config
  2. "Where Is My Stuff?" — NL Query Search
  3. Object Movement History
  4. AI Scene Analysis (Gemini)
"""

import streamlit as st
import os
import requests
import pandas as pd
from pipeline import process_video
from db import (
    init_db,
    get_all_observations,
    get_distinct_tracked_classes,
    get_observation_summary,
    get_latest_observation,
    get_object_history,
)
from query_engine import answer_query
from zones import ZoneManager
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Where Is My Stuff?", layout="wide")
st.title("🔎 Where Is My Stuff?")
st.caption("Computer Vision Object Tracking & Observation System — Task 5 (50% Milestone)")

os.makedirs("uploads", exist_ok=True)
os.makedirs("frames", exist_ok=True)
os.makedirs("crops", exist_ok=True)

# Ensure DB exists
init_db()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Video Processing & Zones",
    "🔍 Where Is My Stuff?",
    "📋 Object Movement History",
    "🤖 AI Scene Analysis",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Video Processing & Zone Configuration
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Video Processing & Zone Configuration")

    col_upload, col_zones = st.columns([3, 2])

    with col_zones:
        st.subheader("Zone Configuration")
        zm = ZoneManager()
        st.markdown("Current zones are loaded from `zones.json` "
                     "(or default quadrant layout).")

        for i, zone in enumerate(zm.zones):
            st.markdown(f"**{i+1}. {zone['name']}** — "
                        f"{len(zone['points'])} vertices, "
                        f"colour `{zone['color']}`")

        st.info("💡 To customise zones, edit `zones.json` in the project "
                "folder and re-run.  Dynamic polygon drawing UI is planned "
                "for the next milestone.")

    with col_upload:
        st.subheader("Upload & Process Video")
        uploaded_file = st.file_uploader("Upload Room Video",
                                         type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            video_path = os.path.join("uploads", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Uploaded **{uploaded_file.name}**")

            if st.button("▶️ Start Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    if total > 0:
                        progress_bar.progress(current / total)
                    status_text.text(
                        f"Processing frame {current}/{total}")

                with st.spinner("Running YOLO + ByteTrack + ReID pipeline…"):
                    output_video_path = "output.mp4"
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)

                    output_video_path = process_video(
                        video_path,
                        output_video_path=output_video_path,
                        progress_callback=update_progress,
                    )

                st.success("✅ Processing Complete!")

                st.subheader("Annotated Video")
                try:
                    st.video(output_video_path)
                except Exception:
                    st.warning("Video cannot play in-browser (codec). "
                               "Download below.")

                with open(output_video_path, "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f,
                                       file_name="annotated_output.mp4")

    # Observations table
    st.markdown("---")
    st.subheader("📊 Observation Database")
    try:
        obs = get_all_observations()
        if obs:
            df = pd.DataFrame(obs)
            # Clean up display
            display_cols = ["frame_number", "object_class",
                            "global_track_id", "tracking_id",
                            "zone", "confidence", "timestamp"]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available], use_container_width=True,
                         height=300)
            st.caption(f"Total observations: {len(df)}")
        else:
            st.info("No observations found. Process a video first.")
    except Exception:
        st.info("Database not initialised or no observations yet.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Natural Language Query ("Where Is My Stuff?")
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Where Is My Stuff?")
    st.markdown("Ask a question about any object tracked in the video.")

    # Example chips
    example_queries = [
        "Where is my laptop?",
        "Where did I leave the bottle?",
        "What objects do you see?",
        "Find my backpack",
        "Where is the chair?",
    ]
    st.caption("💡 Try one of these:")
    chip_cols = st.columns(len(example_queries))
    selected_example = None
    for i, eq in enumerate(example_queries):
        with chip_cols[i]:
            if st.button(eq, key=f"chip_{i}"):
                selected_example = eq

    query_input = st.text_input(
        "Your question:",
        value=selected_example or "",
        placeholder="e.g. Where is my laptop?",
    )

    if st.button("🔎 Search", type="primary") and query_input:
        with st.spinner("Searching…"):
            result = answer_query(query_input)

        if result["success"]:
            st.success("Found!")
        else:
            st.warning("No match found.")

        st.markdown(result["answer"])

        # Show evidence images if available
        latest = result.get("latest")
        if latest:
            evidence_cols = st.columns(2)
            # Show crop
            crop = latest.get("crop_path")
            if crop and os.path.exists(crop):
                with evidence_cols[0]:
                    st.image(crop, caption="Object Crop",
                             use_container_width=True)
            # Show full frame
            fp = latest.get("frame_path")
            if fp and os.path.exists(fp):
                with evidence_cols[1]:
                    st.image(fp, caption="Full Frame",
                             use_container_width=True)

    # Show tracked classes for reference
    st.markdown("---")
    classes = get_distinct_tracked_classes()
    if classes:
        st.caption(f"**Tracked objects in DB:** {', '.join(classes)}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Object Movement History
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📋 Object Movement History")
    st.markdown("See how objects moved between zones over time.")

    classes = get_distinct_tracked_classes()
    if not classes:
        st.info("No data yet. Process a video first.")
    else:
        # Summary cards
        st.subheader("Summary")
        summary = get_observation_summary()
        if summary:
            card_cols = st.columns(min(len(summary), 4))
            for i, s in enumerate(summary):
                with card_cols[i % len(card_cols)]:
                    st.metric(
                        label=s["object_class"].title(),
                        value=s["last_zone"],
                        delta=f"{s['total_detections']} detections",
                    )

        st.markdown("---")
        st.subheader("Zone Transition Timeline")
        selected_class = st.selectbox(
            "Select object class:", classes)

        if selected_class:
            history = get_object_history(object_class=selected_class)
            if history:
                st.markdown(f"**{selected_class.title()}** — "
                            f"{len(history)} zone transition(s):")

                for i, h in enumerate(history):
                    icon = "🟢" if i == 0 else "➡️"
                    st.markdown(
                        f"{icon} **{h['zone']}** — frame "
                        f"{h['frame_number']} "
                        f"(confidence {h['confidence']:.1%})"
                    )
                    # Show frame if available
                    fp = h.get("frame_path")
                    if fp and os.path.exists(fp):
                        st.image(fp, width=320,
                                 caption=f"Frame {h['frame_number']}")
            else:
                st.info(f"No transitions found for **{selected_class}**.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Scene Analysis (Gemini)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🤖 AI Scene Analysis")
    st.markdown("Select a saved frame and send it to the Vision LLM "
                "(Google Gemini 2.5 Flash) for a detailed description.")

    frame_files = sorted(
        [f for f in os.listdir("frames") if f.endswith(".jpg")]
    ) if os.path.exists("frames") else []

    if frame_files:
        selected_frame = st.selectbox("Select a frame:", frame_files)
        frame_path = os.path.join("frames", selected_frame)
        st.image(frame_path, caption=selected_frame,
                 use_container_width=True)

        if st.button("🧠 Analyze with AI", type="primary"):
            with st.spinner("Analyzing with Vision LLM…"):
                try:
                    with open(frame_path, "rb") as f:
                        response = requests.post(
                            "http://localhost:8000/analyze-image",
                            files={"file": f},
                        )

                    if response.status_code == 200:
                        result = response.json()
                        st.subheader("Analysis Result")
                        st.write("**Description:**",
                                 result.get("description"))
                        st.write("**Objects:**",
                                 result.get("objects"))
                        st.write("**Summary:**",
                                 result.get("summary"))
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(
                        f"Failed to connect to API: {e}. "
                        "Is the FastAPI server running on port 8000?"
                    )
    else:
        st.info("No frames available. Process a video first.")
