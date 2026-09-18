import cv2
from ultralytics import YOLO
import datetime
import os
import json
from db import init_db, insert_observation

def classify_zone(x, y, img_width, img_height):
    # Simple heuristic for zones based on quadrants
    if x < img_width / 2:
        if y < img_height / 2:
            return "Desk (Top-Left)"
        else:
            return "Floor (Bottom-Left)"
    else:
        if y < img_height / 2:
            return "Bed (Top-Right)"
        else:
            return "Sofa (Bottom-Right)"

def process_video(video_path, output_video_path="output.mp4", frames_dir="frames", progress_callback=None):
    init_db()
    
    # Initialize YOLO model (nano model for speed)
    model = YOLO("yolov8n.pt") 
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file {video_path}")
        
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or fps != fps:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (img_width, img_height))
    
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_number += 1
        timestamp = str(datetime.datetime.now())
        
        # Run YOLO with ByteTrack
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                class_name = model.names[class_id]
                zone = classify_zone(cx, cy, img_width, img_height)
                
                # Store observation
                box_str = json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                insert_observation(timestamp, frame_number, class_name, track_id, float(conf), box_str, zone)
                
                # Draw on frame
                label = f"{class_name} | ID:{track_id} | {conf:.2f} | {zone}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        out.write(annotated_frame)
        
        # Save every 30th frame for UI selection
        if frame_number % 30 == 0 or frame_number == 1:
            frame_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_path, frame) # save original frame for LLM
            
        if progress_callback:
            progress_callback(frame_number, total_frames)
            
    cap.release()
    out.release()
    return output_video_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_video(sys.argv[1])
