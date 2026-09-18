import sqlite3
import os

DB_PATH = "observations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            frame_number INTEGER,
            object_class TEXT,
            tracking_id INTEGER,
            confidence REAL,
            bounding_box TEXT,
            zone TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_observation(timestamp, frame_number, object_class, tracking_id, confidence, bounding_box, zone):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO observations (timestamp, frame_number, object_class, tracking_id, confidence, bounding_box, zone)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, frame_number, object_class, tracking_id, confidence, bounding_box, zone))
    conn.commit()
    conn.close()

def get_all_observations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM observations ORDER BY frame_number ASC")
    rows = cursor.fetchall()
    conn.close()
    
    # Format as list of dicts
    columns = ["id", "timestamp", "frame_number", "object_class", "tracking_id", "confidence", "bounding_box", "zone"]
    return [dict(zip(columns, row)) for row in rows]

def get_observations_for_frame(frame_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM observations WHERE frame_number = ?", (frame_number,))
    rows = cursor.fetchall()
    conn.close()
    
    columns = ["id", "timestamp", "frame_number", "object_class", "tracking_id", "confidence", "bounding_box", "zone"]
    return [dict(zip(columns, row)) for row in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
