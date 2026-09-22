"""
reid.py — Lightweight Visual Re-Identification & Track Stitching

When ByteTrack loses an object (occlusion, leaving frame) and re-detects it
later with a *new* tracking ID, this module attempts to match it back to its
previous identity using visual appearance features.

Strategy (CPU-friendly, no extra model download):
  1. For each tracked detection, compute a compact appearance descriptor:
     - Normalised HSV colour histogram (3-channel, 16 bins each → 48-dim)
     - Bounding-box aspect ratio
  2. Maintain a gallery of recently-seen tracks (keyed by global_track_id).
  3. When a *new* ByteTrack ID appears, compare its descriptor against the
     gallery of recently-disappeared tracks of the *same object class*.
  4. If the best match is above a similarity threshold, assign the old
     global_track_id instead of creating a fresh one.

This is intentionally simple — the goal for Task 5 is to demonstrate that
identity stitching *works* and improves tracking continuity; a heavier ReID
CNN (e.g., OSNet) is deferred to the remaining 50%.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Configuration ────────────────────────────────────────────────────────────
HIST_BINS = 16          # per HSV channel
SIMILARITY_THRESHOLD = 0.55   # Bhattacharyya distance below this ⇒ match
MAX_DISAPPEARED_FRAMES = 150  # frames before a lost track is evicted
MAX_GALLERY_SIZE = 200        # hard cap on gallery to bound memory


@dataclass
class TrackDescriptor:
    """Appearance descriptor for a single tracked object."""
    global_track_id: int
    object_class: str
    histogram: np.ndarray          # normalised HSV histogram (48-dim)
    aspect_ratio: float
    last_seen_frame: int
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)


class GlobalTracker:
    """Assigns persistent global IDs by stitching fragmented ByteTrack IDs."""

    def __init__(self):
        self._next_global_id: int = 1
        # byte_track_id  →  global_track_id   (for currently active tracks)
        self._active_map: dict[int, int] = {}
        # global_track_id  →  TrackDescriptor (gallery of all known tracks)
        self._gallery: dict[int, TrackDescriptor] = {}
        # byte_track_id set from previous frame (to detect new / lost IDs)
        self._prev_byte_ids: set[int] = set()

    # ── Public API ───────────────────────────────────────────────────────────
    def update(
        self,
        frame: np.ndarray,
        byte_track_id: int,
        object_class: str,
        bbox: tuple[int, int, int, int],
        frame_number: int,
    ) -> int:
        """
        Given a ByteTrack detection, return its global_track_id.

        If this byte_track_id was seen last frame, the existing mapping is
        reused.  If it is *new*, we try to match it against recently lost
        tracks of the same class via appearance similarity.
        """
        # Fast path — already mapped
        if byte_track_id in self._active_map:
            gid = self._active_map[byte_track_id]
            # Refresh gallery descriptor
            desc = self._compute_descriptor(frame, gid, object_class, bbox, frame_number)
            self._gallery[gid] = desc
            return gid

        # New ByteTrack ID — try to match against disappeared tracks
        descriptor = self._compute_descriptor(frame, -1, object_class, bbox, frame_number)
        matched_gid = self._match_against_gallery(descriptor, frame_number)

        if matched_gid is not None:
            gid = matched_gid
        else:
            gid = self._next_global_id
            self._next_global_id += 1

        self._active_map[byte_track_id] = gid
        descriptor.global_track_id = gid
        self._gallery[gid] = descriptor
        return gid

    def end_of_frame(self, current_byte_ids: set[int], frame_number: int):
        """
        Call once per frame after all detections have been processed.
        Removes mappings for ByteTrack IDs that disappeared this frame
        (the gallery entry is kept for potential future re-identification).
        """
        lost = set(self._active_map.keys()) - current_byte_ids
        for bid in lost:
            del self._active_map[bid]

        # Evict very old gallery entries to bound memory
        to_remove = []
        for gid, desc in self._gallery.items():
            if (frame_number - desc.last_seen_frame) > MAX_DISAPPEARED_FRAMES:
                to_remove.append(gid)
        for gid in to_remove:
            del self._gallery[gid]

        # Hard cap
        if len(self._gallery) > MAX_GALLERY_SIZE:
            # keep only the most recently seen
            sorted_items = sorted(self._gallery.items(),
                                  key=lambda kv: kv[1].last_seen_frame, reverse=True)
            self._gallery = dict(sorted_items[:MAX_GALLERY_SIZE])

        self._prev_byte_ids = current_byte_ids.copy()

    # ── Internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_histogram(frame: np.ndarray,
                           bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Compute a normalised HSV colour histogram for the cropped region."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(HIST_BINS * 3, dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hists = []
        for ch in range(3):
            hist = cv2.calcHist([hsv], [ch], None, [HIST_BINS], [0, 256])
            cv2.normalize(hist, hist)
            hists.append(hist.flatten())
        return np.concatenate(hists).astype(np.float32)

    def _compute_descriptor(
        self,
        frame: np.ndarray,
        gid: int,
        object_class: str,
        bbox: tuple[int, int, int, int],
        frame_number: int,
    ) -> TrackDescriptor:
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        hist = self._extract_histogram(frame, bbox)
        return TrackDescriptor(
            global_track_id=gid,
            object_class=object_class,
            histogram=hist,
            aspect_ratio=bw / bh,
            last_seen_frame=frame_number,
            bbox=bbox,
        )

    def _match_against_gallery(
        self,
        query: TrackDescriptor,
        current_frame: int,
    ) -> Optional[int]:
        """
        Find the best-matching recently-disappeared track of the same class.
        Returns the global_track_id if a good match is found, else None.
        """
        # Only consider tracks that are NOT currently active
        active_gids = set(self._active_map.values())
        best_gid: Optional[int] = None
        best_distance = float("inf")

        for gid, gallery_desc in self._gallery.items():
            # Must be same object class
            if gallery_desc.object_class != query.object_class:
                continue
            # Must not be currently active (we want *lost* tracks)
            if gid in active_gids:
                continue
            # Must have disappeared recently
            frames_gone = current_frame - gallery_desc.last_seen_frame
            if frames_gone > MAX_DISAPPEARED_FRAMES or frames_gone <= 0:
                continue

            # Bhattacharyya distance (lower = more similar, 0 = identical)
            dist = cv2.compareHist(
                query.histogram, gallery_desc.histogram, cv2.HISTCMP_BHATTACHARYYA
            )

            # Small penalty for very different aspect ratios
            ar_diff = abs(query.aspect_ratio - gallery_desc.aspect_ratio)
            dist += 0.1 * ar_diff

            if dist < best_distance:
                best_distance = dist
                best_gid = gid

        if best_gid is not None and best_distance < SIMILARITY_THRESHOLD:
            return best_gid
        return None


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = GlobalTracker()
    # Simulate a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gid1 = tracker.update(dummy_frame, byte_track_id=1, object_class="bottle",
                          bbox=(100, 100, 200, 300), frame_number=1)
    gid2 = tracker.update(dummy_frame, byte_track_id=2, object_class="laptop",
                          bbox=(300, 100, 500, 250), frame_number=1)
    tracker.end_of_frame({1, 2}, frame_number=1)
    print(f"Frame 1 -- bottle gid={gid1}, laptop gid={gid2}")

    # Simulate bottle disappearing and reappearing with new ByteTrack ID
    tracker.end_of_frame({2}, frame_number=2)  # bottle (bt=1) lost
    gid3 = tracker.update(dummy_frame, byte_track_id=5, object_class="bottle",
                          bbox=(105, 105, 205, 305), frame_number=3)
    tracker.end_of_frame({2, 5}, frame_number=3)
    print(f"Frame 3 -- bottle reappeared bt=5 -> gid={gid3}  (should be {gid1})")
