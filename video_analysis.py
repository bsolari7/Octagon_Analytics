"""Exploratory YOLO pose analysis for an uploaded fight clip."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

def angle(a, b, c):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)): return np.nan
    ba, bc = a - b, c - b; denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    return np.nan if denominator == 0 else float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denominator, -1, 1))))

def analyze_fight_video(video_bytes, suffix, sample_fps=2, max_seconds=60, confidence=.5):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("The YOLO dependencies are not installed. Check requirements.txt and redeploy.") from error
    with tempfile.TemporaryDirectory(prefix="octagon_pose_") as folder:
        folder = Path(folder); source = folder / f"clip{suffix or '.mp4'}"; result_path = folder / "pose_overlay.mp4"; source.write_bytes(video_bytes)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened(): raise ValueError("This file could not be read as video. Try an MP4 or MOV.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30; width, height = int(capture.get(3)), int(capture.get(4)); total = int(capture.get(7))
        if not width or not height: raise ValueError("The clip has no readable frames.")
        step, maximum = max(1, round(fps / sample_fps)), min(total, int(max_seconds * fps))
        writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (width, height))
        if not writer.isOpened(): raise RuntimeError("Could not write the annotated video.")
        model = YOLO("yolo11n-pose.pt")
        prior, rows, sampled, detected, people, frame = {}, [], 0, 0, 0, 0
        while frame < maximum:
            ok, image = capture.read()
            if not ok: break
            if frame % step:
                frame += 1; continue
            timestamp = frame / fps
            output = model.track(image, persist=True, classes=[0], conf=confidence, verbose=False)[0]
            annotated = output.plot(); cv2.putText(annotated, f"Pose sample: {timestamp:05.1f}s", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA); writer.write(annotated); sampled += 1
            if output.keypoints is not None and len(output.keypoints):
                detected += 1; points = output.keypoints.xyn.cpu().numpy(); conf = output.keypoints.conf.cpu().numpy()
                ids = list(range(1, len(points)+1)) if output.boxes.id is None else output.boxes.id.int().cpu().tolist()
                for subject, pose, pose_conf in zip(ids, points, conf):
                    people += 1
                    def p(i): return pose[i] if pose_conf[i] >= .35 else np.array([np.nan, np.nan])
                    wrists = np.nanmean(np.array([p(9),p(10)]), axis=0)
                    activity = np.nan
                    if subject in prior and not np.any(np.isnan(wrists)):
                        old_time, old_wrists = prior[subject]; activity = float(np.linalg.norm(wrists-old_wrists)/(timestamp-old_time)) if timestamp > old_time and not np.any(np.isnan(old_wrists)) else np.nan
                    if not np.any(np.isnan(wrists)): prior[subject] = (timestamp, wrists)
                    rows.append({"Tracked subject": f"Subject {subject}", "sample": timestamp, "Pose confidence": float(np.nanmean(pose_conf)), "Wrist activity": activity, "Left elbow angle": angle(p(5),p(7),p(9)), "Left knee angle": angle(p(11),p(13),p(15))})
            frame += 1
        capture.release(); writer.release()
        if not sampled: raise ValueError("No video frames could be sampled.")
        records = pd.DataFrame(rows)
        subjects = records.groupby("Tracked subject", as_index=False).agg(**{"Pose samples":("sample","size"), "Mean pose confidence":("Pose confidence","mean"), "Mean wrist activity":("Wrist activity","mean"), "Mean left elbow angle":("Left elbow angle","mean"), "Mean left knee angle":("Left knee angle","mean")}) if len(records) else pd.DataFrame()
        return {"video": result_path.read_bytes(), "subjects": subjects, "sampled": sampled, "detection_rate": detected/sampled, "people_per_sample": people/sampled, "duration": min(total/fps, max_seconds)}
