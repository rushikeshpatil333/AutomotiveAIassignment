import os
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Optional (phone detection via YOLOv8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# =============================
# FaceMesh landmark indices
# =============================
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [263, 387, 385, 362, 380, 373]

# Mouth landmarks (MAR) for yawning
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_UPPER = 13
MOUTH_LOWER = 14

# Head pose landmark indices
POSE_IDXS = [1, 33, 263, 61, 291, 199]


# =============================
# Helpers
# =============================
def euclid(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = euclid(p2, p6)
    B = euclid(p3, p5)
    C = euclid(p1, p4)
    if C == 0:
        return np.nan
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(lms, w, h):
    def pt(i):
        lm = lms[i]
        return (lm.x * w, lm.y * h)
    left = pt(MOUTH_LEFT)
    right = pt(MOUTH_RIGHT)
    upper = pt(MOUTH_UPPER)
    lower = pt(MOUTH_LOWER)
    width = euclid(left, right)
    height = euclid(upper, lower)
    if width == 0:
        return np.nan
    return height / width

def apply_clahe_and_sharpen(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    sharp = cv2.addWeighted(out, 1.20, blur, -0.20, 0)
    return sharp

def time_weighted_ratio(samples):
    if len(samples) < 2:
        return np.nan
    s = list(samples)
    true_time = 0.0
    total = 0.0
    for (t0, v0), (t1, _) in zip(s[:-1], s[1:]):
        dt = max(0.0, t1 - t0)
        total += dt
        if v0:
            true_time += dt
    if total <= 1e-6:
        return np.nan
    return float(true_time / total)

def rotmat_to_euler(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def solve_head_pose_from_lms(lms, w, h):
    image_points = np.array([[lms[i].x * w, lms[i].y * h] for i in POSE_IDXS], dtype=np.float64)

    model_points = np.array([
        (0.0,   0.0,   0.0),
        (-30.0, -30.0, -30.0),
        (30.0,  -30.0, -30.0),
        (-25.0,  30.0, -30.0),
        (25.0,   30.0, -30.0),
        (0.0,    60.0,  10.0),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return (np.nan, np.nan, np.nan)

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotmat_to_euler(R)
    return pitch, yaw, roll

def export_event_clip(frames, out_path, size_wh, fps=30.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size_wh
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()


# =============================
# Main
# =============================
def run_driver_monitor(
    camera_index=0,
    out_dir="dams_output",
    session_name=None,
    display=True,
    resize_width=960,
    enhance_image=True,

    # EAR / PERCLOS
    ear_thresh=0.22,
    ear_calib_factor=0.72,
    perclos_window_sec=30.0,
    perclos_min_fill_sec=8.0,

    # NEW: drowsy hysteresis (ON/OFF thresholds) + awake reset
    perclos_on_thresh=0.28,
    perclos_off_thresh=0.18,
    awake_reset_sec=1.0,

    microsleep_sec=1.2,

    # Yawn (MAR)
    mar_yawn_thresh=0.55,
    yawn_min_sec=0.8,

    # Turned away
    yaw_turn_thresh_deg=25.0,
    pitch_turn_thresh_deg=18.0,
    turned_away_min_sec=1.0,

    # Event clip settings
    pre_event_sec=2.0,
    post_event_sec=2.0,
    min_event_sec=1.0,

    # Full session record
    record_full_session=True,
    full_session_filename="session.mp4",

    # Phone detection
    enable_phone_detection=True,
    yolo_model_name="yolov8n.pt",
    phone_conf_thresh=0.35,
    phone_event_min_sec=1.0,

    stop_key="q"
):
    os.makedirs(out_dir, exist_ok=True)
    if session_name is None:
        session_name = time.strftime("%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Cannot open webcam index {camera_index}")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read from webcam.")

    if resize_width is not None and frame.shape[1] > resize_width:
        scale = resize_width / frame.shape[1]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    h, w = frame.shape[:2]
    size_wh = (w, h)

    session_path = os.path.join(out_dir, f"{session_name}__{full_session_filename}")
    session_writer = None
    if record_full_session:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        session_writer = cv2.VideoWriter(session_path, fourcc, 30.0, (w, h))

    # YOLO for phone
    yolo = None
    if enable_phone_detection:
        if not YOLO_AVAILABLE:
            print("[WARN] ultralytics not installed; phone detection disabled.")
            enable_phone_detection = False
        else:
            try:
                yolo = YOLO(yolo_model_name)
                print(f"[YOLO] Loaded model: {yolo_model_name}")
            except Exception as e:
                print(f"[WARN] Could not load YOLO model '{yolo_model_name}': {e}")
                enable_phone_detection = False

    ring = deque()
    ring_max_sec = max(0.1, pre_event_sec)

    closure_hist = deque()

    ts_hist = deque(maxlen=60)
    t0 = time.time()
    frame_idx = 0

    # Calibration
    calib_mode = False
    calib_ears = []

    # Continuous closure
    closed_since_t = None

    # NEW: drowsy state + open-eyes timer for reset
    drowsy_state = False
    eyes_open_since_t = None

    # Yawn state
    yawn_active = False
    yawn_start_t = None
    yawn_count = 0

    # Phone state
    phone_active = False

    rows = []
    events = []
    pending_export = []
    active = {"drowsy": None, "yawn": None, "turned_away": None, "phone": None}

    def session_time_s():
        return time.time() - t0

    def trim_ring(now_t):
        while ring and (now_t - ring[0][0]) > ring_max_sec:
            ring.popleft()

    def trim_hist(hist, now_t, window_sec):
        while hist and (now_t - hist[0][0]) > window_sec:
            hist.popleft()

    def start_event(event_type, now_t, now_frame):
        event_id = len([e for e in events if e["event_type"] == event_type]) + 1
        active[event_type] = {
            "event_type": event_type,
            "event_id": event_id,
            "start_time_s": now_t,
            "start_frame": now_frame,
            "end_time_s": None,
            "end_frame": None,
            "frames": [fr for _, fr in ring],
        }

    def end_event(event_type, now_t, now_frame, min_sec):
        ev = active[event_type]
        if ev is None:
            return
        ev["end_time_s"] = now_t
        ev["end_frame"] = now_frame
        dur = float(ev["end_time_s"] - ev["start_time_s"])
        if dur < min_sec:
            active[event_type] = None
            return
        pending_export.append((ev, now_t + post_event_sec))
        active[event_type] = None

    def add_frame_to_active(frame_bgr):
        for ev in active.values():
            if ev is not None:
                ev["frames"].append(frame_bgr.copy())

    def export_ready(now_t):
        keep = []
        for ev, export_at in pending_export:
            if now_t < export_at:
                keep.append((ev, export_at))
                continue
            clip_path = os.path.join(out_dir, f"{session_name}__{ev['event_type']}_event_{ev['event_id']:03d}.mp4")
            export_event_clip(ev["frames"], clip_path, size_wh, fps=30.0)
            events.append({
                "event_type": ev["event_type"],
                "event_id": ev["event_id"],
                "start_time_s": ev["start_time_s"],
                "end_time_s": ev["end_time_s"],
                "duration_s": float(ev["end_time_s"] - ev["start_time_s"]),
                "start_frame": ev["start_frame"],
                "end_frame": ev["end_frame"],
                "clip_path": clip_path,
            })
        pending_export[:] = keep

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("Controls: q=quit | c=calibrate EAR (eyes open, press again to apply)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if resize_width is not None and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

            if enhance_image:
                frame = apply_clahe_and_sharpen(frame)

            h, w = frame.shape[:2]
            size_wh = (w, h)

            ts = session_time_s()
            ts_hist.append(ts)
            fps_est = np.nan
            if len(ts_hist) >= 2:
                dt = ts_hist[-1] - ts_hist[0]
                if dt > 0:
                    fps_est = (len(ts_hist) - 1) / dt

            if session_writer is not None:
                session_writer.write(frame)

            ring.append((ts, frame.copy()))
            trim_ring(ts)

            # FaceMesh
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            face_detected = bool(res.multi_face_landmarks)

            ear_avg = np.nan
            eyes_closed = False
            continuous_closed_s = 0.0
            perclos = np.nan

            mar = np.nan
            yawning = False

            pitch = yaw = roll = np.nan
            head_turned_away = False

            drowsy_now = False
            drowsy_reason = ""

            if face_detected:
                lms = res.multi_face_landmarks[0].landmark

                def pt(i):
                    lm = lms[i]
                    return (lm.x * w, lm.y * h)

                # EAR
                ear_l = eye_aspect_ratio([pt(i) for i in LEFT_EYE_EAR])
                ear_r = eye_aspect_ratio([pt(i) for i in RIGHT_EYE_EAR])
                ear_avg = np.nanmean([ear_l, ear_r])

                if calib_mode and (not np.isnan(ear_avg)):
                    calib_ears.append(float(ear_avg))

                eyes_closed = bool((not np.isnan(ear_avg)) and (ear_avg < ear_thresh))

                # Continuous closure
                if eyes_closed:
                    if closed_since_t is None:
                        closed_since_t = ts
                    continuous_closed_s = ts - closed_since_t
                else:
                    closed_since_t = None

                # PERCLOS history
                closure_hist.append((ts, eyes_closed))
                trim_hist(closure_hist, ts, perclos_window_sec)

                if len(closure_hist) >= 2:
                    fill = closure_hist[-1][0] - closure_hist[0][0]
                    if fill >= perclos_min_fill_sec:
                        perclos = time_weighted_ratio(closure_hist)

                # --- NEW drowsy logic with hysteresis + awake reset ---
                if not eyes_closed:
                    if eyes_open_since_t is None:
                        eyes_open_since_t = ts
                else:
                    eyes_open_since_t = None

                # turn ON
                if (not drowsy_state) and (
                    ((not np.isnan(perclos)) and (perclos >= perclos_on_thresh)) or
                    (continuous_closed_s >= microsleep_sec)
                ):
                    drowsy_state = True

                # turn OFF (requires perclos low AND eyes open for a bit)
                if drowsy_state:
                    can_clear_by_perclos = (not np.isnan(perclos)) and (perclos <= perclos_off_thresh)
                    open_long_enough = (eyes_open_since_t is not None) and ((ts - eyes_open_since_t) >= awake_reset_sec)
                    if can_clear_by_perclos and open_long_enough:
                        drowsy_state = False

                drowsy_now = drowsy_state

                reasons = []
                if drowsy_now:
                    if (not np.isnan(perclos)) and perclos >= perclos_on_thresh:
                        reasons.append(f"perclos>={perclos_on_thresh:.2f}")
                    if continuous_closed_s >= microsleep_sec:
                        reasons.append(f"microsleep>={microsleep_sec:.1f}s")
                drowsy_reason = ",".join(reasons)

                # Yawn logic
                mar = mouth_aspect_ratio(lms, w, h)
                yawning = bool((not np.isnan(mar)) and (mar >= mar_yawn_thresh))

                # Head pose
                pitch, yaw, roll = solve_head_pose_from_lms(lms, w, h)
                head_turned_away = bool(
                    (not np.isnan(yaw) and abs(yaw) > yaw_turn_thresh_deg) or
                    (not np.isnan(pitch) and abs(pitch) > pitch_turn_thresh_deg)
                )

                # Events: drowsy
                if drowsy_now and active["drowsy"] is None:
                    start_event("drowsy", ts, frame_idx)
                if (not drowsy_now) and active["drowsy"] is not None:
                    end_event("drowsy", ts, frame_idx, min_event_sec)

                # Events: yawn
                if yawning and (not yawn_active):
                    yawn_active = True
                    yawn_start_t = ts
                    start_event("yawn", ts, frame_idx)
                if (not yawning) and yawn_active:
                    dur = ts - yawn_start_t
                    if dur >= yawn_min_sec:
                        yawn_count += 1
                    yawn_active = False
                    yawn_start_t = None
                    end_event("yawn", ts, frame_idx, yawn_min_sec)

                # Events: turned away
                if head_turned_away and active["turned_away"] is None:
                    start_event("turned_away", ts, frame_idx)
                if (not head_turned_away) and active["turned_away"] is not None:
                    end_event("turned_away", ts, frame_idx, turned_away_min_sec)

            else:
                # if face lost, close face-based events
                eyes_open_since_t = None
                for et in ["drowsy", "yawn", "turned_away"]:
                    if active[et] is not None:
                        end_event(et, ts, frame_idx, min_event_sec)

            # Phone detection
            phone_present = False
            if enable_phone_detection and yolo is not None:
                try:
                    r = yolo.predict(frame, conf=phone_conf_thresh, verbose=False)
                    names = r[0].names
                    boxes = r[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        cls = boxes.cls.cpu().numpy().astype(int)
                        for c in cls:
                            if names.get(int(c), "") == "cell phone":
                                phone_present = True
                                break
                except Exception:
                    phone_present = False

            # Phone events
            if phone_present and (not phone_active):
                phone_active = True
                start_event("phone", ts, frame_idx)
            if (not phone_present) and phone_active:
                phone_active = False
                end_event("phone", ts, frame_idx, phone_event_min_sec)

            add_frame_to_active(frame)
            export_ready(ts)

            # Log
            rows.append({
                "frame": frame_idx,
                "time_s": ts,
                "fps_est": float(fps_est) if not np.isnan(fps_est) else np.nan,
                "face_detected": face_detected,
                "ear_avg": ear_avg,
                "eyes_closed": eyes_closed,
                "continuous_closed_s": continuous_closed_s,
                "perclos": perclos,
                "drowsy_now": drowsy_now,
                "drowsy_reason": drowsy_reason,
                "mar": mar,
                "yawning": yawning,
                "yawn_count": yawn_count,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "roll_deg": roll,
                "head_turned_away": head_turned_away,
                "phone_present": phone_present,
            })

            # UI (no EAR/PERCLOS bars)
            if display:
                overlay = frame.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = 0.45
                th = 1

                cv2.putText(overlay, f"t={ts:.1f}s fps~{(fps_est if not np.isnan(fps_est) else 0):.1f}",
                            (10, 18), font, fs, (240, 240, 240), th, cv2.LINE_AA)

                cv2.putText(overlay, f"DROWSY={drowsy_now} ({drowsy_reason})",
                            (10, 38), font, fs, (0, 0, 255) if drowsy_now else (240, 240, 240),
                            2 if drowsy_now else 1, cv2.LINE_AA)

                cv2.putText(overlay, f"YAWN={yawning}  yawn_count={yawn_count}  mar={mar if not np.isnan(mar) else np.nan:.2f}",
                            (10, 58), font, fs, (240, 240, 240), th, cv2.LINE_AA)

                cv2.putText(overlay, f"TURNED_AWAY={head_turned_away}  yaw={yaw if not np.isnan(yaw) else np.nan:.1f}  pitch={pitch if not np.isnan(pitch) else np.nan:.1f}",
                            (10, 78), font, fs, (240, 240, 240), th, cv2.LINE_AA)

                cv2.putText(overlay, f"PHONE={phone_present}",
                            (10, 98), font, fs, (240, 240, 240), th, cv2.LINE_AA)

                if calib_mode:
                    cv2.putText(overlay, "CALIB: eyes OPEN (press c to apply)", (10, h - 10),
                                font, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(overlay, "Keys: q quit | c calibrate EAR", (10, h - 10),
                                font, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

                cv2.imshow("Driver Monitor", overlay)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(stop_key):
                    break

                if key == ord("c"):
                    calib_mode = not calib_mode
                    if calib_mode:
                        calib_ears = []
                        print("[CALIB] Started: keep eyes OPEN.")
                    else:
                        if len(calib_ears) >= 30:
                            baseline = float(np.median(calib_ears))
                            ear_thresh = float(ear_calib_factor * baseline)
                            print(f"[CALIB] baseline EAR={baseline:.4f} -> ear_thresh={ear_thresh:.4f}")
                        else:
                            print("[CALIB] Not enough samples; try again with better lighting.")

    # Cleanup
    cap.release()
    if session_writer is not None:
        session_writer.release()
    if display:
        cv2.destroyAllWindows()

    now_t = time.time() - t0
    export_ready(now_t + post_event_sec + 0.2)

    df = pd.DataFrame(rows)
    per_frame_csv = os.path.join(out_dir, f"{session_name}__per_frame_metrics.csv")
    df.to_csv(per_frame_csv, index=False)

    events_df = pd.DataFrame(events)
    events_csv = os.path.join(out_dir, f"{session_name}__events.csv")
    if len(events_df) > 0:
        events_df.to_csv(events_csv, index=False)
    else:
        events_csv = None

    return {
        "out_dir": out_dir,
        "session_name": session_name,
        "per_frame_csv": per_frame_csv,
        "events_csv": events_csv,
        "session_video": session_path if record_full_session else None,
        "df": df,
        "events_df": events_df,
        "ear_thresh_final": ear_thresh,
        "phone_detection_enabled": enable_phone_detection and (yolo is not None),
    }


if __name__ == "__main__":
    print("Starting Driver Monitor...")
    print("Tip: press 'c' to calibrate EAR (recommended for glasses).")
    result = run_driver_monitor(
        camera_index=0,
        out_dir="dams_output",
        display=True,
        resize_width=960,
        enhance_image=True,

        ear_thresh=0.22,
        perclos_window_sec=30.0,
        perclos_min_fill_sec=8.0,
        perclos_on_thresh=0.28,
        perclos_off_thresh=0.18,
        awake_reset_sec=1.0,
        microsleep_sec=1.2,

        mar_yawn_thresh=0.55,
        yawn_min_sec=0.8,

        yaw_turn_thresh_deg=25.0,
        pitch_turn_thresh_deg=18.0,
        turned_away_min_sec=1.0,

        pre_event_sec=2.0,
        post_event_sec=2.0,
        min_event_sec=1.0,

        record_full_session=True,

        enable_phone_detection=True,
        yolo_model_name="yolov8n.pt",
        phone_conf_thresh=0.35,
        phone_event_min_sec=1.0,
    )

    print("Done.")
    print("Saved to:", result["out_dir"])
    print("Events:")
    print(result["events_df"])