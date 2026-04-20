import cv2
import os

# ======================== CONFIG ========================
VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output.mp4"
LOGO_PATH = "logo.png"   # <-- put your logo in same folder

RANDOM_TEXT = "Rushikesh Patil - MTech Automotive"
RIBBON_TEXT = "Automotive python assignment, version 2"

# ======================== FUNCTION ========================
def process_video():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video")
        return

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # ======================== LOAD LOGO ========================
    logo = None
    if os.path.exists(LOGO_PATH):
        logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
        logo = cv2.resize(logo, (100, 100))

    # ======================== TEXT MOVEMENT ========================
    x, y = 50, height // 2
    speed_x, speed_y = 1, 1
    ribbon_height = 60

    frame_count = 0

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ======================== FPS ========================
        current_time = frame_count / fps
        fps_text = f"FPS: {fps:.1f}"

        # ======================== MOVE TEXT ========================
        x += speed_x
        y += speed_y

        (tw, th), _ = cv2.getTextSize(RANDOM_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)

        if x <= 0 or x >= width - tw:
            speed_x = -speed_x
        if y <= th or y >= height - ribbon_height - 10:
            speed_y = -speed_y

        # ======================== FADE EFFECT ========================
        alpha = (abs((frame_count % 100) - 50)) / 50  # fade in/out loop
        overlay = frame.copy()

        cv2.putText(overlay, RANDOM_TEXT, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3)

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # ======================== RIBBON ========================
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, height - ribbon_height),
                      (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)

        (rw, rh), _ = cv2.getTextSize(RIBBON_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        rx = (width - rw) // 2
        ry = height - (ribbon_height - rh) // 2

        cv2.putText(frame, RIBBON_TEXT, (rx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # ======================== LOGO WATERMARK ========================
        if logo is not None:
            lh, lw = logo.shape[:2]

            # Position (top-right corner)
            y1, y2 = 10, 10 + lh
            x1, x2 = width - lw - 10, width - 10

            if logo.shape[2] == 4:
                # PNG with transparency
                alpha_logo = logo[:, :, 3] / 255.0
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        alpha_logo * logo[:, :, c] +
                        (1 - alpha_logo) * frame[y1:y2, x1:x2, c]
                    )
            else:
                frame[y1:y2, x1:x2] = logo

        # ======================== FPS DISPLAY ========================
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print("✅ Done! Output saved as:", OUTPUT_PATH)


# ======================== RUN ========================
if __name__ == "__main__":
    process_video()