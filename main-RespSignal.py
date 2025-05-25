import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# Inisialisasi pose landmarker
model_path = "pose_landmarker_full.task"
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = PoseLandmarker.create_from_options(options)

# Kamera dan buffer
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30  # fallback
frame_idx = 0
y_buffer = deque(maxlen=150)
time_buffer = deque(maxlen=150)

# Matplotlib figure dan canvas
fig, ax = plt.subplots(figsize=(5, 3))
canvas = FigureCanvas(fig)

def draw_plot(times, y_vals):
    ax.clear()
    ax.plot(times, y_vals, color='green')
    ax.set_ylim(min(y_vals) - 10, max(y_vals) + 10)
    ax.set_xlim(max(0, times[-1] - 5), times[-1] + 0.1)
    ax.set_title("Respiration (Shoulder Y Position)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y Pos")
    ax.grid(True)

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(int(height), int(width), 4)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image_rgb

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(frame_idx * 1000 / fps)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        l_sh = landmarks[11]
        r_sh = landmarks[12]

        lx, ly = int(l_sh.x * w), int(l_sh.y * h)
        rx, ry = int(r_sh.x * w), int(r_sh.y * h)
        avg_y = int((ly + ry) / 2)

        y_buffer.append(avg_y)
        time_buffer.append(frame_idx / fps)

        # Gambar titik bahu
        cv2.circle(frame, (lx, ly), 5, (0, 255, 0), -1)
        cv2.circle(frame, (rx, ry), 5, (0, 255, 0), -1)
        cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 0), 2)

    # Buat grafik
    if len(y_buffer) > 10:
        plot_img = draw_plot(list(time_buffer), list(y_buffer))
        plot_img = cv2.resize(plot_img, (w, h))  # samakan tinggi

        # Gabungkan webcam dan grafik secara horizontal (sejajar kanan)
        combined = np.hstack((frame, plot_img))
        cv2.imshow("Respiration Signal (Webcam + Plot)", combined)
    else:
        cv2.imshow("Respiration Signal (Webcam + Plot)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
landmarker.close()