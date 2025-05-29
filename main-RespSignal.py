import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import scipy.signal as signal
import tkinter as tk
from tkinter import ttk

def bandpass_filter(data, lowcut, highcut, fs=30.0, order=4):
    """Terapkan band-pass filter Butterworth ke data sinyal Y bahu."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def draw_plot(times, raw_y, filtered_y):
    ax1.clear()
    ax2.clear()

    ax1.plot(times, raw_y, label="Raw Y", color='gray')
    ax1.set_title("Raw Shoulder Y Position")
    ax1.set_ylabel("Y")
    ax1.grid(True)

    ax2.plot(times, filtered_y, label="Filtered", color='green')
    ax2.set_title("Filtered Respiratory Signal")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    if times:
        ax1.set_xlim(max(0, times[-1] - 5), times[-1])
        ax2.set_xlim(max(0, times[-1] - 5), times[-1])

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(int(height), int(width), 4)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

# === GUI Setup ===
root = tk.Tk()
root.title("Respiration Filter Control")

lowcut_var = tk.DoubleVar(value=0.1)
highcut_var = tk.DoubleVar(value=0.5)

control_frame = ttk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X)

ttk.Label(control_frame, text="Low Cutoff (Hz)").pack(side=tk.LEFT, padx=5)
lowcut_slider = tk.Scale(control_frame, variable=lowcut_var, from_=0.05, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=200)
lowcut_slider.pack(side=tk.LEFT)

ttk.Label(control_frame, text="High Cutoff (Hz)").pack(side=tk.LEFT, padx=5)
highcut_slider = tk.Scale(control_frame, variable=highcut_var, from_=0.3, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=200)
highcut_slider.pack(side=tk.LEFT)

# === MediaPipe Setup ===
model_path = "pose_landmarker_full.task"
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)
landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_idx = 0

# === Buffers ===
raw_buffer = deque(maxlen=300)
filtered_buffer = deque(maxlen=300)
time_buffer = deque(maxlen=300)

# === Matplotlib ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=100)
fig.tight_layout(pad=3.0)
canvas = FigureCanvas(fig)

# === Frame rendering loop ===
def process_frame():
    global frame_idx
    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(frame_idx * 1000 / fps)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        lmk = result.pose_landmarks[0]
        l_sh = lmk[11]
        r_sh = lmk[12]

        ly = int(l_sh.y * h)
        ry = int(r_sh.y * h)
        avg_y = (ly + ry) / 2

        raw_buffer.append(avg_y)
        time_buffer.append(frame_idx / fps)

        # Tampilkan landmark
        lx, rx = int(l_sh.x * w), int(r_sh.x * w)
        cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)
        cv2.circle(frame, (rx, ry), 4, (0, 255, 0), -1)
        cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 0), 2)

    if len(raw_buffer) >= 30:
        low = lowcut_var.get()
        high = highcut_var.get()
        if low < high:
            filtered = bandpass_filter(list(raw_buffer), lowcut=low, highcut=high, fs=fps)
            filtered_buffer.append(filtered[-1])
            plot_img = draw_plot(list(time_buffer), list(raw_buffer), list(filtered))
            plot_img = cv2.resize(plot_img, (w, h))
            combined = np.hstack((frame, plot_img))
            cv2.imshow("Respiration Signal", combined)
        else:
            cv2.putText(frame, "Lowcut must be < Highcut!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Respiration Signal", frame)
    else:
        cv2.imshow("Respiration Signal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        root.destroy()
        return

    frame_idx += 1
    root.after(10, process_frame)

root.after(0, process_frame)
root.mainloop()