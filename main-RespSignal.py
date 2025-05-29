import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import deque
from scipy.signal import butter, filtfilt
import logging
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# ===== Logging setup =====
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# ===== Filter Functions =====
def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def dominant_frequency_fft(signal_data, fs):
    n = len(signal_data)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_data))
    peak_idx = np.argmax(fft_vals)
    return freqs[peak_idx] if fft_vals[peak_idx] > 0 else 0

# ===== Plotting Function =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=100)
fig.tight_layout(pad=3.0)
canvas = FigureCanvas(fig)

def draw_plot(times, raw, filtered):
    ax1.clear()
    ax2.clear()

    ax1.plot(times, raw, color='gray', label="Raw")
    ax2.plot(times, filtered, color='green', label="Filtered")

    ax1.set_title("Raw Shoulder Y")
    ax2.set_title("Filtered Respiratory Signal")
    ax1.set_ylabel("Y Pos")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time (s)")
    ax1.grid(True)
    ax2.grid(True)

    if times:
        tmin = max(0, times[-1] - 5)
        ax1.set_xlim(tmin, times[-1])
        ax2.set_xlim(tmin, times[-1])

    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 4)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

# ===== GUI Setup =====
root = tk.Tk()
root.title("Respiration Filter Control")

lowcut_var = tk.DoubleVar(value=0.1)
highcut_var = tk.DoubleVar(value=0.5)

control_frame = ttk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X)

ttk.Label(control_frame, text="Low Cutoff (Hz)").pack(side=tk.LEFT, padx=5)
tk.Scale(control_frame, variable=lowcut_var, from_=0.05, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT)

ttk.Label(control_frame, text="High Cutoff (Hz)").pack(side=tk.LEFT, padx=5)
tk.Scale(control_frame, variable=highcut_var, from_=0.3, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT)

# ===== MediaPipe Setup =====
model_path = "pose_landmarker_full.task"
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)
landmarker = PoseLandmarker.create_from_options(options)

# ===== Webcam Setup & Buffers =====
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_idx = 0

raw_y_buffer = deque(maxlen=300)
filtered_y_buffer = deque(maxlen=300)
time_buffer = deque(maxlen=300)

# ===== Processing Loop =====
def process_frame():
    global frame_idx
    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((frame_idx / fps) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        l_sh = landmarks[11]
        r_sh = landmarks[12]

        ly = int(l_sh.y * h)
        ry = int(r_sh.y * h)
        avg_y = (ly + ry) / 2

        raw_y_buffer.append(avg_y)
        time_buffer.append(frame_idx / fps)

        # Draw landmarks
        lx, rx = int(l_sh.x * w), int(r_sh.x * w)
        cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)
        cv2.circle(frame, (rx, ry), 4, (0, 255, 0), -1)
        cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 0), 2)

    if len(raw_y_buffer) >= 30:
        low = lowcut_var.get()
        high = highcut_var.get()
        if low < high:
            filtered = butter_bandpass(list(raw_y_buffer), low, high, fs=fps)
            filtered_y_buffer.append(filtered[-1])

            freq = dominant_frequency_fft(filtered, fs=fps)
            bpm = freq * 60
            logging.info(f"Dominant Frequency: {freq:.2f} Hz | Respiration Rate: {bpm:.1f} BPM")
            plot_img = draw_plot(list(time_buffer), list(raw_y_buffer), list(filtered))
            plot_img = cv2.resize(plot_img, (w, h))
            combined = np.hstack((frame, plot_img))
            cv2.imshow("Respiration Signal", combined)

        else:
            cv2.putText(frame, "Lowcut must be < Highcut!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Respiration Signal", frame)
    else:
        cv2.imshow("Respiration Signal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cleanup()
        return

    frame_idx += 1
    root.after(10, process_frame)

def cleanup():
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    root.destroy()

# ===== Start Application =====
root.after(0, process_frame)
root.mainloop()