import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import threading
import time

# Global variables
timestamps = []
y_positions = []
running = True

# Inisialisasi model pose
def init_pose_model(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1
    )
    return PoseLandmarker.create_from_options(options)

# Webcam + Pose Detection Thread
def webcam_thread(model_path):
    global timestamps, y_positions, running

    detector = init_pose_model(model_path)
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            height, width = frame.shape[:2]
            y_avg = int((left_shoulder.y + right_shoulder.y) * height / 2)

            elapsed = time.time() - start_time

            # Locking update
            timestamps.append(elapsed)
            y_positions.append(y_avg)

            # Batasi panjang data
            if len(timestamps) > 300:
                timestamps = timestamps[-300:]
                y_positions = y_positions[-300:]

            # Gambar landmark
            for lm in [left_shoulder, right_shoulder]:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Webcam Shoulder Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break


    cap.release()
    cv2.destroyAllWindows()
    detector.close()

# Realtime plotting (di thread utama agar stabil)
def plot_loop():
    global timestamps, y_positions, running
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'g-')
    ax.set_title("Shoulder Y vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y Position")

    while running:
        # Pastikan panjang data sinkron
        min_len = min(len(timestamps), len(y_positions))
        t = timestamps[-min_len:]
        y = y_positions[-min_len:]

        if t and y:
            t0 = t[0]
            t_rel = [ti - t0 for ti in t]
            line.set_data(t_rel, y)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        time.sleep(0.1)

    plt.ioff()
    plt.close()

# Main control
def main_realtime(model_path):
    global running
    try:
        t1 = threading.Thread(target=webcam_thread, args=(model_path,))
        t1.start()

        # Plot di thread utama
        plot_loop()

        t1.join()

    except KeyboardInterrupt:
        print("Interrupted by user. Stopping...")
        running = False


# Entry point
if __name__ == "__main__":
    model_path = "pose_landmarker_full.task"
    main_realtime(model_path)