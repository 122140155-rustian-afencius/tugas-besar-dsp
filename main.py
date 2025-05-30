import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
from collections import deque
import scipy.signal as signal
import pywt
from datetime import datetime
import logging
from PIL import Image, ImageTk

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

class RPPGProcessor:
    """Real-time rPPG processor using POS method for heart rate detection."""
    
    def __init__(self, fps=30, window_length=1.6):
        self.fps = fps
        self.window_length = window_length
        self.window_size = int(window_length * fps)
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        
        # Signal buffers
        self.max_buffer_size = fps * 30
        self.r_signal = deque(maxlen=self.max_buffer_size)
        self.g_signal = deque(maxlen=self.max_buffer_size)
        self.b_signal = deque(maxlen=self.max_buffer_size)
        self.rppg_signal = deque(maxlen=self.max_buffer_size)
        self.filtered_rppg = deque(maxlen=self.max_buffer_size)
        self.timestamps = deque(maxlen=self.max_buffer_size)
        
        # Filter parameters
        self.lowcut = 0.7
        self.highcut = 2.5
        self.filter_order = 3
        
        # Heart rate calculation
        self.current_hr = 0
        self.hr_history = deque(maxlen=15)
        self.hr_timestamps = deque(maxlen=15)
        self.last_valid_hr_time = 0
        
        # Smoothing and wavelet parameters
        self.smooth_window_size = 9
        self.wavelet_name = 'sym4'
        self.wavelet_level = 3
        
        self.min_peak_distance = int(self.fps * 0.5)
        self.peak_prominence = 0.3
        self.min_signal_quality = 1.2
        self.force_recalc_interval = 1.0
        self.last_calculation_time = 0
        self.prev_filtered_value = 0
        self.last_filter_time = 0
    
    def cpu_POS(self, signal_array):
        """POS method implementation for rPPG signal extraction."""
        eps = 1e-9
        X = signal_array
        e, c, f = X.shape
        w = self.window_size
        
        if f < w:
            return np.zeros(f)
        
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)
        H = np.zeros((e, f))
        
        for n in range(w, f):
            m = n - w + 1
            Cn = X[:, :, m:(n + 1)]
            M = 1.0 / (np.mean(Cn, axis=2) + eps)
            M = np.expand_dims(M, axis=2)
            Cn = np.multiply(M, Cn)
            
            S = np.dot(Q, Cn)
            S = S[0, :, :, :]
            S = np.swapaxes(S, 0, 1)
            
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            Hn = np.add(S1, alpha * S2)
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
            
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
        
        return H[0, :]
    
    def extract_face_roi(self, frame):
        """Extract forehead region using MediaPipe face detection."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return None, None
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Extract forehead region
        forehead_height = int(height * 0.2)
        forehead_width = int(width * 0.8)
        forehead_x = x + (width - forehead_width) // 2
        forehead_y = y
        
        # Ensure coordinates are within frame
        forehead_x = max(0, min(forehead_x, w - forehead_width))
        forehead_y = max(0, min(forehead_y, h - forehead_height))
        
        roi = frame[forehead_y:forehead_y+forehead_height, forehead_x:forehead_x+forehead_width]
        bbox_coords = (forehead_x, forehead_y, forehead_width, forehead_height)
        
        return roi, bbox_coords
    
    def process_frame(self, frame):
        """Process frame for rPPG signal extraction."""
        processed_frame = frame.copy()
        face_detected = False
        
        roi, bbox_coords = self.extract_face_roi(frame)
        
        if roi is not None and roi.size > 0:
            face_detected = True
            x, y, width, height = bbox_coords
            
            cv2.rectangle(processed_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(processed_frame, "Face ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Extract RGB signals
            r_mean = np.mean(roi[:, :, 2])
            g_mean = np.mean(roi[:, :, 1])
            b_mean = np.mean(roi[:, :, 0])
            
            self.r_signal.append(r_mean)
            self.g_signal.append(g_mean)
            self.b_signal.append(b_mean)
            self.timestamps.append(time.time())
            
            if len(self.r_signal) >= self.window_size:
                self._update_rppg_signal()
                self._calculate_heart_rate()
        
        return processed_frame, face_detected, self.current_hr
    
    def _update_rppg_signal(self):
        """Update rPPG signal using POS method."""
        if len(self.r_signal) < self.window_size:
            return
        
        rgb_array = np.array([
            list(self.r_signal),
            list(self.g_signal),
            list(self.b_signal)
        ])
        rgb_array = rgb_array.reshape(1, 3, -1)
        
        rppg = self.cpu_POS(rgb_array)
        
        if len(rppg) > 0:
            self.rppg_signal.append(rppg[-1])
            
            if len(self.rppg_signal) >= self.window_size:
                self._apply_filter()
    
    def _apply_filter(self):
        """Apply filtering to rPPG signal."""
        if len(self.rppg_signal) < self.window_size:
            return
        
        current_time = time.time()
        if current_time - self.last_filter_time < 0.033 and len(self.filtered_rppg) > 0:
            return
        
        self.last_filter_time = current_time
        analysis_window = min(int(self.fps * 4), len(self.rppg_signal))
        
        if analysis_window < 10:
            return
            
        recent_samples = list(self.rppg_signal)[-analysis_window:]
        
        try:
            nyquist = self.fps / 2
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            
            padded_samples = np.pad(recent_samples, (5, 5), mode='edge')
            b, a = signal.butter(self.filter_order, [low, high], btype='bandpass')
            filtered_full = signal.filtfilt(b, a, padded_samples)
            filtered_segment = filtered_full[5:-5]
            
            if len(filtered_segment) > 2**self.wavelet_level + 2:
                filtered_segment = self._wavelet_denoise(filtered_segment)
            
            window_length = min(15, len(filtered_segment) - 2)
            if window_length > 3:
                if window_length % 2 == 0:
                    window_length -= 1
                filtered_segment = signal.savgol_filter(filtered_segment, window_length, 2)
            
            signal_var = np.var(filtered_segment)
            alpha = min(0.5, max(0.1, 0.3 / (1 + signal_var)))
            
            latest_value = filtered_segment[-1]
            
            if len(self.filtered_rppg) > 0:
                smoothed_value = (alpha * latest_value + 
                                 (1 - alpha) * self.prev_filtered_value)
            else:
                smoothed_value = latest_value
            
            self.prev_filtered_value = smoothed_value
            self.filtered_rppg.append(smoothed_value)
            
        except Exception as e:
            print(f"Filtering error: {e}")
            if len(self.rppg_signal) > 0:
                self.filtered_rppg.append(self.rppg_signal[-1])
    
    def _wavelet_denoise(self, data):
        """Apply wavelet denoising."""
        if len(data) <= 2**self.wavelet_level:
            return data
            
        coeffs = pywt.wavedec(data, self.wavelet_name, level=self.wavelet_level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        return pywt.waverec(coeffs, self.wavelet_name)
    
    def _calculate_heart_rate(self):
        """Calculate heart rate from filtered signal."""
        if len(self.filtered_rppg) < self.window_size:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_calculation_time
        if time_since_last < 0.2 and time_since_last < self.force_recalc_interval:
            return
        
        self.last_calculation_time = current_time
        
        medium_window = min(int(self.fps * 5), len(self.filtered_rppg))
        
        if medium_window < self.fps * 1.5:
            return
        
        medium_segment = list(self.filtered_rppg)[-medium_window:]
        
        try:
            norm_signal = self._normalize_signal(medium_segment)
            signal_quality = np.abs(np.max(norm_signal) - np.min(norm_signal))
            
            prominence = 0.2
            if signal_quality > 2.5:
                prominence = 0.35
            elif signal_quality < 1.5:
                prominence = 0.15
            
            peaks, _ = signal.find_peaks(
                norm_signal,
                prominence=prominence, 
                distance=self.min_peak_distance,
                width=int(self.fps * 0.08)
            )
            
            if len(peaks) >= 2:
                peak_hrs = []
                for i in range(1, len(peaks)):
                    samples_between = peaks[i] - peaks[i-1]
                    if samples_between > 0:
                        hr_from_peaks = 60.0 * self.fps / samples_between
                        if 40 <= hr_from_peaks <= 180:
                            peak_hrs.append(hr_from_peaks)
                
                if peak_hrs:
                    final_hr = np.median(peak_hrs)
                    self.hr_history.append(final_hr)
                    self.hr_timestamps.append(current_time)
                    self.last_valid_hr_time = current_time
                    
                    if len(self.hr_history) >= 3:
                        weights = np.linspace(0.5, 1.0, len(self.hr_history))
                        self.current_hr = np.average(self.hr_history, weights=weights)
                    else:
                        self.current_hr = final_hr
        
        except Exception as e:
            print(f"Heart rate calculation error: {str(e)}")
    
    def _normalize_signal(self, signal_data):
        """Normalize signal to zero mean and unit variance."""
        signal_array = np.array(signal_data)
        return (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-9)

class RespirationProcessor:
    """Real-time respiration processor using pose landmarks."""
    
    def __init__(self, fps=30):
        self.fps = fps
        
        # Initialize MediaPipe pose detection
        try:
            model_path = "pose_landmarker_full.task"
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            self.pose_available = True
        except:
            # Fallback to basic pose estimation
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_available = True
            self.landmarker = None
        
        # Signal buffers
        self.max_buffer_size = fps * 60  # 1 minute
        self.raw_y_buffer = deque(maxlen=self.max_buffer_size)
        self.filtered_y_buffer = deque(maxlen=self.max_buffer_size)
        self.time_buffer = deque(maxlen=self.max_buffer_size)
        
        # Filter parameters
        self.lowcut = 0.1   # Hz
        self.highcut = 0.5  # Hz
        
        # Respiration rate
        self.current_rr = 0
        self.frame_idx = 0
    
    def butter_bandpass(self, data, lowcut, highcut, fs, order=4):
        """Apply bandpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def dominant_frequency_fft(self, signal_data, fs):
        """Calculate dominant frequency using FFT."""
        n = len(signal_data)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(signal_data))
        peak_idx = np.argmax(fft_vals)
        return freqs[peak_idx] if fft_vals[peak_idx] > 0 else 0
    
    def process_frame(self, frame):
        """Process frame for respiration signal extraction."""
        processed_frame = frame.copy()
        pose_detected = False
        
        h, w = frame.shape[:2]
        
        if self.landmarker:
            # Use MediaPipe tasks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((self.frame_idx / self.fps) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                l_sh = landmarks[11]  # Left shoulder
                r_sh = landmarks[12]  # Right shoulder
                
                ly = int(l_sh.y * h)
                ry = int(r_sh.y * h)
                avg_y = (ly + ry) / 2
                
                # Draw landmarks
                lx, rx = int(l_sh.x * w), int(r_sh.x * w)
                cv2.circle(processed_frame, (lx, ly), 4, (255, 0, 0), -1)
                cv2.circle(processed_frame, (rx, ry), 4, (255, 0, 0), -1)
                cv2.line(processed_frame, (lx, ly), (rx, ry), (0, 255, 255), 2)
                cv2.putText(processed_frame, "Pose ROI", (lx, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                pose_detected = True
                
                # Add to buffers
                self.raw_y_buffer.append(avg_y)
                self.time_buffer.append(self.frame_idx / self.fps)
                
                # Process if enough data
                if len(self.raw_y_buffer) >= 30:
                    self._calculate_respiration_rate()
        else:
            # Use basic pose estimation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                l_sh = landmarks[11]  # Left shoulder
                r_sh = landmarks[12]  # Right shoulder
                
                ly = int(l_sh.y * h)
                ry = int(r_sh.y * h)
                avg_y = (ly + ry) / 2
                
                # Draw landmarks
                lx, rx = int(l_sh.x * w), int(r_sh.x * w)
                cv2.circle(processed_frame, (lx, ly), 4, (255, 0, 0), -1)
                cv2.circle(processed_frame, (rx, ry), 4, (255, 0, 0), -1)
                cv2.line(processed_frame, (lx, ly), (rx, ry), (0, 255, 255), 2)
                cv2.putText(processed_frame, "Pose ROI", (lx, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                pose_detected = True
                
                # Add to buffers
                self.raw_y_buffer.append(avg_y)
                self.time_buffer.append(self.frame_idx / self.fps)
                
                # Process if enough data
                if len(self.raw_y_buffer) >= 30:
                    self._calculate_respiration_rate()
        
        self.frame_idx += 1
        return processed_frame, pose_detected, self.current_rr
    
    def _calculate_respiration_rate(self):
        """Calculate respiration rate from shoulder movement."""
        if self.lowcut < self.highcut:
            try:
                filtered = self.butter_bandpass(
                    list(self.raw_y_buffer), 
                    self.lowcut, 
                    self.highcut, 
                    fs=self.fps
                )
                self.filtered_y_buffer.append(filtered[-1])
                
                # Calculate respiration rate
                freq = self.dominant_frequency_fft(filtered, fs=self.fps)
                self.current_rr = freq * 60  # Convert to BPM
                
                logging.info(f"Respiration Rate: {self.current_rr:.1f} BPM")
                
            except Exception as e:
                print(f"Respiration calculation error: {e}")
    
    def update_filter_params(self, lowcut, highcut):
        """Update filter parameters."""
        self.lowcut = lowcut
        self.highcut = highcut

class UnifiedVitalSignsApp:
    """Unified GUI application for rPPG and respiration monitoring."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unified Vital Signs Monitor - Heart Rate & Respiration")
        self.root.geometry("1400x900")
        
        # Application state
        self.is_running = False
        self.cap = None
        self.rppg_processor = RPPGProcessor()
        self.resp_processor = RespirationProcessor()
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
        
        # Video display
        self.video_label = None
        
        # Setup GUI
        self._setup_gui()
        self._setup_plots()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _setup_gui(self):
        """Setup the main GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status and readings
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT, padx=(20, 10))
        
        # Vital signs display
        vitals_frame = ttk.Frame(control_frame)
        vitals_frame.pack(side=tk.RIGHT)
        
        self.hr_label = ttk.Label(vitals_frame, text="Heart Rate: -- BPM", font=("Arial", 12, "bold"), foreground="red")
        self.hr_label.pack(side=tk.TOP)
        
        self.rr_label = ttk.Label(vitals_frame, text="Respiration: -- BPM", font=("Arial", 12, "bold"), foreground="blue")
        self.rr_label.pack(side=tk.TOP)
        
        # Content area with notebook for tabs
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame (left side)
        self.video_frame = ttk.LabelFrame(content_frame, text="Video Feed", padding="5")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Add a label to display the video feed
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for plots (right side)
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # rPPG tab
        self.rppg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rppg_frame, text="Heart Rate Signals")
        
        # Respiration tab
        self.resp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.resp_frame, text="Respiration Signals")
        
        # Respiration filter controls
        filter_frame = ttk.LabelFrame(self.resp_frame, text="Filter Controls", padding="5")
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.lowcut_var = tk.DoubleVar(value=0.1)
        self.highcut_var = tk.DoubleVar(value=0.5)
        
        ttk.Label(filter_frame, text="Low Cutoff (Hz)").pack(side=tk.LEFT, padx=5)
        tk.Scale(filter_frame, variable=self.lowcut_var, from_=0.05, to=0.5, 
                resolution=0.01, orient=tk.HORIZONTAL, length=150,
                command=self._update_resp_filter).pack(side=tk.LEFT)
        
        ttk.Label(filter_frame, text="High Cutoff (Hz)").pack(side=tk.LEFT, padx=5)
        tk.Scale(filter_frame, variable=self.highcut_var, from_=0.3, to=1.0, 
                resolution=0.01, orient=tk.HORIZONTAL, length=150,
                command=self._update_resp_filter).pack(side=tk.LEFT)
    
    def _update_resp_filter(self, value=None):
        """Update respiration filter parameters."""
        self.resp_processor.update_filter_params(
            self.lowcut_var.get(), 
            self.highcut_var.get()
        )
    
    def _setup_plots(self):
        """Setup matplotlib plots for both rPPG and respiration."""
        # rPPG plots
        self.rppg_fig, self.rppg_axes = plt.subplots(3, 1, figsize=(8, 10))
        self.rppg_fig.tight_layout(pad=3.0)
        
        # Initialize rPPG plot data
        self.rppg_time_data = deque(maxlen=300)
        self.r_data = deque(maxlen=300)
        self.g_data = deque(maxlen=300)
        self.b_data = deque(maxlen=300)
        self.rppg_data = deque(maxlen=300)
        self.filtered_rppg_data = deque(maxlen=300)
        
        # Setup rPPG subplots
        self.rppg_axes[0].set_title("RGB Signals")
        self.rppg_axes[0].set_ylabel("Amplitude")
        self.r_line, = self.rppg_axes[0].plot([], [], 'r-', label='Red', alpha=0.7)
        self.g_line, = self.rppg_axes[0].plot([], [], 'g-', label='Green', alpha=0.7)
        self.b_line, = self.rppg_axes[0].plot([], [], 'b-', label='Blue', alpha=0.7)
        self.rppg_axes[0].legend()
        self.rppg_axes[0].grid(True, alpha=0.3)
        
        self.rppg_axes[1].set_title("Raw rPPG Signal")
        self.rppg_axes[1].set_ylabel("Amplitude")
        self.rppg_line, = self.rppg_axes[1].plot([], [], 'k-', linewidth=1.5)
        self.rppg_axes[1].grid(True, alpha=0.3)
        
        self.rppg_axes[2].set_title("Filtered rPPG Signal")
        self.rppg_axes[2].set_xlabel("Time (s)")
        self.rppg_axes[2].set_ylabel("Amplitude")
        self.filtered_rppg_line, = self.rppg_axes[2].plot([], [], 'purple', linewidth=2)
        self.rppg_axes[2].grid(True, alpha=0.3)
        
        # Embed rPPG plots
        self.rppg_canvas = FigureCanvasTkAgg(self.rppg_fig, self.rppg_frame)
        self.rppg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Respiration plots
        self.resp_fig, self.resp_axes = plt.subplots(2, 1, figsize=(8, 6))
        self.resp_fig.tight_layout(pad=3.0)
        
        # Initialize respiration plot data
        self.resp_time_data = deque(maxlen=300)
        self.raw_resp_data = deque(maxlen=300)
        self.filtered_resp_data = deque(maxlen=300)
        
        # Setup respiration subplots
        self.resp_axes[0].set_title("Raw Shoulder Y Position")
        self.resp_axes[0].set_ylabel("Y Position")
        self.raw_resp_line, = self.resp_axes[0].plot([], [], 'gray', label='Raw')
        self.resp_axes[0].grid(True, alpha=0.3)
        
        self.resp_axes[1].set_title("Filtered Respiratory Signal")
        self.resp_axes[1].set_xlabel("Time (s)")
        self.resp_axes[1].set_ylabel("Amplitude")
        self.filtered_resp_line, = self.resp_axes[1].plot([], [], 'green', linewidth=2)
        self.resp_axes[1].grid(True, alpha=0.3)
        
        # Embed respiration plots
        self.resp_canvas = FigureCanvasTkAgg(self.resp_fig, self.resp_frame)
        self.resp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animation
        self.rppg_ani = FuncAnimation(self.rppg_fig, self._update_rppg_plots, interval=100, blit=False)
        self.resp_ani = FuncAnimation(self.resp_fig, self._update_resp_plots, interval=100, blit=False)
    
    def start_monitoring(self):
        """Start the unified monitoring process."""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera!")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            # Update GUI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Monitoring...")
            
            # Start processing frames
            self._process_frames()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_running = False
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Update GUI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        self.hr_label.config(text="Heart Rate: -- BPM")
        self.rr_label.config(text="Respiration: -- BPM")
        
        # Clear video display
        cv2.destroyAllWindows()
    
    def _capture_frames(self):
        """Capture frames from camera in separate thread."""
        start_time = time.time()
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time() - start_time
            
            try:
                self.frame_queue.put((frame, current_time), timeout=0.01)
            except queue.Full:
                pass
            
            time.sleep(1/30)
    
    def _process_frames(self):
        """Process captured frames for both rPPG and respiration."""
        if not self.is_running:
            return
        
        try:
            frame, timestamp = self.frame_queue.get_nowait()
            
            # Process frame for both rPPG and respiration
            rppg_frame, face_detected, hr = self.rppg_processor.process_frame(frame)
            resp_frame, pose_detected, rr = self.resp_processor.process_frame(rppg_frame)
            
            # Update display
            current_time = time.time()
            hr_staleness = current_time - self.rppg_processor.last_valid_hr_time
            
            # Update heart rate display
            if face_detected and hr > 0:
                if hr_staleness < 3.0:
                    self.hr_label.config(text=f"Heart Rate: {hr:.1f} BPM", foreground="red")
                else:
                    self.hr_label.config(text=f"Heart Rate: {hr:.1f} BPM (updating...)", foreground="#AA0000")
            elif face_detected:
                self.hr_label.config(text="Heart Rate: Calculating...", foreground="red")
            else:
                self.hr_label.config(text="Heart Rate: No Face", foreground="gray")
            
            # Update respiration display
            if pose_detected and rr > 0:
                self.rr_label.config(text=f"Respiration: {rr:.1f} BPM", foreground="blue")
            elif pose_detected:
                self.rr_label.config(text="Respiration: Calculating...", foreground="blue")
            else:
                self.rr_label.config(text="Respiration: No Pose", foreground="gray")
            
            # Update status
            status_parts = []
            if face_detected:
                status_parts.append("Face OK")
            if pose_detected:
                status_parts.append("Pose OK")
            
            if status_parts:
                self.status_label.config(text=f"Status: {' | '.join(status_parts)}")
            else:
                self.status_label.config(text="Status: No Detection")
            
            # Update plot data
            self._update_plot_data(timestamp)
            
            # Display video
            self._display_frame(resp_frame)
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Processing error: {e}")
        
        # Schedule next frame processing
        if self.is_running:
            self.root.after(33, self._process_frames)
    
    def _update_plot_data(self, timestamp):
        """Update data for both rPPG and respiration plots."""
        # Update rPPG data
        if len(self.rppg_processor.r_signal) > 0:
            self.rppg_time_data.append(timestamp)
            self.r_data.append(self.rppg_processor.r_signal[-1])
            self.g_data.append(self.rppg_processor.g_signal[-1])
            self.b_data.append(self.rppg_processor.b_signal[-1])
            
            if len(self.rppg_processor.rppg_signal) > 0:
                self.rppg_data.append(self.rppg_processor.rppg_signal[-1])
            else:
                self.rppg_data.append(0)
            
            if len(self.rppg_processor.filtered_rppg) > 0:
                self.filtered_rppg_data.append(self.rppg_processor.filtered_rppg[-1])
            else:
                self.filtered_rppg_data.append(0)
        
        # Update respiration data
        if len(self.resp_processor.raw_y_buffer) > 0:
            self.resp_time_data.append(timestamp)
            self.raw_resp_data.append(self.resp_processor.raw_y_buffer[-1])
            
            if len(self.resp_processor.filtered_y_buffer) > 0:
                self.filtered_resp_data.append(self.resp_processor.filtered_y_buffer[-1])
            else:
                self.filtered_resp_data.append(0)
    
    def _update_rppg_plots(self, frame):
        """Update rPPG plots."""
        if len(self.rppg_time_data) < 2:
            return
        
        time_array = np.array(self.rppg_time_data)
        
        # Update lines
        self.r_line.set_data(time_array, self.r_data)
        self.g_line.set_data(time_array, self.g_data)
        self.b_line.set_data(time_array, self.b_data)
        self.rppg_line.set_data(time_array, self.rppg_data)
        self.filtered_rppg_line.set_data(time_array, self.filtered_rppg_data)
        
        # Adjust limits
        for ax in self.rppg_axes:
            ax.relim()
            ax.autoscale_view()
        
        return [self.r_line, self.g_line, self.b_line, self.rppg_line, self.filtered_rppg_line]
    
    def _update_resp_plots(self, frame):
        """Update respiration plots."""
        if len(self.resp_time_data) < 2:
            return
        
        time_array = np.array(self.resp_time_data)
        
        # Update lines
        self.raw_resp_line.set_data(time_array, self.raw_resp_data)
        self.filtered_resp_line.set_data(time_array, self.filtered_resp_data)
        
        # Adjust limits
        for ax in self.resp_axes:
            ax.relim()
            ax.autoscale_view()
            
            # Show only last 5 seconds
            if len(time_array) > 0:
                tmin = max(0, time_array[-1] - 5)
                ax.set_xlim(tmin, time_array[-1])
        
        return [self.raw_resp_line, self.filtered_resp_line]
    
    def _display_frame(self, frame):
        """Display video frame in the GUI."""
        if self.video_label is not None:
            # Resize frame to fit the video label
            h, w = frame.shape[:2]
            # Get the current size of the video label
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            # If the label has been rendered and has a valid size
            if label_width > 1 and label_height > 1:
                # Calculate the aspect ratio
                aspect_ratio = w / h
                
                # Determine new dimensions to fit in label while preserving aspect ratio
                if label_width / label_height > aspect_ratio:
                    # Label is wider than needed
                    new_height = label_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Label is taller than needed
                    new_width = label_width
                    new_height = int(new_width / aspect_ratio)
                
                # Resize the frame
                resized_frame = cv2.resize(frame, (new_width, new_height))
            else:
                # Default size if label dimensions are not yet available
                resized_frame = cv2.resize(frame, (640, 480))
                
            # Convert the image to PIL format
            image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image=image)
            
            # Update the label
            self.video_label.configure(image=image)
            self.video_label.image = image  # Keep a reference to prevent garbage collection
    
    def on_closing(self):
        """Handle application closing."""
        self.stop_monitoring()
        if hasattr(self.resp_processor, 'landmarker') and self.resp_processor.landmarker:
            self.resp_processor.landmarker.close()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the unified application."""
        self.root.mainloop()

def main():
    """Main function to run the unified vital signs monitoring application."""
    try:
        app = UnifiedVitalSignsApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Critical Error", f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()