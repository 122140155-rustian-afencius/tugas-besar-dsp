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
import pywt  # For wavelet denoising
from datetime import datetime

class RPPGProcessor:
    """
    Real-time rPPG processor using POS method.
    
    This class handles the core signal processing for remote photoplethysmography
    including face detection, RGB signal extraction, POS algorithm, and heart rate calculation.
    """
    
    def __init__(self, fps=30, window_length=1.6):
        """
        Initialize the rPPG processor.
        
        Args:
            fps (int): Camera frame rate
            window_length (float): Sliding window length in seconds for POS algorithm
        """
        self.fps = fps
        self.window_length = window_length
        self.window_size = int(window_length * fps)
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        
        # Signal buffers with maximum length
        self.max_buffer_size = fps * 30  # 30 seconds buffer
        self.r_signal = deque(maxlen=self.max_buffer_size)
        self.g_signal = deque(maxlen=self.max_buffer_size)
        self.b_signal = deque(maxlen=self.max_buffer_size)
        self.rppg_signal = deque(maxlen=self.max_buffer_size)
        self.filtered_rppg = deque(maxlen=self.max_buffer_size)
        self.timestamps = deque(maxlen=self.max_buffer_size)
        
        # Enhanced filter parameters - more specific to heart rate range
        self.lowcut = 0.8  # 48 BPM
        self.highcut = 2.0  # 120 BPM
        self.filter_order = 4  # Higher order for steeper cutoff
        
        # Filter state variables for stability
        self.prev_filtered_value = 0
        
        # Added smoothing parameters
        self.smooth_window_size = 9  # Smoothing window size
        
        # Heart rate calculation parameters - more robust
        self.current_hr = 0
        self.hr_history = deque(maxlen=15)  # Store more HR measurements for better averaging
        self.prev_peaks_timestamps = deque(maxlen=10)  # For calculating instantaneous HR
        
        # Wavelet transform parameters
        self.wavelet_name = 'sym4'  # Symlet wavelet - good for biomedical signals
        self.wavelet_level = 3      # Decomposition level
        
        # Dynamic peak detection parameters
        self.min_peak_distance = int(self.fps * 0.5)  # Minimum 0.5s between peaks (120 BPM max)
        self.peak_prominence = 0.3  # Starting prominence
        
    def cpu_POS(self, signal_array):
        """
        POS method implementation for rPPG signal extraction.
        
        Args:
            signal_array (np.ndarray): RGB signals with shape (1, 3, frames)
            
        Returns:
            np.ndarray: Extracted rPPG signal
            
        Reference:
            Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). 
            Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
        """
        eps = 1e-9
        X = signal_array
        e, c, f = X.shape  # e = #estimators, c = 3 rgb channels, f = #frames
        w = self.window_size  # window length
        
        if f < w:
            return np.zeros(f)
        
        # POS projection matrix
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)
        
        # Initialize output
        H = np.zeros((e, f))
        
        for n in range(w, f):
            # Sliding window start index
            m = n - w + 1
            
            # Temporal normalization
            Cn = X[:, :, m:(n + 1)]
            M = 1.0 / (np.mean(Cn, axis=2) + eps)
            M = np.expand_dims(M, axis=2)
            Cn = np.multiply(M, Cn)
            
            # Projection
            S = np.dot(Q, Cn)
            S = S[0, :, :, :]
            S = np.swapaxes(S, 0, 1)
            
            # Tuning
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            Hn = np.add(S1, alpha * S2)
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
            
            # Overlap-adding
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
        
        return H[0, :]
    
    def extract_face_roi(self, frame):
        """
        Extract face region of interest using MediaPipe.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            tuple: (roi_frame, bbox_coords) or (None, None) if no face detected
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return None, None
        
        detection = results.detections[0]  # Use first detected face
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        
        # Convert relative coordinates to pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Specifically target the forehead region only
        # Make the height smaller (20% of face height) and position it at the very top of face detection
        forehead_height = int(height * 0.2)  # Use top 20% of face height only
        forehead_width = int(width * 0.8)    # Use 80% of face width for forehead
        
        # Center horizontally, position at the very top of the detected face
        forehead_x = x + (width - forehead_width) // 2
        forehead_y = y  # Start directly at the top of the detected face
        
        # Additional adjustment to move higher if possible (into the hairline area)
        hairline_offset = int(height * 0.05)  # Try to go slightly above detected face
        if forehead_y - hairline_offset > 0:
            forehead_y -= hairline_offset
        
        # Ensure coordinates are within frame boundaries
        forehead_x = max(0, min(forehead_x, w - forehead_width))
        forehead_y = max(0, min(forehead_y, h - forehead_height))
        
        # Extract the forehead ROI
        roi = frame[forehead_y:forehead_y+forehead_height, forehead_x:forehead_x+forehead_width]
        bbox_coords = (forehead_x, forehead_y, forehead_width, forehead_height)
        
        return roi, bbox_coords
    
    def process_frame(self, frame):
        """
        Process a single frame to extract RGB signals and update rPPG.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            tuple: (processed_frame, face_detected, current_hr)
        """
        processed_frame = frame.copy()
        face_detected = False
        
        roi, bbox_coords = self.extract_face_roi(frame)
        
        if roi is not None and roi.size > 0:
            face_detected = True
            x, y, width, height = bbox_coords
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Extract RGB signals
            r_mean = np.mean(roi[:, :, 2])  # OpenCV uses BGR
            g_mean = np.mean(roi[:, :, 1])
            b_mean = np.mean(roi[:, :, 0])
            
            # Add to signal buffers
            self.r_signal.append(r_mean)
            self.g_signal.append(g_mean)
            self.b_signal.append(b_mean)
            self.timestamps.append(time.time())
            
            # Process rPPG if we have enough samples
            if len(self.r_signal) >= self.window_size:
                self._update_rppg_signal()
                self._calculate_heart_rate()
        
        return processed_frame, face_detected, self.current_hr
    
    def _update_rppg_signal(self):
        """Update rPPG signal using POS method."""
        if len(self.r_signal) < self.window_size:
            return
        
        # Prepare RGB signals for POS
        rgb_array = np.array([
            list(self.r_signal),
            list(self.g_signal),
            list(self.b_signal)
        ])
        rgb_array = rgb_array.reshape(1, 3, -1)
        
        # Apply POS method
        rppg = self.cpu_POS(rgb_array)
        
        # Update rPPG buffer (only add the latest sample)
        if len(rppg) > 0:
            self.rppg_signal.append(rppg[-1])
            
            # Apply bandpass filter if we have enough samples
            if len(self.rppg_signal) >= self.window_size:
                self._apply_filter()
    
    def _apply_filter(self):
        """
        Apply advanced multi-stage filtering to rPPG signal for stable heart rate detection.
        Uses a combination of bandpass filtering, wavelet denoising, and smoothing.
        """
        if len(self.rppg_signal) < self.window_size:
            return
        
        # Get recent samples for filtering - use a longer segment for better frequency response
        analysis_window = min(int(self.fps * 4), len(self.rppg_signal))  # Use 4 seconds or all available data
        recent_samples = list(self.rppg_signal)[-analysis_window:]
        
        try:
            # Stage 1: Butterworth bandpass filter - focused on heart rate frequency band
            nyquist = self.fps / 2
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            
            # Apply zero-phase Butterworth filter (more stable than standard)
            b, a = signal.butter(self.filter_order, [low, high], btype='bandpass')
            filtered_segment = signal.filtfilt(b, a, recent_samples)
            
            # Stage 2: Wavelet denoising (effective at removing high-frequency noise)
            filtered_segment = self._wavelet_denoise(filtered_segment)
            
            # Stage 3: Savitzky-Golay filter (preserves peaks better than moving average)
            window_length = min(15, len(filtered_segment) - 2)
            if window_length % 2 == 0:  # Must be odd
                window_length -= 1
            if window_length >= 3:  # Must be at least 3
                filtered_segment = signal.savgol_filter(filtered_segment, window_length, 2)
            
            # Stage 4: Apply exponential smoothing for stable transitions
            alpha = 0.3  # Smoothing factor (0-1), higher = less smoothing
            last_value = self.prev_filtered_value if len(self.filtered_rppg) > 0 else filtered_segment[-1]
            smoothed_value = alpha * filtered_segment[-1] + (1 - alpha) * last_value
            
            # Update state
            self.prev_filtered_value = smoothed_value
            
            # Add the filtered/smoothed value to the buffer
            self.filtered_rppg.append(smoothed_value)
            
        except Exception as e:
            print(f"Advanced filter error: {e}")
            if len(self.rppg_signal) > 0:
                self.filtered_rppg.append(self.rppg_signal[-1])
    
    def _wavelet_denoise(self, data):
        """
        Apply wavelet denoising to the signal.
        
        Args:
            data (np.ndarray): Input signal
            
        Returns:
            np.ndarray: Denoised signal
        """
        # Ensure enough data for wavelet decomposition
        if len(data) <= 2**self.wavelet_level:
            return data
            
        # Wavelet decomposition
        coeffs = pywt.wavedec(data, self.wavelet_name, level=self.wavelet_level)
        
        # Threshold calculation - adaptive based on signal statistics
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust estimate of noise
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Apply soft thresholding to detail coefficients (keep approximation)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # Reconstruct signal
        return pywt.waverec(coeffs, self.wavelet_name)
    
    def _calculate_heart_rate(self):
        """Calculate heart rate from filtered rPPG signal using robust peak detection."""
        if len(self.filtered_rppg) < self.window_size:
            return
        
        # Get recent filtered samples - use a longer segment for better peak detection
        analysis_window = min(int(self.fps * 8), len(self.filtered_rppg))  # 8 seconds of data
        recent_filtered = list(self.filtered_rppg)[-analysis_window:]
        recent_timestamps = list(self.timestamps)[-analysis_window:]
        
        # Normalize signal
        signal_array = np.array(recent_filtered)
        signal_array = (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-9)
        
        # Dynamic adjustment of peak detection parameters based on signal quality
        signal_quality = np.abs(np.max(signal_array) - np.min(signal_array))
        
        # Lower prominence for better peak detection
        prominence = 0.2  # Default prominence (reduced from 0.25)
        
        # Debug info
        print(f"Signal quality: {signal_quality:.2f}, Analysis window: {len(recent_filtered)} samples")
        
        try:
            # Find peaks with adaptive parameters
            peaks, peak_props = signal.find_peaks(
                signal_array, 
                prominence=prominence,
                distance=self.min_peak_distance,
                width=int(self.fps * 0.1)  # Minimum peak width
            )
            
            # Debug info
            print(f"Peaks detected: {len(peaks)}")
            
            # Calculate heart rate even with fewer peaks for faster feedback
            if len(peaks) >= 1:
                # Calculate heart rate directly from peaks frequency
                # Number of peaks / time window in minutes
                window_duration_minutes = len(recent_filtered) / (self.fps * 60)
                direct_hr = len(peaks) / window_duration_minutes
                
                # Only use direct calculation when we have few peaks
                if len(peaks) < 3 and 40 <= direct_hr <= 180:
                    print(f"Using direct HR calculation: {direct_hr:.1f} BPM")
                    self.hr_history.append(direct_hr)
                    self.current_hr = np.mean(list(self.hr_history))
                    return
                    
                # Only continue with time-based calculation if we have enough peaks
                if len(peaks) >= 2:
                    # Get peak timestamps
                    peak_times = [recent_timestamps[i] for i in peaks if i < len(recent_timestamps)]
                    
                    if len(peak_times) >= 2:
                        # Calculate instantaneous heart rates from adjacent peaks
                        inst_hrs = []
                        for i in range(1, len(peak_times)):
                            time_diff = peak_times[i] - peak_times[i-1]
                            if 0.3 <= time_diff <= 1.5:  # Valid range: 40-200 BPM
                                inst_hr = 60.0 / time_diff
                                inst_hrs.append(inst_hr)
                        
                        # Debug info
                        print(f"Instantaneous HRs: {[round(hr, 1) for hr in inst_hrs]}")
                        
                        # If we have instantaneous measurements
                        if inst_hrs:
                            # Remove outliers (outside 1.5 IQR)
                            if len(inst_hrs) > 3:
                                q1, q3 = np.percentile(inst_hrs, [25, 75])
                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                inst_hrs = [hr for hr in inst_hrs if lower_bound <= hr <= upper_bound]
                            
                            # Calculate median HR (more robust than mean)
                            if inst_hrs:
                                median_hr = np.median(inst_hrs)
                                
                                # Only accept if within physiological range
                                if 40 <= median_hr <= 180:
                                    print(f"Adding HR to history: {median_hr:.1f} BPM")
                                    self.hr_history.append(median_hr)
                                    
                                    # If we have enough history, use it; otherwise use the median
                                    if len(self.hr_history) > 2:
                                        # Calculate weighted moving average for stable output
                                        weights = np.linspace(0.5, 1.0, len(self.hr_history))
                                        weighted_hrs = weights * np.array(list(self.hr_history))
                                        self.current_hr = np.sum(weighted_hrs) / np.sum(weights)
                                    else:
                                        self.current_hr = median_hr
                                    
                                    print(f"Current HR: {self.current_hr:.1f} BPM")
                                    
            # If we still don't have a valid heart rate but have a history, use the history
            if self.current_hr == 0 and len(self.hr_history) > 0:
                self.current_hr = np.mean(list(self.hr_history))
                
        except Exception as e:
            print(f"Heart rate calculation error: {e}")
            # Don't reset current_hr to 0 if there's an error - keep the last valid value

class RPPGApp:
    """
    Main GUI application for real-time rPPG monitoring.
    
    This class manages the user interface, video capture, real-time plotting,
    and coordinates between different components of the system.
    """
    
    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("Real-time rPPG Heart Rate Monitor")
        self.root.geometry("1200x800")
        
        # Application state
        self.is_running = False
        self.cap = None
        self.processor = RPPGProcessor()
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
        
        # GUI setup
        self._setup_gui()
        self._setup_plots()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _setup_gui(self):
        """Set up the graphical user interface."""
        # Main frame
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
        
        # Status labels
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT, padx=(20, 10))
        
        self.hr_label = ttk.Label(control_frame, text="Heart Rate: -- BPM", font=("Arial", 12, "bold"))
        self.hr_label.pack(side=tk.RIGHT)
        
        # Video and plots frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(content_frame, text="Video Feed", padding="5")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Plots frame
        self.plots_frame = ttk.LabelFrame(content_frame, text="Real-time Signals", padding="5")
        self.plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def _setup_plots(self):
        """Set up matplotlib plots for real-time visualization."""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 10))
        self.fig.tight_layout(pad=3.0)
        
        # Initialize plot data
        self.time_data = deque(maxlen=300)  # 10 seconds at 30 fps
        self.r_data = deque(maxlen=300)
        self.g_data = deque(maxlen=300)
        self.b_data = deque(maxlen=300)
        self.rppg_data = deque(maxlen=300)
        self.filtered_data = deque(maxlen=300)
        
        # Set up subplots
        self.axes[0].set_title("RGB Signals")
        self.axes[0].set_ylabel("Amplitude")
        self.r_line, = self.axes[0].plot([], [], 'r-', label='Red', alpha=0.7)
        self.g_line, = self.axes[0].plot([], [], 'g-', label='Green', alpha=0.7)
        self.b_line, = self.axes[0].plot([], [], 'b-', label='Blue', alpha=0.7)
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)
        
        self.axes[1].set_title("Raw rPPG Signal")
        self.axes[1].set_ylabel("Amplitude")
        self.rppg_line, = self.axes[1].plot([], [], 'k-', linewidth=1.5)
        self.axes[1].grid(True, alpha=0.3)
        
        self.axes[2].set_title("Filtered rPPG Signal")
        self.axes[2].set_xlabel("Time (s)")
        self.axes[2].set_ylabel("Amplitude")
        self.filtered_line, = self.axes[2].plot([], [], 'purple', linewidth=2)
        self.axes[2].grid(True, alpha=0.3)
        
        # Embed plots in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animation
        self.ani = FuncAnimation(self.fig, self._update_plots, interval=100, blit=False)
    
    def start_monitoring(self):
        """Start the real-time monitoring process."""
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
        
        # Clear video display
        cv2.destroyAllWindows()
    
    def _capture_frames(self):
        """Capture frames from camera in separate thread."""
        start_time = time.time()
        frame_count = 0
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Add timestamp
            current_time = time.time() - start_time
            
            try:
                self.frame_queue.put((frame, current_time), timeout=0.01)
            except queue.Full:
                # Skip frame if queue is full
                pass
            
            frame_count += 1
            time.sleep(1/30)  # Maintain ~30 FPS
    
    def _process_frames(self):
        """Process captured frames and update display."""
        if not self.is_running:
            return
        
        try:
            # Get frame from queue
            frame, timestamp = self.frame_queue.get_nowait()
            
            # Process frame
            processed_frame, face_detected, hr = self.processor.process_frame(frame)
            
            # Update heart rate display
            if face_detected:
                if hr > 0:
                    self.hr_label.config(text=f"Heart Rate: {hr:.1f} BPM")
                    status_text = "Status: Face Detected - Monitoring"
                else:
                    # Still show we're detecting, just waiting for heart rate
                    status_text = "Status: Face Detected - Calculating HR..."
                    # Keep the existing HR text if it already shows a value
                    if "BPM" not in self.hr_label.cget("text"):
                        self.hr_label.config(text="Heart Rate: Calculating...")
            else:
                status_text = "Status: No Face Detected"
                # Keep the existing HR text if it shows a value
                if "BPM" not in self.hr_label.cget("text"):
                    self.hr_label.config(text="Heart Rate: -- BPM")
            
            self.status_label.config(text=status_text)
            
            # Update plot data
            self._update_plot_data(timestamp)
            
            # Display video
            self._display_frame(processed_frame)
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Processing error: {e}")
        
        # Schedule next frame processing
        if self.is_running:
            self.root.after(33, self._process_frames)  # ~30 FPS
    
    def _update_plot_data(self, timestamp):
        """Update data for real-time plots."""
        if len(self.processor.r_signal) > 0:
            self.time_data.append(timestamp)
            self.r_data.append(self.processor.r_signal[-1])
            self.g_data.append(self.processor.g_signal[-1])
            self.b_data.append(self.processor.b_signal[-1])
            
            if len(self.processor.rppg_signal) > 0:
                self.rppg_data.append(self.processor.rppg_signal[-1])
            else:
                self.rppg_data.append(0)
            
            if len(self.processor.filtered_rppg) > 0:
                self.filtered_data.append(self.processor.filtered_rppg[-1])
            else:
                self.filtered_data.append(0)
    
    def _update_plots(self, frame):
        """Update matplotlib plots with current data."""
        if len(self.time_data) < 2:
            return
        
        time_array = np.array(self.time_data)
        
        # Update RGB signals
        self.r_line.set_data(time_array, self.r_data)
        self.g_line.set_data(time_array, self.g_data)
        self.b_line.set_data(time_array, self.b_data)
        
        # Update rPPG signals
        self.rppg_line.set_data(time_array, self.rppg_data)
        self.filtered_line.set_data(time_array, self.filtered_data)
        
        # Adjust plot limits
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        
        return [self.r_line, self.g_line, self.b_line, self.rppg_line, self.filtered_line]
    
    def _display_frame(self, frame):
        """Display video frame in OpenCV window."""
        cv2.imshow('rPPG Monitor - Press Q to focus on plots', frame)
        cv2.waitKey(1)
    
    def on_closing(self):
        """Handle application closing."""
        self.stop_monitoring()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    """
    Main function to run the rPPG monitoring application.
    
    This function initializes and starts the GUI application for real-time
    remote photoplethysmography monitoring using webcam input.
    """
    try:
        # Create and run the application
        app = RPPGApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Critical Error", f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()
