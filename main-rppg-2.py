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
        
        # Enhanced filter parameters - better optimized for heart rate range
        self.lowcut = 0.7  # 42 BPM - wider range for better detection
        self.highcut = 2.5  # 150 BPM
        self.filter_order = 3  # Lower order for less ringing
        
        # Filter state variables for stability
        self.prev_filtered_value = 0
        self.last_filter_time = 0  # Track when we last successfully filtered
        
        # Heart rate calculation parameters
        self.current_hr = 0
        # Store HR as simple values, not tuples for cleaner handling
        self.hr_history = deque(maxlen=15)
        self.hr_timestamps = deque(maxlen=15)  # Separate timestamps for clarity
        self.last_valid_hr_time = 0  # Track when we last had a valid HR
        
        # Added smoothing parameters
        self.smooth_window_size = 9  # Smoothing window size
        
        # Wavelet transform parameters
        self.wavelet_name = 'sym4'  # Symlet wavelet - good for biomedical signals
        self.wavelet_level = 3      # Decomposition level
        
        # Dynamic peak detection parameters
        self.min_peak_distance = int(self.fps * 0.5)  # Minimum 0.5s between peaks (120 BPM max)
        self.peak_prominence = 0.3  # Starting prominence
        
        # Minimum signal quality needed for reliable HR estimation
        self.min_signal_quality = 1.2
        
        # Force a recalculation if too much time has passed
        self.force_recalc_interval = 1.0  # seconds
        self.last_calculation_time = 0
    
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
        
        # Check if we need to filter (at least every 33ms - ~30fps)
        current_time = time.time()
        if current_time - self.last_filter_time < 0.033 and len(self.filtered_rppg) > 0:
            return
        
        self.last_filter_time = current_time
        
        # Get recent samples for filtering
        analysis_window = min(int(self.fps * 4), len(self.rppg_signal))  # Use 4 seconds
        if analysis_window < 10:  # Need minimum data for meaningful filtering
            return
            
        recent_samples = list(self.rppg_signal)[-analysis_window:]
        
        try:
            # Stage 1: Butterworth bandpass filter with guard for short signals
            nyquist = self.fps / 2
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            
            # Apply zero-phase Butterworth filter with padding for stability
            padded_samples = np.pad(recent_samples, (5, 5), mode='edge')
            b, a = signal.butter(self.filter_order, [low, high], btype='bandpass')
            filtered_full = signal.filtfilt(b, a, padded_samples)
            filtered_segment = filtered_full[5:-5]  # Remove padding
            
            # Stage 2: Wavelet denoising if we have enough data
            if len(filtered_segment) > 2**self.wavelet_level + 2:
                filtered_segment = self._wavelet_denoise(filtered_segment)
            
            # Stage 3: Savitzky-Golay filter with careful window sizing
            window_length = min(15, len(filtered_segment) - 2)
            if window_length > 3:
                if window_length % 2 == 0:  # Must be odd
                    window_length -= 1
                filtered_segment = signal.savgol_filter(filtered_segment, window_length, 2)
            
            # Stage 4: Dynamic smoothing based on signal variance
            signal_var = np.var(filtered_segment)
            # More aggressive smoothing for noisier signals
            alpha = min(0.5, max(0.1, 0.3 / (1 + signal_var)))
            
            # Only take the latest value for real-time filtering
            latest_value = filtered_segment[-1]
            
            # Apply exponential smoothing
            if len(self.filtered_rppg) > 0:
                smoothed_value = (alpha * latest_value + 
                                 (1 - alpha) * self.prev_filtered_value)
            else:
                smoothed_value = latest_value
            
            # Update state for next iteration
            self.prev_filtered_value = smoothed_value
            
            # Add the filtered value to buffer
            self.filtered_rppg.append(smoothed_value)
            
        except Exception as e:
            print(f"Filtering error: {e}")
            # On error, copy the raw signal as fallback
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
        """
        Calculate heart rate from filtered rPPG signal with improved reliability.
        Uses multiple methods and robust peak detection to prevent sticking.
        """
        # Need enough data to calculate heart rate
        if len(self.filtered_rppg) < self.window_size:
            return
            
        current_time = time.time()
        
        # Don't recalculate too frequently unless forced
        time_since_last = current_time - self.last_calculation_time
        if time_since_last < 0.2 and time_since_last < self.force_recalc_interval:
            return
            
        # Force recalculation if it's been too long since last valid HR
        force_recalc = (current_time - self.last_valid_hr_time > 2.0)
        
        # Set the calculation timestamp
        self.last_calculation_time = current_time
        
        # Get data for analysis - use multiple window sizes for better accuracy
        short_window = min(int(self.fps * 3), len(self.filtered_rppg))  # 3 seconds
        medium_window = min(int(self.fps * 5), len(self.filtered_rppg))  # 5 seconds
        long_window = min(int(self.fps * 8), len(self.filtered_rppg))  # 8 seconds
        
        # Only proceed if we have enough data
        if short_window < self.fps * 1.5:  # Need at least 1.5 seconds
            return
        
        # Get recent data segments of different sizes
        short_segment = list(self.filtered_rppg)[-short_window:]
        medium_segment = list(self.filtered_rppg)[-medium_window:] if medium_window > short_window else short_segment
        long_segment = list(self.filtered_rppg)[-long_window:] if long_window > medium_window else medium_segment
        
        # Get corresponding timestamps
        recent_timestamps = list(self.timestamps)[-long_window:]
        
        try:
            # Process signal using multiple methods for robustness
            
            # Method 1: Standard peak detection on normalized signal
            # -----------------------------------------------------
            norm_signal = self._normalize_signal(medium_segment)
            
            # Evaluate signal quality
            signal_quality = np.abs(np.max(norm_signal) - np.min(norm_signal))
            print(f"Signal quality: {signal_quality:.2f}")
            
            # Adjust prominence based on signal quality
            prominence = 0.2  # Default
            if signal_quality > 2.5:
                prominence = 0.35  # Strong signal, be selective
            elif signal_quality < 1.5:
                prominence = 0.15  # Weak signal, be more permissive
            
            # Find peaks
            peaks, _ = signal.find_peaks(
                norm_signal,
                prominence=prominence, 
                distance=self.min_peak_distance,
                width=int(self.fps * 0.08)  # Minimum width
            )
            
            peak_count = len(peaks)
            print(f"Peaks detected: {peak_count}")
            
            # Method 2: Frequency domain analysis for confirmation
            # --------------------------------------------------
            freq_hr = self._frequency_analysis(long_segment)
            
            # Method 3: Autocorrelation method as additional check
            # --------------------------------------------------
            auto_hr = self._autocorrelation_hr(medium_segment)
            
            print(f"Multiple HR estimates - Peaks: {peak_count}, Freq: {freq_hr:.1f}, Auto: {auto_hr:.1f}")
            
            # Determine heart rate from combination of methods
            if peak_count >= 2:
                # Get inter-peak intervals for time-domain HR
                peak_intervals = []
                peak_hrs = []
                
                # Calculate time-based HR from peaks
                for i in range(1, len(peaks)):
                    samples_between = peaks[i] - peaks[i-1]
                    if samples_between > 0:
                        hr_from_peaks = 60.0 * self.fps / samples_between
                        if 40 <= hr_from_peaks <= 180:  # Valid physiological range
                            peak_hrs.append(hr_from_peaks)
                
                if peak_hrs:
                    # Use median for robustness
                    peak_based_hr = np.median(peak_hrs)
                    
                    # Check whether time-domain and frequency-domain methods agree
                    if freq_hr > 0 and abs(freq_hr - peak_based_hr) < 15:
                        # Methods agree, use time-domain (more precise)
                        final_hr = peak_based_hr
                    elif freq_hr > 0:
                        # Methods disagree, use weighted average
                        final_hr = (peak_based_hr * 2 + freq_hr) / 3
                    else:
                        # Freq analysis failed, use time-domain
                        final_hr = peak_based_hr
                        
                    print(f"Calculated HR: {final_hr:.1f} BPM")
                    
                    # Add to history with timestamp
                    self.hr_history.append(final_hr)
                    self.hr_timestamps.append(current_time)
                    self.last_valid_hr_time = current_time
                    
                    # Calculate stable output with weighted average
                    if len(self.hr_history) >= 3:
                        # Create weights favoring recent measurements
                        weights = np.linspace(0.5, 1.0, len(self.hr_history))
                        self.current_hr = np.average(self.hr_history, weights=weights)
                        print(f"Stable HR: {self.current_hr:.1f} BPM (from {len(self.hr_history)} measurements)")
                    else:
                        self.current_hr = final_hr
            
            # Fallback methods if peak detection fails
            elif freq_hr > 0 and (peak_count == 0 or force_recalc):
                print(f"Using frequency-based HR: {freq_hr:.1f} BPM")
                if auto_hr > 0 and abs(freq_hr - auto_hr) < 15:
                    # If autocorrelation agrees with frequency analysis
                    final_hr = (freq_hr + auto_hr) / 2
                else:
                    final_hr = freq_hr
                
                if 40 <= final_hr <= 180:
                    self.hr_history.append(final_hr)
                    self.hr_timestamps.append(current_time)
                    self.last_valid_hr_time = current_time
                    
                    # Simple average for frequency-based estimates
                    self.current_hr = np.mean(list(self.hr_history)[-5:])
        
        except Exception as e:
            print(f"Heart rate calculation error: {str(e)}")
            # Don't reset current_hr on error
            
    def _normalize_signal(self, signal_data):
        """Normalize signal to zero mean and unit variance."""
        signal_array = np.array(signal_data)
        return (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-9)
    
    def _frequency_analysis(self, signal_data):
        """Estimate heart rate using frequency domain analysis."""
        if len(signal_data) < self.fps * 2:  # Need at least 2 seconds
            return 0
            
        # Prepare signal - remove mean and apply window
        signal_array = np.array(signal_data)
        signal_array = signal_array - np.mean(signal_array)
        window = signal.windows.hann(len(signal_array))
        signal_array = signal_array * window
        
        # Compute FFT and frequency axis
        n_samples = len(signal_array)
        fft_data = np.abs(np.fft.rfft(signal_array))
        freqs = np.fft.rfftfreq(n_samples, 1/self.fps)
        
        # Limit to physiological heart rate range (40-180 BPM)
        valid_range = np.logical_and(freqs >= 40/60, freqs <= 180/60)
        if not np.any(valid_range):
            return 0
            
        valid_freqs = freqs[valid_range]
        valid_fft = fft_data[valid_range]
        
        # Find the dominant frequency
        if len(valid_fft) > 0:
            max_idx = np.argmax(valid_fft)
            dominant_freq = valid_freqs[max_idx]
            hr = dominant_freq * 60  # Convert Hz to BPM
            return hr
        
        return 0
    
    def _autocorrelation_hr(self, signal_data):
        """Estimate heart rate using autocorrelation."""
        if len(signal_data) < self.fps * 2:
            return 0
            
        # Prepare signal
        signal_array = np.array(signal_data)
        signal_array = signal_array - np.mean(signal_array)
        
        # Calculate autocorrelation
        autocorr = np.correlate(signal_array, signal_array, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only second half
        
        # Find peaks in autocorrelation
        min_lag = int(self.fps * 60/180)  # 180 BPM -> 0.33s
        max_lag = int(self.fps * 60/40)   # 40 BPM -> 1.5s
        
        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
            
        if max_lag <= min_lag:
            return 0
            
        # Focus on the physiologically relevant range
        autocorr_segment = autocorr[min_lag:max_lag]
        
        if len(autocorr_segment) < 3:
            return 0
            
        # Find first major peak (excluding the zero lag)
        peaks, _ = signal.find_peaks(autocorr_segment, height=0)
        
        if not len(peaks):
            return 0
            
        # Convert peak position to heart rate
        first_peak = peaks[0] + min_lag
        hr = 60 * self.fps / first_peak
        
        if 40 <= hr <= 180:
            return hr
            
        return 0

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
            
            # Update heart rate display with staleness detection
            current_time = time.time()
            hr_staleness = current_time - self.processor.last_valid_hr_time
            
            if face_detected:
                if hr > 0:
                    # Show HR with freshness indicator
                    if hr_staleness < 3.0:  # Fresh measurement
                        hr_text = f"Heart Rate: {hr:.1f} BPM"
                        self.hr_label.config(text=hr_text, foreground="black")
                    else:  # Stale measurement
                        hr_text = f"Heart Rate: {hr:.1f} BPM (updating...)"
                        self.hr_label.config(text=hr_text, foreground="#555555")
                    
                    status_text = "Status: Face Detected - Monitoring"
                else:
                    status_text = "Status: Face Detected - Calculating HR..."
                    if "BPM" not in self.hr_label.cget("text"):
                        self.hr_label.config(text="Heart Rate: Calculating...", foreground="black")
            else:
                status_text = "Status: No Face Detected"
                if "BPM" not in self.hr_label.cget("text") and "Calculating" not in self.hr_label.cget("text"):
                    self.hr_label.config(text="Heart Rate: -- BPM", foreground="black")
            
            self.status_label.config(text=status_text)
            
            # Check for stalled calculations
            if hr_staleness > 10.0 and "BPM" in self.hr_label.cget("text"):
                # Force recalculation by slightly changing processor state
                self.processor.force_recalc_interval = 0.1  # More frequent recalculation
                
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
