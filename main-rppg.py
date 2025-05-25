import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy import signal
from collections import deque
import threading
import time

class RPPGProcessor:
    """
    Kelas untuk memproses sinyal rPPG menggunakan metode POS (Plane-Orthogonal-to-Skin).
    
    Attributes:
        fps (int): Frame rate video
        window_size (int): Ukuran window untuk analisis sinyal
        overlap (float): Persentase overlap antar window
        hr_buffer (deque): Buffer untuk menyimpan nilai heart rate
    """
    
    def __init__(self, fps=30, window_size=150, overlap=0.8):
        """
        Inisialisasi processor rPPG.
        
        Args:
            fps (int): Frame rate video
            window_size (int): Ukuran window dalam frame
            overlap (float): Persentase overlap (0-1)
        """
        self.fps = fps
        self.window_size = window_size
        self.overlap = overlap
        self.rgb_buffer = deque(maxlen=window_size)
        self.hr_buffer = deque(maxlen=20)
        self.timestamps = deque(maxlen=window_size)
        
        # Parameter filter untuk POS
        self.pos_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
        
    def extract_roi(self, frame, face_coords):
        """
        Ekstraksi Region of Interest (ROI) dari area wajah.
        
        Args:
            frame (numpy.ndarray): Frame video
            face_coords (tuple): Koordinat wajah (x, y, w, h)
            
        Returns:
            numpy.ndarray: ROI yang diekstrak atau None jika tidak valid
        """
        if face_coords is None:
            return None
            
        x, y, w, h = face_coords
        
        # Definisi area ROI khusus di dahi
        # Perbarui koordinat untuk posisi dahi yang lebih tepat
        forehead_roi = frame[y + int(0.1*h):y + int(0.15*h), 
                          x + int(0.25*w):x + int(0.75*w)]
        
        if forehead_roi.size == 0:
            return None
            
        return forehead_roi
    
    def calculate_mean_rgb(self, roi):
        """
        Menghitung rata-rata nilai RGB dari ROI.
        
        Args:
            roi (numpy.ndarray): Region of Interest
            
        Returns:
            tuple: Nilai rata-rata (R, G, B) atau None jika ROI tidak valid
        """
        if roi is None or roi.size == 0:
            return None
            
        # Konversi BGR ke RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Hitung rata-rata untuk setiap channel
        mean_r = np.mean(roi_rgb[:, :, 0])
        mean_g = np.mean(roi_rgb[:, :, 1])
        mean_b = np.mean(roi_rgb[:, :, 2])
        
        return (mean_r, mean_g, mean_b)
    
    def apply_pos_algorithm(self, rgb_signals):
        """
        Implementasi algoritma POS untuk ekstraksi sinyal rPPG.
        
        Args:
            rgb_signals (numpy.ndarray): Sinyal RGB dengan shape (3, n_samples)
            
        Returns:
            numpy.ndarray: Sinyal rPPG yang diekstrak
        """
        if rgb_signals.shape[1] < 10:  # Minimum data requirement
            return np.array([])
            
        # Normalisasi sinyal RGB
        mean_rgb = np.mean(rgb_signals, axis=1, keepdims=True)
        norm_rgb = rgb_signals / (mean_rgb + 1e-8)
        
        # Aplikasi transformasi POS
        pos_signals = np.dot(self.pos_matrix, norm_rgb)
        
        # Kombinasi dua komponen POS untuk mendapatkan sinyal rPPG
        alpha = np.std(pos_signals[0, :]) / (np.std(pos_signals[1, :]) + 1e-8)
        rppg_signal = pos_signals[0, :] - alpha * pos_signals[1, :]
        
        return rppg_signal
    
    def bandpass_filter(self, signal_data, lowcut=0.7, highcut=3.5):
        """
        Aplikasi bandpass filter untuk isolasi sinyal heart rate.
        
        Args:
            signal_data (numpy.ndarray): Sinyal input
            lowcut (float): Frekuensi cutoff bawah (Hz)
            highcut (float): Frekuensi cutoff atas (Hz)
            
        Returns:
            numpy.ndarray: Sinyal yang telah difilter
        """
        if len(signal_data) < 20:
            return signal_data
            
        nyquist = self.fps / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Pastikan frekuensi dalam rentang valid
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            return filtered_signal
        except:
            return signal_data
    
    def estimate_heart_rate(self, rppg_signal):
        """
        Estimasi heart rate dari sinyal rPPG menggunakan FFT.
        
        Args:
            rppg_signal (numpy.ndarray): Sinyal rPPG
            
        Returns:
            float: Heart rate dalam BPM
        """
        if len(rppg_signal) < 50:
            return 0
            
        # FFT untuk analisis frekuensi
        fft_signal = np.fft.fft(rppg_signal)
        freqs = np.fft.fftfreq(len(rppg_signal), 1/self.fps)
        
        # Fokus pada rentang frekuensi heart rate (0.7-3.5 Hz)
        valid_indices = (freqs >= 0.7) & (freqs <= 3.5)
        valid_freqs = freqs[valid_indices]
        valid_fft = np.abs(fft_signal[valid_indices])
        
        if len(valid_fft) == 0:
            return 0
            
        # Cari frekuensi dominan
        peak_idx = np.argmax(valid_fft)
        heart_rate_hz = valid_freqs[peak_idx]
        heart_rate_bpm = heart_rate_hz * 60
        
        return heart_rate_bpm
    
    def process_frame(self, rgb_values, timestamp):
        """
        Proses frame untuk ekstraksi rPPG dan kalkulasi heart rate.
        
        Args:
            rgb_values (tuple): Nilai RGB rata-rata
            timestamp (float): Timestamp frame
            
        Returns:
            tuple: (rppg_signal, heart_rate, filtered_signal)
        """
        if rgb_values is None:
            return None, 0, None
            
        # Tambah data ke buffer
        self.rgb_buffer.append(rgb_values)
        self.timestamps.append(timestamp)
        
        if len(self.rgb_buffer) < self.window_size:
            return None, 0, None
            
        # Konversi buffer ke array numpy
        rgb_array = np.array(self.rgb_buffer).T  # Shape: (3, window_size)
        
        # Aplikasi algoritma POS
        rppg_signal = self.apply_pos_algorithm(rgb_array)
        
        if len(rppg_signal) == 0:
            return None, 0, None
            
        # Filter sinyal
        filtered_signal = self.bandpass_filter(rppg_signal)
        
        # Estimasi heart rate
        heart_rate = self.estimate_heart_rate(filtered_signal)
        
        # Smooth heart rate dengan moving average
        if 40 <= heart_rate <= 200:  # Validasi range heart rate normal
            self.hr_buffer.append(heart_rate)
            
        avg_hr = np.mean(self.hr_buffer) if self.hr_buffer else 0
        
        return rppg_signal, avg_hr, filtered_signal

class RPPGApp:
    """
    Aplikasi GUI untuk sistem rPPG real-time.
    
    Attributes:
        root (tk.Tk): Root window tkinter
        processor (RPPGProcessor): Instance processor rPPG
        face_cascade (cv2.CascadeClassifier): Classifier untuk deteksi wajah
        cap (cv2.VideoCapture): Video capture object
    """
    
    def __init__(self, root):
        """
        Inisialisasi aplikasi GUI.
        
        Args:
            root (tk.Tk): Root window tkinter
        """
        self.root = root
        self.root.title("Real-time rPPG Heart Rate Monitor")
        self.root.geometry("1200x800")
        
        # Inisialisasi komponen
        self.processor = RPPGProcessor()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.is_running = False
        
        # Data untuk plotting
        self.plot_data = deque(maxlen=300)
        self.plot_times = deque(maxlen=300)
        self.filtered_data = deque(maxlen=300)
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup interface GUI dengan kontrol dan visualisasi."""
        # Frame utama
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame kontrol
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tombol kontrol
        self.start_button = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Label untuk heart rate
        self.hr_label = ttk.Label(control_frame, text="Heart Rate: -- BPM", font=("Arial", 14, "bold"))
        self.hr_label.pack(side=tk.RIGHT)
        
        # Frame untuk video dan grafik
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup matplotlib untuk plot real-time
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)
        
        # Inisialisasi line plots dengan data kosong untuk update nanti
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=1, alpha=0.7)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=1.5)
        
        # Konfigurasi subplot dengan batasan awal
        self.ax1.set_title("Raw rPPG Signal")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.set_xlim(0, 10)  # Set batasan awal
        self.ax1.set_ylim(-1, 1)  # Set batasan awal
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(['Raw Signal'], loc='upper right')
        
        self.ax2.set_title("Filtered rPPG Signal")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.set_xlim(0, 10)  # Set batasan awal
        self.ax2.set_ylim(-1, 1)  # Set batasan awal
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(['Filtered Signal'], loc='upper right')
        
        # Embed matplotlib ke tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
    def start_camera(self):
        """Memulai capture video dan processing."""
        try:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
                
            # Set resolusi dan FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Camera started - Processing...")
            
            # Mulai thread untuk processing
            self.processing_thread = threading.Thread(target=self.process_video)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Mulai animasi plot dengan interval lebih cepat
            self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, 
                                    cache_frame_data=False, blit=False)
            self.canvas.draw()  # Initial draw
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            
    def stop_camera(self):
        """Menghentikan capture video dan processing."""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            
        cv2.destroyAllWindows()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.hr_label.config(text="Heart Rate: -- BPM")
        self.status_var.set("Camera stopped")
        
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
            
    def detect_face(self, frame):
        """
        Deteksi wajah menggunakan Haar Cascade.
        
        Args:
            frame (numpy.ndarray): Frame video
            
        Returns:
            tuple: Koordinat wajah terbesar atau None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Ambil wajah terbesar
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            return tuple(largest_face)
        return None
        
    def process_video(self):
        """Thread utama untuk processing video real-time."""
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            current_time = time.time() - start_time
            
            # Deteksi wajah
            face_coords = self.detect_face(frame)
            
            if face_coords is not None:
                # Gambar kotak di sekitar wajah
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Ekstrak ROI dan hitung rata-rata RGB
                roi = self.processor.extract_roi(frame, face_coords)
                rgb_values = self.processor.calculate_mean_rgb(roi)
                
                # Process untuk mendapatkan rPPG signal dan heart rate
                rppg_signal, heart_rate, filtered_signal = self.processor.process_frame(rgb_values, current_time)
                
                if rppg_signal is not None and len(rppg_signal) > 0:
                    # Update data untuk plotting - gunakan root.after untuk thread safety
                    self.root.after(0, self.update_plot_data, rppg_signal[-1], filtered_signal[-1] if filtered_signal is not None else 0, current_time)
                    
                    # Update heart rate display - gunakan root.after untuk thread safety
                    self.root.after(0, lambda hr=heart_rate: self.hr_label.config(text=f"Heart Rate: {hr:.1f} BPM"))
                
                # Gambar ROI pada frame - perbarui posisi untuk menunjukkan dahi
                if roi is not None:
                    roi_x = x + int(0.25*w)
                    roi_y = y + int(0.1*h)
                    roi_w = int(0.5*w)
                    roi_h = int(0.2*h)
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)
                    cv2.putText(frame, "ROI", (roi_x, roi_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Tampilkan frame
            cv2.imshow('rPPG Heart Rate Monitor', frame)
            
            # Break jika ESC ditekan
            if cv2.waitKey(1) & 0xFF == 27:
                self.stop_camera()
                break
                
    def update_plot_data(self, raw_value, filtered_value, timestamp):
        """
        Update data untuk plot secara thread-safe.
        
        Args:
            raw_value (float): Nilai rPPG mentah
            filtered_value (float): Nilai rPPG yang telah difilter
            timestamp (float): Waktu saat ini
        """
        self.plot_data.append(raw_value)
        self.filtered_data.append(filtered_value)
        self.plot_times.append(timestamp)
                
    def update_plot(self, frame):
        """
        Update plot real-time untuk sinyal rPPG.
        
        Args:
            frame: Frame number (tidak digunakan)
        """
        if len(self.plot_data) < 2:  # Minimal 2 poin untuk membuat garis
            return
            
        # Plot raw signal
        times = list(self.plot_times)
        raw_data = list(self.plot_data)
        filtered = list(self.filtered_data)
        
        # Update data pada line plots
        self.line1.set_data(times, raw_data)
        self.line2.set_data(times, filtered)
        
        # Dynamically set the xlim to show the most recent 10 seconds
        current_time = times[-1]
        self.ax1.set_xlim(max(0, current_time - 10), current_time + 0.5)
        self.ax2.set_xlim(max(0, current_time - 10), current_time + 0.5)
        
        # Set y limits to show the data properly
        if len(raw_data) > 1:
            raw_min, raw_max = min(raw_data), max(raw_data)
            if raw_min != raw_max:  # Avoid division by zero
                y_range = raw_max - raw_min
                self.ax1.set_ylim(raw_min - 0.2 * y_range, raw_max + 0.2 * y_range)
            
        if len(filtered) > 1:
            filt_min, filt_max = min(filtered), max(filtered)
            if filt_min != filt_max:  # Avoid division by zero
                y_range = filt_max - filt_min
                self.ax2.set_ylim(filt_min - 0.2 * y_range, filt_max + 0.2 * y_range)
        
        # Make sure to redraw the canvas
        self.canvas.draw()
        
    def on_closing(self):
        """Handler untuk penutupan aplikasi."""
        self.stop_camera()
        self.root.destroy()

def main():
    """
    Fungsi utama untuk menjalankan aplikasi rPPG.
    """
    # Buat root window
    root = tk.Tk()
    
    # Inisialisasi aplikasi
    app = RPPGApp(root)
    
    # Setup handler untuk penutupan window
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Jalankan aplikasi
    root.mainloop()

if __name__ == "__main__":
    main()
