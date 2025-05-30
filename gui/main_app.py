"""
Main application GUI for unified vital signs monitoring.

This module contains the main application class that coordinates the GUI,
video processing, and real-time visualization for both heart rate and
respiration monitoring.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
from PIL import Image, ImageTk
from typing import Optional, Tuple

from processors import RPPGProcessor, RespirationProcessor
from gui.plot_manager import PlotManager


class UnifiedVitalSignsApp:
    """
    Unified GUI application for real-time vital signs monitoring.
    
    This application provides a comprehensive interface for monitoring both
    heart rate (using rPPG) and respiration rate (using pose detection)
    from video input with real-time visualization and controls.
    """
    
    def __init__(self):
        """Initialize the unified vital signs monitoring application."""
        # Setup main window
        self._setup_main_window()
        
        # Initialize application state
        self._initialize_application_state()
        
        # Initialize signal processors
        self._initialize_processors()
        
        # Setup GUI components
        self._setup_gui_components()
        
        # Initialize threading components
        self._initialize_threading()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _setup_main_window(self) -> None:
        """Setup the main application window with proper sizing and title."""
        self.root = tk.Tk()
        self.root.title("Heart Rate & Respiration Realtime Monitoring")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)  # Set minimum window size
        
        # Configure main window style
        style = ttk.Style()
        style.theme_use('clam')  # Use a modern theme
    
    def _initialize_application_state(self) -> None:
        """Initialize application state variables."""
        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_label: Optional[tk.Label] = None
    
    def _initialize_processors(self) -> None:
        """Initialize signal processing modules."""
        # Initialize rPPG processor for heart rate detection
        self.rppg_processor = RPPGProcessor(fps=30, window_length=1.6)
        
        # Initialize respiration processor for breathing analysis
        self.resp_processor = RespirationProcessor(fps=30)
    
    def _setup_gui_components(self) -> None:
        """Setup all GUI components and layout."""
        # Create main container frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Setup control panel
        self._setup_control_panel(main_frame)
        
        # Setup content area with video and plots
        self._setup_content_area(main_frame)
    
    def _setup_control_panel(self, parent: tk.Widget) -> None:
        """
        Setup the control panel with buttons, status, and vital signs display.
        
        Args:
            parent: Parent widget to contain the control panel
        """
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Control buttons
        self.start_button = ttk.Button(
            button_frame, 
            text="Start Monitoring", 
            command=self.start_monitoring,
            style='Accent.TButton'
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="Stop Monitoring", 
            command=self.stop_monitoring, 
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_button = ttk.Button(
            button_frame, 
            text="Reset Signals", 
            command=self.reset_signals
        )
        self.reset_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status display
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        self.status_label = ttk.Label(
            status_frame, 
            text="Status: Ready", 
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Vital signs display
        vitals_frame = ttk.Frame(control_frame)
        vitals_frame.pack(side=tk.RIGHT)
        
        self.hr_label = ttk.Label(
            vitals_frame, 
            text="Heart Rate: -- BPM", 
            font=("Arial", 14, "bold"), 
            foreground="red"
        )
        self.hr_label.pack(side=tk.TOP, pady=(0, 5))
        
        self.rr_label = ttk.Label(
            vitals_frame, 
            text="Respiration: -- BPM", 
            font=("Arial", 14, "bold"), 
            foreground="blue"
        )
        self.rr_label.pack(side=tk.TOP)
    
    def _setup_content_area(self, parent: tk.Widget) -> None:
        """
        Setup the main content area with video display and plots.
        
        Args:
            parent: Parent widget to contain the content area
        """
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup video display area
        self._setup_video_area(content_frame)
        
        # Setup plots area with tabbed interface
        self._setup_plots_area(content_frame)
    
    def _setup_video_area(self, parent: tk.Widget) -> None:
        """
        Setup video display area with proper sizing.
        
        Args:
            parent: Parent widget to contain the video area
        """
        # Create fixed-width container for video
        video_container = ttk.Frame(parent, width=720)
        video_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        video_container.pack_propagate(False)  # Maintain fixed width
        
        # Video display frame
        self.video_frame = ttk.LabelFrame(video_container, text="Live Video Feed", padding="5")
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display label
        self.video_label = tk.Label(self.video_frame, bg='black', text="No Video")
        self.video_label.pack(fill=tk.BOTH, expand=True)
    
    def _setup_plots_area(self, parent: tk.Widget) -> None:
        """
        Setup plots area with tabbed interface for different signal types.
        
        Args:
            parent: Parent widget to contain the plots area
        """
        # Create notebook for tabbed plots interface
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create tabs for different signal types
        self._create_rppg_tab()
        self._create_respiration_tab()
        
        # Initialize plot manager
        self.plot_manager = PlotManager(self.rppg_frame, self.resp_plots_frame)
    
    def _create_rppg_tab(self) -> None:
        """Create the heart rate (rPPG) monitoring tab."""
        self.rppg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rppg_frame, text="Heart Rate Signals")
    
    def _create_respiration_tab(self) -> None:
        """Create the respiration monitoring tab with filter controls."""
        self.resp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.resp_frame, text="Respiration Signals")
        
        # Filter controls for respiration
        self._setup_respiration_controls()
        
        # Plots container
        self.resp_plots_frame = ttk.Frame(self.resp_frame)
        self.resp_plots_frame.pack(fill=tk.BOTH, expand=True)
    
    def _setup_respiration_controls(self) -> None:
        """Setup filter controls for respiration signal processing."""
        filter_frame = ttk.LabelFrame(self.resp_frame, text="Filter Controls", padding="5")
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Initialize filter variables
        self.lowcut_var = tk.DoubleVar(value=0.1)
        self.highcut_var = tk.DoubleVar(value=0.5)
        
        # Low cutoff control
        ttk.Label(filter_frame, text="Low Cutoff (Hz):").pack(side=tk.LEFT, padx=5)
        low_scale = tk.Scale(
            filter_frame, 
            variable=self.lowcut_var, 
            from_=0.05, 
            to=0.5,
            resolution=0.01, 
            orient=tk.HORIZONTAL, 
            length=150,
            command=self._update_resp_filter
        )
        low_scale.pack(side=tk.LEFT, padx=5)
        
        # High cutoff control
        ttk.Label(filter_frame, text="High Cutoff (Hz):").pack(side=tk.LEFT, padx=5)
        high_scale = tk.Scale(
            filter_frame, 
            variable=self.highcut_var, 
            from_=0.3, 
            to=1.0,
            resolution=0.01, 
            orient=tk.HORIZONTAL, 
            length=150,
            command=self._update_resp_filter
        )
        high_scale.pack(side=tk.LEFT, padx=5)
    
    def _update_resp_filter(self, value=None) -> None:
        """
        Update respiration filter parameters when controls change.
        
        Args:
            value: New filter value (unused but required by Scale callback)
        """
        try:
            self.resp_processor.update_filter_params(
                self.lowcut_var.get(), 
                self.highcut_var.get()
            )
        except ValueError as e:
            messagebox.showwarning("Filter Error", str(e))
    
    def _initialize_threading(self) -> None:
        """Initialize threading components for video capture and processing."""
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self) -> None:
        """
        Start the unified vital signs monitoring process.
        
        This method initializes the camera, starts video capture thread,
        and begins real-time processing of both heart rate and respiration.
        """
        try:
            # Initialize camera with error handling
            self._initialize_camera()
            
            # Update application state
            self.is_running = True
            
            # Start video capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_frames, 
                daemon=True
            )
            self.capture_thread.start()
            
            # Update GUI state
            self._update_gui_for_monitoring_start()
            
            # Begin frame processing loop
            self._process_frames()
            
        except Exception as e:
            messagebox.showerror("Startup Error", f"Failed to start monitoring: {str(e)}")
            self.stop_monitoring()
    
    def _initialize_camera(self) -> None:
        """
        Initialize camera with optimal settings.
        
        Raises:
            RuntimeError: If camera cannot be accessed
        """
        # Try different camera indices
        for camera_index in [1, 0, 2]:
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                break
        else:
            raise RuntimeError("Cannot access any camera!")
        
        # Configure camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    def _update_gui_for_monitoring_start(self) -> None:
        """Update GUI elements when monitoring starts."""
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Monitoring...")
    
    def stop_monitoring(self) -> None:
        """
        Stop the monitoring process and clean up resources.
        
        This method safely stops video capture, releases camera resources,
        and updates the GUI to reflect the stopped state.
        """
        # Stop processing
        self.is_running = False
        
        # Release camera resources
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Update GUI state
        self._update_gui_for_monitoring_stop()
        
        # Clear video display
        cv2.destroyAllWindows()
    
    def _update_gui_for_monitoring_stop(self) -> None:
        """Update GUI elements when monitoring stops."""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        self.hr_label.config(text="Heart Rate: -- BPM")
        self.rr_label.config(text="Respiration: -- BPM")
        
        # Clear video display
        if self.video_label:
            self.video_label.config(image='', text="No Video")
    
    def reset_signals(self) -> None:
        """
        Reset all signal data and plots to initial state.
        
        This method clears all signal buffers, resets vital sign estimates,
        and updates the display to show the reset state.
        """
        # Reset processor signals
        self.rppg_processor.reset_signals()
        self.resp_processor.reset_signals()
        
        # Clear plot data
        self.plot_manager.clear_all_data()
        
        # Update display
        self.hr_label.config(text="Heart Rate: -- BPM")
        self.rr_label.config(text="Respiration: -- BPM")
        self.status_label.config(text="Status: Signals Reset")
        
        print("All signals have been reset")
    
    def _capture_frames(self) -> None:
        """
        Capture video frames in a separate thread.
        
        This method runs continuously while monitoring is active,
        capturing frames from the camera and placing them in a queue
        for processing by the main thread.
        """
        start_time = time.time()
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            current_time = time.time() - start_time
            
            try:
                # Try to add frame to queue (non-blocking)
                self.frame_queue.put((frame, current_time), timeout=0.01)
            except queue.Full:
                # Skip frame if queue is full to maintain real-time performance
                pass
            
            # Control frame rate
            time.sleep(1/30)
    
    def _process_frames(self) -> None:
        """
        Process captured frames for vital signs extraction.
        
        This method runs in the main thread and processes frames from the
        capture queue, extracting both heart rate and respiration signals
        and updating the display accordingly.
        """
        if not self.is_running:
            return
        
        try:
            # Get frame from queue (non-blocking)
            frame, timestamp = self.frame_queue.get_nowait()
            
            # Process frame for both rPPG and respiration
            rppg_frame, face_detected, hr = self.rppg_processor.process_frame(frame)
            resp_frame, pose_detected, rr = self.resp_processor.process_frame(rppg_frame)
            
            # Update vital signs display
            self._update_vital_signs_display(face_detected, pose_detected, hr, rr)
            
            # Update status display
            self._update_status_display(face_detected, pose_detected)
            
            # Update plot data
            self._update_plot_data(timestamp)
            
            # Display processed video frame
            self._display_frame(resp_frame)
            
        except queue.Empty:
            # No frame available, continue
            pass
        except Exception as e:
            print(f"Frame processing error: {e}")
        
        # Schedule next frame processing
        if self.is_running:
            self.root.after(33, self._process_frames)  # ~30 FPS
    
    def _update_vital_signs_display(self, face_detected: bool, pose_detected: bool, 
                                  hr: float, rr: float) -> None:
        """
        Update vital signs display labels based on detection status and values.
        
        Args:
            face_detected: Whether face was detected for heart rate
            pose_detected: Whether pose was detected for respiration
            hr: Current heart rate estimate
            rr: Current respiration rate estimate
        """
        current_time = time.time()
        hr_staleness = current_time - self.rppg_processor.last_valid_hr_time
        
        # Update heart rate display
        if face_detected and hr > 0:
            if hr_staleness < 3.0:
                self.hr_label.config(
                    text=f"Heart Rate: {hr:.1f} BPM", 
                    foreground="red"
                )
            else:
                self.hr_label.config(
                    text=f"Heart Rate: {hr:.1f} BPM (updating...)", 
                    foreground="#AA0000"
                )
        elif face_detected:
            self.hr_label.config(
                text="Heart Rate: Calculating...", 
                foreground="red"
            )
        else:
            self.hr_label.config(
                text="Heart Rate: No Face", 
                foreground="gray"
            )
        
        # Update respiration display
        if pose_detected and rr > 0:
            self.rr_label.config(
                text=f"Respiration: {rr:.1f} BPM", 
                foreground="blue"
            )
        elif pose_detected:
            self.rr_label.config(
                text="Respiration: Calculating...", 
                foreground="blue"
            )
        else:
            self.rr_label.config(
                text="Respiration: No Pose", 
                foreground="gray"
            )
    
    def _update_status_display(self, face_detected: bool, pose_detected: bool) -> None:
        """
        Update status display based on detection results.
        
        Args:
            face_detected: Whether face was detected
            pose_detected: Whether pose was detected
        """
        status_parts = []
        if face_detected:
            status_parts.append("Face OK")
        if pose_detected:
            status_parts.append("Pose OK")
        
        if status_parts:
            self.status_label.config(text=f"Status: {' | '.join(status_parts)}")
        else:
            self.status_label.config(text="Status: No Detection")
    
    def _update_plot_data(self, timestamp: float) -> None:
        """
        Update plot data with current signal values.
        
        Args:
            timestamp: Current timestamp for the data point
        """
        # Update rPPG plot data
        if len(self.rppg_processor.r_signal) > 0:
            r_val = self.rppg_processor.r_signal[-1]
            g_val = self.rppg_processor.g_signal[-1]
            b_val = self.rppg_processor.b_signal[-1]
            
            rppg_val = (self.rppg_processor.rppg_signal[-1] 
                       if len(self.rppg_processor.rppg_signal) > 0 else 0)
            
            filtered_val = (self.rppg_processor.filtered_rppg[-1] 
                           if len(self.rppg_processor.filtered_rppg) > 0 else 0)
            
            self.plot_manager.update_rppg_data(
                timestamp, r_val, g_val, b_val, rppg_val, filtered_val
            )
        
        # Update respiration plot data
        if len(self.resp_processor.raw_y_buffer) > 0:
            raw_val = self.resp_processor.raw_y_buffer[-1]
            
            filtered_val = (self.resp_processor.filtered_y_buffer[-1] 
                           if len(self.resp_processor.filtered_y_buffer) > 0 else 0)
            
            self.plot_manager.update_respiration_data(
                timestamp, raw_val, filtered_val
            )
    
    def _display_frame(self, frame: np.ndarray) -> None:
        """
        Display video frame in the GUI with proper scaling.
        
        Args:
            frame: Video frame to display
        """
        if self.video_label is None:
            return
        
        try:
            # Get current label dimensions
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            # Resize frame to fit label while preserving aspect ratio
            if label_width > 1 and label_height > 1:
                resized_frame = self._resize_frame_to_fit(
                    frame, label_width, label_height
                )
            else:
                # Use default size if label dimensions not yet available
                resized_frame = cv2.resize(frame, (640, 480))
            
            # Convert to PIL format for tkinter
            image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image=image)
            
            # Update label with new image
            self.video_label.configure(image=image, text="")
            self.video_label.image = image  # Keep reference to prevent garbage collection
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def _resize_frame_to_fit(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Resize frame to fit within given dimensions while preserving aspect ratio.
        
        Args:
            frame: Input frame to resize
            width: Target width
            height: Target height
            
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        # Calculate new dimensions to fit within target size
        if width / height > aspect_ratio:
            # Target is wider than needed
            new_height = height
            new_width = int(new_height * aspect_ratio)
        else:
            # Target is taller than needed
            new_width = width
            new_height = int(new_width / aspect_ratio)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def on_closing(self) -> None:
        """
        Handle application closing event.
        
        This method ensures proper cleanup of resources when the
        application window is closed.
        """
        # Stop monitoring if running
        self.stop_monitoring()
        
        # Close any MediaPipe resources
        if hasattr(self.resp_processor, 'landmarker') and self.resp_processor.landmarker:
            self.resp_processor.landmarker.close()
        
        # Destroy OpenCV windows
        cv2.destroyAllWindows()
        
        # Close tkinter application
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> None:
        """
        Start the unified vital signs monitoring application.
        
        This method starts the main GUI event loop and runs the application
        until the user closes the window.
        """
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
            self.on_closing()
