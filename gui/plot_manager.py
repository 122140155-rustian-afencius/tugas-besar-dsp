"""
Plot manager for real-time visualization of vital signs data.

This module handles all matplotlib plotting functionality for displaying
heart rate and respiration signals in real-time with proper scaling and
performance optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from collections import deque
import tkinter as tk
from typing import Tuple, List, Optional


class PlotManager:
    """
    Manages real-time plotting for rPPG and respiration signals.
    
    This class handles the creation, updating, and animation of matplotlib
    plots for visualizing heart rate and respiration monitoring data.
    """
    
    def __init__(self, rppg_parent: tk.Widget, resp_parent: tk.Widget):
        """
        Initialize the plot manager with parent widgets for embedding plots.
        
        Args:
            rppg_parent: Parent widget for rPPG plots
            resp_parent: Parent widget for respiration plots
        """
        self.rppg_parent = rppg_parent
        self.resp_parent = resp_parent
        
        # Initialize plot data buffers
        self._initialize_data_buffers()
        
        # Setup plots
        self._setup_rppg_plots()
        self._setup_respiration_plots()
        
        # Start animations
        self._start_animations()
    
    def _initialize_data_buffers(self) -> None:
        """Initialize data storage buffers for all signals."""
        # Buffer size for approximately 10 seconds of data at 30 FPS
        buffer_size = 300
        
        # rPPG signal buffers
        self.rppg_time_data = deque(maxlen=buffer_size)
        self.r_data = deque(maxlen=buffer_size)
        self.g_data = deque(maxlen=buffer_size)
        self.b_data = deque(maxlen=buffer_size)
        self.rppg_data = deque(maxlen=buffer_size)
        self.filtered_rppg_data = deque(maxlen=buffer_size)
        
        # Respiration signal buffers
        self.resp_time_data = deque(maxlen=buffer_size)
        self.raw_resp_data = deque(maxlen=buffer_size)
        self.filtered_resp_data = deque(maxlen=buffer_size)
    
    def _setup_rppg_plots(self) -> None:
        """
        Setup matplotlib plots for rPPG heart rate monitoring.
        
        Creates three subplots:
        1. RGB channel signals
        2. Raw rPPG signal
        3. Filtered rPPG signal
        """
        # Create figure with subplots
        self.rppg_fig, self.rppg_axes = plt.subplots(3, 1, figsize=(8, 10))
        self.rppg_fig.tight_layout(pad=3.0)
        
        # Configure RGB signals subplot
        self._setup_rgb_subplot()
        
        # Configure raw rPPG subplot
        self._setup_raw_rppg_subplot()
        
        # Configure filtered rPPG subplot
        self._setup_filtered_rppg_subplot()
        
        # Embed plot in GUI
        self.rppg_canvas = FigureCanvasTkAgg(self.rppg_fig, self.rppg_parent)
        self.rppg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _setup_rgb_subplot(self) -> None:
        """Configure the RGB signals subplot."""
        ax = self.rppg_axes[0]
        ax.set_title("RGB Channel Signals from Face ROI", fontsize=12, fontweight='bold')
        ax.set_ylabel("Pixel Intensity", fontsize=10)
        
        # Create line objects for each RGB channel
        self.r_line, = ax.plot([], [], 'r-', label='Red Channel', alpha=0.8, linewidth=1.5)
        self.g_line, = ax.plot([], [], 'g-', label='Green Channel', alpha=0.8, linewidth=1.5)
        self.b_line, = ax.plot([], [], 'b-', label='Blue Channel', alpha=0.8, linewidth=1.5)
        
        # Configure subplot appearance
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f8f8')
    
    def _setup_raw_rppg_subplot(self) -> None:
        """Configure the raw rPPG signal subplot."""
        ax = self.rppg_axes[1]
        ax.set_title("Raw rPPG Signal (POS Method)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Amplitude", fontsize=10)
        
        # Create line object for raw rPPG signal
        self.rppg_line, = ax.plot([], [], 'k-', linewidth=1.5, alpha=0.7)
        
        # Configure subplot appearance
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f8f8')
    
    def _setup_filtered_rppg_subplot(self) -> None:
        """Configure the filtered rPPG signal subplot."""
        ax = self.rppg_axes[2]
        ax.set_title("Filtered rPPG Signal (Heart Rate)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        
        # Create line object for filtered rPPG signal
        self.filtered_rppg_line, = ax.plot([], [], 'purple', linewidth=2.5, alpha=0.9)
        
        # Configure subplot appearance
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f8f8')
    
    def _setup_respiration_plots(self) -> None:
        """
        Setup matplotlib plots for respiration monitoring.
        
        Creates two subplots:
        1. Raw shoulder position signal
        2. Filtered respiratory signal
        """
        # Create figure with subplots
        self.resp_fig, self.resp_axes = plt.subplots(2, 1, figsize=(8, 6))
        self.resp_fig.tight_layout(pad=3.0)
        
        # Configure raw respiration subplot
        self._setup_raw_respiration_subplot()
        
        # Configure filtered respiration subplot
        self._setup_filtered_respiration_subplot()
        
        # Embed plot in GUI
        self.resp_canvas = FigureCanvasTkAgg(self.resp_fig, self.resp_parent)
        self.resp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _setup_raw_respiration_subplot(self) -> None:
        """Configure the raw shoulder position subplot."""
        ax = self.resp_axes[0]
        ax.set_title("Raw Shoulder Y Position", fontsize=12, fontweight='bold')
        ax.set_ylabel("Y Position (pixels)", fontsize=10)
        
        # Create line object for raw signal
        self.raw_resp_line, = ax.plot([], [], 'gray', label='Raw Signal', 
                                     linewidth=1.5, alpha=0.7)
        
        # Configure subplot appearance
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_facecolor('#f8f8f8')
    
    def _setup_filtered_respiration_subplot(self) -> None:
        """Configure the filtered respiratory signal subplot."""
        ax = self.resp_axes[1]
        ax.set_title("Filtered Respiratory Signal", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        
        # Create line object for filtered signal
        self.filtered_resp_line, = ax.plot([], [], 'green', linewidth=2.5, 
                                          alpha=0.9, label='Respiration')
        
        # Configure subplot appearance
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_facecolor('#f8f8f8')
    
    def _start_animations(self) -> None:
        """Start matplotlib animations for real-time updating."""
        # Animation for rPPG plots (update every 100ms)
        self.rppg_ani = FuncAnimation(
            self.rppg_fig, 
            self._update_rppg_plots, 
            interval=100, 
            blit=False,
            cache_frame_data=False
        )
        
        # Animation for respiration plots (update every 100ms)
        self.resp_ani = FuncAnimation(
            self.resp_fig, 
            self._update_resp_plots, 
            interval=100, 
            blit=False,
            cache_frame_data=False
        )
    
    def update_rppg_data(self, timestamp: float, r_val: float, g_val: float, 
                        b_val: float, rppg_val: float, filtered_val: float) -> None:
        """
        Update rPPG plot data with new signal values.
        
        Args:
            timestamp: Current timestamp in seconds
            r_val: Red channel value
            g_val: Green channel value
            b_val: Blue channel value
            rppg_val: Raw rPPG signal value
            filtered_val: Filtered rPPG signal value
        """
        self.rppg_time_data.append(timestamp)
        self.r_data.append(r_val)
        self.g_data.append(g_val)
        self.b_data.append(b_val)
        self.rppg_data.append(rppg_val)
        self.filtered_rppg_data.append(filtered_val)
    
    def update_respiration_data(self, timestamp: float, raw_val: float, 
                              filtered_val: float) -> None:
        """
        Update respiration plot data with new signal values.
        
        Args:
            timestamp: Current timestamp in seconds
            raw_val: Raw shoulder position value
            filtered_val: Filtered respiratory signal value
        """
        self.resp_time_data.append(timestamp)
        self.raw_resp_data.append(raw_val)
        self.filtered_resp_data.append(filtered_val)
    
    def _update_rppg_plots(self, frame) -> List:
        """
        Update rPPG plots with current data.
        
        Args:
            frame: Animation frame number (unused but required by FuncAnimation)
            
        Returns:
            List of updated line objects
        """
        if len(self.rppg_time_data) < 2:
            return []
        
        # Convert data to numpy arrays for efficient plotting
        time_array = np.array(self.rppg_time_data)
        
        # Update line data
        self.r_line.set_data(time_array, np.array(self.r_data))
        self.g_line.set_data(time_array, np.array(self.g_data))
        self.b_line.set_data(time_array, np.array(self.b_data))
        self.rppg_line.set_data(time_array, np.array(self.rppg_data))
        self.filtered_rppg_line.set_data(time_array, np.array(self.filtered_rppg_data))
        
        # Auto-scale and update axis limits
        self._update_rppg_axis_limits(time_array)
        
        return [self.r_line, self.g_line, self.b_line, 
                self.rppg_line, self.filtered_rppg_line]
    
    def _update_rppg_axis_limits(self, time_array: np.ndarray) -> None:
        """
        Update axis limits for rPPG plots with smart scaling.
        
        Args:
            time_array: Array of time values
        """
        if len(time_array) == 0:
            return
        
        # Show last 10 seconds of data
        time_window = 10.0
        t_max = time_array[-1]
        t_min = max(0, t_max - time_window)
        
        for i, ax in enumerate(self.rppg_axes):
            ax.set_xlim(t_min, t_max)
            
            # Auto-scale Y axis with some padding
            try:
                # Get visible data within time window
                mask = (time_array >= t_min) & (time_array <= t_max)
                
                if i == 0:  # RGB signals
                    r_visible = np.array(self.r_data)[mask[-len(self.r_data):]]
                    g_visible = np.array(self.g_data)[mask[-len(self.g_data):]]
                    b_visible = np.array(self.b_data)[mask[-len(self.b_data):]]
                    
                    if len(r_visible) > 0:
                        y_min = min(np.min(r_visible), np.min(g_visible), np.min(b_visible))
                        y_max = max(np.max(r_visible), np.max(g_visible), np.max(b_visible))
                        y_padding = 0.1 * (y_max - y_min) if y_max > y_min else 1
                        ax.set_ylim(y_min - y_padding, y_max + y_padding)
                
                elif i == 1:  # Raw rPPG
                    rppg_visible = np.array(self.rppg_data)[mask[-len(self.rppg_data):]]
                    if len(rppg_visible) > 0 and np.std(rppg_visible) > 0:
                        y_mean = np.mean(rppg_visible)
                        y_std = np.std(rppg_visible)
                        ax.set_ylim(y_mean - 3*y_std, y_mean + 3*y_std)
                
                elif i == 2:  # Filtered rPPG
                    filt_visible = np.array(self.filtered_rppg_data)[mask[-len(self.filtered_rppg_data):]]
                    if len(filt_visible) > 0 and np.std(filt_visible) > 0:
                        y_mean = np.mean(filt_visible)
                        y_std = np.std(filt_visible)
                        ax.set_ylim(y_mean - 3*y_std, y_mean + 3*y_std)
                        
            except (IndexError, ValueError):
                # Fallback to auto-scaling
                ax.relim()
                ax.autoscale_view()
    
    def _update_resp_plots(self, frame) -> List:
        """
        Update respiration plots with current data.
        
        Args:
            frame: Animation frame number (unused but required by FuncAnimation)
            
        Returns:
            List of updated line objects
        """
        if len(self.resp_time_data) < 2:
            return []
        
        # Convert data to numpy arrays
        time_array = np.array(self.resp_time_data)
        
        # Update line data
        self.raw_resp_line.set_data(time_array, np.array(self.raw_resp_data))
        self.filtered_resp_line.set_data(time_array, np.array(self.filtered_resp_data))
        
        # Update axis limits
        self._update_resp_axis_limits(time_array)
        
        return [self.raw_resp_line, self.filtered_resp_line]
    
    def _update_resp_axis_limits(self, time_array: np.ndarray) -> None:
        """
        Update axis limits for respiration plots.
        
        Args:
            time_array: Array of time values
        """
        if len(time_array) == 0:
            return
        
        # Show last 20 seconds for respiration (longer window for breathing cycles)
        time_window = 20.0
        t_max = time_array[-1]
        t_min = max(0, t_max - time_window)
        
        for i, ax in enumerate(self.resp_axes):
            ax.set_xlim(t_min, t_max)
            
            # Auto-scale Y axis
            try:
                mask = (time_array >= t_min) & (time_array <= t_max)
                
                if i == 0:  # Raw signal
                    raw_visible = np.array(self.raw_resp_data)[mask[-len(self.raw_resp_data):]]
                    if len(raw_visible) > 0:
                        y_min, y_max = np.min(raw_visible), np.max(raw_visible)
                        y_padding = 0.1 * (y_max - y_min) if y_max > y_min else 5
                        ax.set_ylim(y_min - y_padding, y_max + y_padding)
                
                elif i == 1:  # Filtered signal
                    filt_visible = np.array(self.filtered_resp_data)[mask[-len(self.filtered_resp_data):]]
                    if len(filt_visible) > 0 and np.std(filt_visible) > 0:
                        y_mean = np.mean(filt_visible)
                        y_std = np.std(filt_visible)
                        ax.set_ylim(y_mean - 3*y_std, y_mean + 3*y_std)
                        
            except (IndexError, ValueError):
                ax.relim()
                ax.autoscale_view()
    
    def clear_all_data(self) -> None:
        """Clear all plot data buffers and reset displays."""
        # Clear rPPG data
        self.rppg_time_data.clear()
        self.r_data.clear()
        self.g_data.clear()
        self.b_data.clear()
        self.rppg_data.clear()
        self.filtered_rppg_data.clear()
        
        # Clear respiration data
        self.resp_time_data.clear()
        self.raw_resp_data.clear()
        self.filtered_resp_data.clear()
        
        # Force plot updates
        self._update_rppg_plots(None)
        self._update_resp_plots(None)
        self.rppg_canvas.draw_idle()
        self.resp_canvas.draw_idle()
    
    def save_plots(self, filename_prefix: str = "vital_signs") -> None:
        """
        Save current plots to image files.
        
        Args:
            filename_prefix: Prefix for saved filenames
        """
        try:
            # Save rPPG plots
            self.rppg_fig.savefig(f"{filename_prefix}_heart_rate.png", 
                                 dpi=300, bbox_inches='tight')
            
            # Save respiration plots
            self.resp_fig.savefig(f"{filename_prefix}_respiration.png", 
                                 dpi=300, bbox_inches='tight')
            
            print(f"Plots saved as {filename_prefix}_*.png")
            
        except Exception as e:
            print(f"Error saving plots: {e}")
