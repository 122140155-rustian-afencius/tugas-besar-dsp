"""
Heart Rate & Respiration Realtime Monitoring

This application provides real-time monitoring of vital signs using computer vision:
- Heart rate detection using remote photoplethysmography (rPPG)
- Respiration rate detection using pose landmark analysis

The application features a clean, modular architecture with separate components
for signal processing, GUI management, and real-time visualization.
"""

import logging
import sys
from tkinter import messagebox

# Import modular components
try:
    from gui.main_app import UnifiedVitalSignsApp
except ImportError as e:
    print(f"Error importing GUI components: {e}")
    print("Please ensure all required packages are installed and modules are in the correct location.")
    sys.exit(1)


def setup_logging():
    """
    Configure logging for application monitoring.
    
    Sets up both console and file logging with appropriate formatting
    to track application events and potential issues.
    """
    logging.basicConfig(
        level=logging.INFO, 
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vital_signs_monitor.log')
        ]
    )
    
    # Log application startup
    logging.info("Vital Signs Monitor application starting...")


def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    required_modules = [
        'cv2', 'numpy', 'scipy', 'matplotlib', 'tkinter', 
        'mediapipe', 'PIL', 'pywt'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        error_msg = f"Missing required modules: {', '.join(missing_modules)}"
        logging.error(error_msg)
        messagebox.showerror(
            "Missing Dependencies", 
            f"{error_msg}\n\nPlease install the required packages using:\npip install -r requirements.txt"
        )
        return False
    
    return True


def main():
    """
    Main entry point for the Vital Signs Monitor application.
    
    This function:
    1. Sets up logging
    2. Checks for required dependencies
    3. Creates and runs the main application
    4. Handles any startup errors gracefully
    """
    try:
        # Setup application logging
        setup_logging()
        
        # Check dependencies
        if not check_dependencies():
            logging.error("Dependency check failed. Exiting application.")
            return
        
        logging.info("All dependencies verified successfully")
        
        # Create and run the main application
        logging.info("Initializing main application...")
        app = UnifiedVitalSignsApp()
        
        logging.info("Starting application main loop...")
        app.run()
        
    except Exception as e:
        error_msg = f"Fatal error during application startup: {e}"
        logging.error(error_msg, exc_info=True)
        
        # Show user-friendly error message
        try:
            messagebox.showerror(
                "Application Error", 
                f"An error occurred while starting the application:\n\n{e}\n\nCheck the log file for more details."
            )
        except:
            # If even tkinter fails, print to console
            print(f"FATAL ERROR: {error_msg}")
        
        sys.exit(1)
    
    finally:
        logging.info("Vital Signs Monitor application shutdown complete")


if __name__ == "__main__":
    """
    Application entry point.
    
    This ensures the application only runs when this file is executed directly,
    not when imported as a module.
    """
    main()