import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torchvision.transforms as T
import cv2
import numpy as np
import torch
import os
import time
import psutil
from datetime import datetime
import concurrent.futures  # For background inference

# Custom face-detection function that uses OpenCV
from src.opencv_detector import detect_face_opencv

# Matplotlib imports (for plotting charts in Tkinter)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import the model
from model.masking import MaskedResNet

# Try to import a custom auth manager for the login screen.
# If it's not found, use a simple fallback method.
try:
    from auth_manager import authenticate_user
except ImportError:
    def authenticate_user(username, password):
        return (username == "admin" and password == "admin")


###############################################################################
# Preprocessing: from face ROI to tensor (1, 3, 48, 48)
###############################################################################
def preprocess_image(face_roi):
    """
    Preprocess the detected face ROI for our ResNet model:
      1) Convert to grayscale.
      2) Resize to 48x48.
      3) Normalize pixel values to [0,1].
      4) Replicate grayscale to 3 channels.
      5) Convert to a torch.Tensor and apply ImageNet normalization.
      6) Add a batch dimension → final shape: (1, 3, 48, 48).
    """
    # Convert from BGR to grayscale.
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48 (FER standard resolution).
    face_resized = cv2.resize(face_gray, (48, 48))
    # Normalize pixel values.
    face_normalized = face_resized / 255.0
    # Expand grayscale image to 3 channels.
    face_normalized = np.repeat(face_normalized[:, :, np.newaxis], 3, axis=2)
    # Convert NumPy array to a PyTorch tensor.
    face_tensor = T.ToTensor()(face_normalized).type(torch.float32)
    # Normalize using ImageNet statistics.
    face_tensor = T.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])(face_tensor)
    # Add batch dimension → (1, 3, 48, 48)
    face_tensor = face_tensor.unsqueeze(0)
    return face_tensor


###############################################################################
# Full-Screen Login Window
###############################################################################
def show_login_window():
    """
    Displays a full-screen login window that requires username and password.
    Closes upon successful authentication or when ESC is pressed.
    """
    login_window = tk.Toplevel()
    login_window.title("Login")
    login_window.attributes("-fullscreen", True)

    title_label = tk.Label(login_window, text="Login", font=("Helvetica", 24))
    title_label.pack(pady=50)

    # Username entry.
    tk.Label(login_window, text="Username:", font=("Helvetica", 16)).pack(pady=5)
    username_entry = tk.Entry(login_window, font=("Helvetica", 16))
    username_entry.pack()

    # Password entry.
    tk.Label(login_window, text="Password:", font=("Helvetica", 16)).pack(pady=5)
    password_entry = tk.Entry(login_window, show="*", font=("Helvetica", 16))
    password_entry.pack()

    def attempt_login():
        username = username_entry.get()
        password = password_entry.get()
        if authenticate_user(username, password):
            messagebox.showinfo("Login", "Login successful!")
            login_window.destroy()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    tk.Button(login_window, text="Login", font=("Helvetica", 16),
              command=attempt_login).pack(pady=30)

    def exit_fullscreen(event=None):
        login_window.attributes("-fullscreen", False)

    login_window.bind("<Escape>", exit_fullscreen)
    login_window.grab_set()
    login_window.focus_set()


###############################################################################
# Main GUI Application Class
###############################################################################
class BorderControlFERGUI:
    """
    Main GUI class for the Facial Expression Recognition (FER) system.
    Handles model loading, real-time webcam/video analysis, face detection and emotion
    classification, chart updates, and system resource monitoring.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Recognition System")
        self.root.geometry("1000x700")
        self.bg_color = "#1e1e2f"
        self.accent_color = "#9B1313"
        self.text_color = "#ffffff"
        self.button_text = "#ffffff"
        self.button_shadow = "#8e0000"
        self.root.configure(bg=self.bg_color)

        # Configure ttk style.
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Custom.TFrame", background=self.bg_color)
        self.style.configure("Custom.TLabel", background=self.bg_color, foreground=self.text_color,
                             font=("Helvetica", 12))
        self.style.configure("Header.TLabel", background=self.bg_color, foreground=self.text_color,
                             font=("Helvetica", 14, "bold"))
        self.style.configure("Accent.TButton", background=self.accent_color, foreground=self.button_text,
                             borderwidth=2, focusthickness=3, focuscolor="none", padding=10, relief="raised", anchor="center")
        self.style.map("Accent.TButton",
                       relief=[('pressed', 'sunken'), ('active', 'ridge')],
                       background=[('active', self.button_shadow), ('!active', self.accent_color)])

        # Load the model.
        self.load_model()

        # Define emotion classes.
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        self.high_risk_emotions = {"Angry", "Fear"}
        self.danger_threshold = 0.80        # Danger threshold set to 0.8 based on experimental results.

        # Flags/variables for webcam/video processing.
        self.is_webcam_running = False
        self.is_paused = False
        self.video_cap = None
        self.frame_count = 0
        self.last_prediction_time = time.time()

        # Create folder for saving snapshots.
        os.makedirs("snapshots", exist_ok=True)

        # Canvas dimensions.
        self.canvas_width = 600
        self.canvas_height = 450

        # Session emotion counts.
        self.emotion_counts = {label: 0 for label in self.emotion_labels}

        # Build the GUI layout.
        self.create_widgets()

        # Thread pool for asynchronous inference.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Show the login window.
        show_login_window()

    ############################################################################
    # Model Loading
    ############################################################################
    def load_model(self):
        default_model_path = "/Users/zeynep/PycharmProjects/FER/model/best_model.pth"
        if not os.path.exists(default_model_path):
            print("No model found at:", default_model_path)
            self.model = None
            return
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = MaskedResNet(arch="resnet18", pretrained=False,
                                      num_classes=7, dropout_p=0.3)
            state_dict = torch.load(default_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            print(f"[INFO] PyTorch model loaded from {default_model_path}")
        except Exception as e:
            print(f"[ERROR] Error loading PyTorch model: {e}")
            self.model = None

    ############################################################################
    # GUI Layout Creation
    ############################################################################
    def create_widgets(self):
        # Top information frame.
        info_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=20)
        info_frame.pack(fill=tk.X)
        header_label = ttk.Label(info_frame,
                                 text="Facial Expression Recognition System\nBorder Security",
                                 style="Header.TLabel", wraplength=980, justify="left")
        header_label.pack()

        # Button frame for webcam/video control.
        button_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=10)
        button_frame.pack(fill=tk.X, pady=5)
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)
        self.webcam_btn = ttk.Button(button_frame, text="Start Webcam", style="Accent.TButton",
                                     command=self.toggle_webcam)
        self.webcam_btn.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.pause_btn = ttk.Button(button_frame, text="Pause/Resume", style="Accent.TButton",
                                    command=self.pause_resume)
        self.pause_btn.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        upload_video_btn = ttk.Button(button_frame, text="Upload Video", style="Accent.TButton",
                                      command=self.upload_video)
        upload_video_btn.grid(row=0, column=2, padx=10, pady=5, sticky="nsew")

        # Secondary controls (camera index, danger threshold, CPU usage).
        controls_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=5)
        controls_frame.pack(fill=tk.X)
        ttk.Label(controls_frame, text="Camera Index:", style="Custom.TLabel").pack(side=tk.LEFT, padx=(10,5))
        self.camera_index_var = tk.IntVar(value=0)
        self.camera_combo = ttk.Combobox(controls_frame, textvariable=self.camera_index_var,
                                          values=[0, 1, 2, 3], width=5)
        self.camera_combo.pack(side=tk.LEFT, padx=(0,15))
        ttk.Label(controls_frame, text="Danger Threshold:", style="Custom.TLabel").pack(side=tk.LEFT, padx=(10,5))
        self.threshold_var = tk.DoubleVar(value=self.danger_threshold)
        threshold_entry = ttk.Entry(controls_frame, textvariable=self.threshold_var, width=5)
        threshold_entry.pack(side=tk.LEFT, padx=(0,15))
        self.resource_label = ttk.Label(controls_frame, text="CPU Usage: 0%", style="Custom.TLabel")
        self.resource_label.pack(side=tk.RIGHT, padx=10)

        # Main area: canvas for video and embedded charts.
        main_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main_frame, bg='#2e2e3e', width=600, height=450, highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.fig = Figure(figsize=(3,2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim([0,1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.bar_canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.bar_canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.fig2 = Figure(figsize=(3,2), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Session Emotions (Counts)")
        self.spectrum_canvas = FigureCanvasTkAgg(self.fig2, master=main_frame)
        self.spectrum_canvas.get_tk_widget().grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        self.results_label = ttk.Label(self.root, text="Predicted Emotion: [None]",
                                       style="Header.TLabel", padding=15)
        self.results_label.pack()

        self.update_resource_usage()

    def on_canvas_configure(self, event):
        """
        Updates internal canvas dimensions upon resize.
        """
        self.canvas_width = event.width
        self.canvas_height = event.height

    ############################################################################
    # Helper: Clear Charts
    ############################################################################
    def clear_charts(self):
        """
        Clears the bar charts (if needed). In this version, we retain charts after video playback.
        """
        self.ax.clear()
        self.ax.set_ylim([0,1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.bar_canvas.draw()

        self.ax2.clear()
        self.ax2.set_title("Session Emotions (Counts)")
        self.spectrum_canvas.draw()
        # Optionally, reset the session counts:
        # self.emotion_counts = {label: 0 for label in self.emotion_labels}

    ############################################################################
    # Webcam / Video Controls
    ############################################################################
    def toggle_webcam(self):
        """
        Toggles the webcam feed on or off.
        """
        if self.is_webcam_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        """
        Opens the webcam stream and starts capturing frames.
        """
        cam_index = self.camera_index_var.get()
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            self.results_label.config(text="Error: Unable to access the camera.")
            return

        self.is_webcam_running = True
        self.is_paused = False
        self.webcam_btn.config(text="Stop Webcam")
        self.results_label.config(text="Webcam active...")
        self.capture_frames()

    def stop_webcam(self):
        """
        Stops the webcam stream, releases the capture, and resets the UI.
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.is_webcam_running = False
        self.webcam_btn.config(text="Start Webcam")
        self.results_label.config(text="Predicted Emotion: [None]")
        self.canvas.delete("all")

    def pause_resume(self):
        """
        Pauses or resumes the current video/webcam feed.
        """
        if not self.is_webcam_running and not (self.video_cap and self.video_cap.isOpened()):
            self.results_label.config(text="No active feed to pause/resume.")
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.results_label.config(text="Detection Paused.")
        else:
            self.results_label.config(text="Detection Resumed.")
            if self.is_webcam_running:
                self.capture_frames()
            elif self.video_cap and self.video_cap.isOpened():
                self.play_video_frames()

    def capture_frames(self):
        """
        Captures frames from the webcam and processes them. Inference is performed every 10 frames,
        and inference is offloaded to a background thread.
        """
        if not self.is_webcam_running or self.is_paused:
            return
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            frame_interval = 10  # Process one out of every 10 frames.
            if self.frame_count % frame_interval == 0:
                # Use a background thread for model inference.
                self.executor.submit(self.inference_and_update, frame.copy())
            else:
                self.display_image(frame)
        self.root.after(10, self.capture_frames)

    ############################################################################
    # Offloaded Inference and UI Update
    ############################################################################
    def inference_and_update(self, frame):
        """
        Runs face detection and emotion classification in a background thread.
        Once complete, schedules the UI update on the main thread.
        """
        try:
            detection_result = detect_face_opencv(frame)
            if detection_result is None:
                self.root.after(0, lambda: self.results_label.config(text="No face detected."))
                self.root.after(0, lambda: self.display_image(frame))
                return

            face_roi, x, y, w, h, image_bgr = detection_result
            face_tensor = preprocess_image(face_roi).to(self.device)
            self.model.eval()
            with torch.no_grad():
                output = self.model(face_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probs)
                predicted_emotion = self.emotion_labels[predicted_class]
                confidence = probs[predicted_class]

            def update_ui():
                self.results_label.config(
                    text=f"Predicted Emotion: {predicted_emotion} ({confidence*100:.1f}%)"
                )
                self.update_bar_chart(probs)
                self.update_emotion_counts_chart(predicted_emotion)
                color = (0, 255, 0)
                if predicted_emotion in self.high_risk_emotions and confidence >= self.danger_threshold:
                    color = (0, 0, 255)
                    self.trigger_alert(image_bgr, predicted_emotion, confidence)
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), color, 2)
                self.display_image(image_bgr)
            self.root.after(0, update_ui)
        except Exception as e:
            self.error=e
            self.root.after(0, lambda: self.results_label.config(text=f"Error: {str(self.error)}"))

    ############################################################################
    # Video Playback Methods
    ############################################################################
    def upload_video(self):
        """
        Opens a file dialog to select a video file and starts playback.
        """
        self.clear_charts()
        self.canvas.delete("all")
        self.results_label.config(text="Predicted Emotion: [None]")
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        file_path = filedialog.askopenfilename(title="Select a Video", filetypes=filetypes)
        if file_path:
            self.start_video_playback(file_path)
        else:
            print("No video file selected.")

    def start_video_playback(self, file_path):
        """
        Starts video capture from the selected file.
        """
        if self.is_webcam_running:
            self.stop_webcam()
        if hasattr(self, 'video_cap') and self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(file_path)
        if not self.video_cap.isOpened():
            self.results_label.config(text="Error: Unable to open video file.")
            return
        self.frame_count = 0
        self.last_prediction_time = time.time()
        self.is_paused = False
        self.results_label.config(text="Playing Video...")
        self.canvas.delete("all")
        self.play_video_frames()

    def play_video_frames(self):
        """
        Reads and processes frames from the video file. Inference is performed every 10 frames.
        At the end of the video, only the canvas is cleared (charts are retained).
        """
        if not self.video_cap or not self.video_cap.isOpened():
            return
        if self.is_paused:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            # Video finished: clear only the canvas.
            self.video_cap.release()
            self.results_label.config(text="Video playback finished.")
            self.canvas.delete("all")
            return

        self.frame_count += 1
        frame_interval = 10  # Inference every 10 frames.
        if self.frame_count % frame_interval == 0:
            self.executor.submit(self.inference_and_update, frame.copy())
        else:
            self.display_image(frame)

        self.root.after(15, self.play_video_frames)

    ############################################################################
    # Display and Chart Update Methods
    ############################################################################
    def display_image(self, image_bgr):
        """
        Resizes and converts a cv2 BGR image into a format suitable for display on the Tkinter canvas.
        """
        frame_resized = cv2.resize(image_bgr, (self.canvas_width, self.canvas_height),
                                   interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(frame_resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        self.canvas.image = tk_img

    def update_bar_chart(self, predictions):
        """
        Updates the confidence bar chart with current model output probabilities.
        """
        self.ax.clear()
        self.ax.bar(self.emotion_labels, predictions)
        self.ax.set_ylim([0, 1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.ax.tick_params(axis='x', labelsize=8)
        self.ax.tick_params(axis='y', labelsize=8)
        self.bar_canvas.draw()

    def update_emotion_counts_chart(self, predicted_emotion):
        """
        Updates the session emotion counts chart by incrementing the count for the predicted emotion.
        """
        self.emotion_counts[predicted_emotion] += 1
        self.ax2.clear()
        emotions = list(self.emotion_counts.keys())
        counts = list(self.emotion_counts.values())
        self.ax2.bar(emotions, counts, color="orange")
        self.ax2.set_title("Session Emotions (Counts)")
        self.ax2.tick_params(axis='x', labelsize=8)
        self.ax2.tick_params(axis='y', labelsize=8)
        self.spectrum_canvas.draw()

    def trigger_alert(self, frame_bgr, emotion, confidence):
        """
        Triggers an alert (system bell and snapshot) if a high-risk emotion is detected above the danger threshold.
        """
        self.root.bell()
        alert_msg = f"ALERT! High-risk emotion detected: {emotion} ({confidence*100:.1f}%)"
        print(alert_msg)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"snapshots/alert_{emotion}_{timestamp_str}.jpg"
        cv2.imwrite(snapshot_filename, frame_bgr)
        print(f"Snapshot saved: {snapshot_filename}")

    def update_resource_usage(self):
        """
        Updates the CPU usage display periodically (every 2 seconds) using psutil.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        self.resource_label.config(text=f"CPU Usage: {cpu_percent:.0f}%")
        self.root.after(2000, self.update_resource_usage)

###############################################################################
# Main Entry Point
###############################################################################
def launch_gui():
    """
    Creates the main Tkinter window, initializes the BorderControlFERGUI application,
    and starts the Tk event loop.
    """
    root = tk.Tk()
    app = BorderControlFERGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()