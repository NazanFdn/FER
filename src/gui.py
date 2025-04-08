
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

# Custom face-detection function that uses OpenCV
from src.opencv_detector import detect_face_opencv

# Matplotlib imports (for plotting charts in Tkinter)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import the model
from model.masking import MaskedResNet

# Try to import a custom auth manager for the login screen
# If it's not found, use a simple fallback method
try:
    from auth_manager import authenticate_user
except ImportError:
    def authenticate_user(username, password):
        return (username == "admin" and password == "admin")


###############################################################################
# Preprocessing: from face ROI
###############################################################################
def preprocess_image(face_roi):
    """
    Preprocess the detected face ROI for our ResNet model:
      1) Convert to grayscale (although the face might already be BGR).
      2) Resize to (48,48). The dataset contains
      3) Normalize pixel values to [0,1].
      4) Replicate to 3 channels (since ResNet expects 3-channel input).
      5) Convert to a torch.Tensor and apply the same mean/std normalization
         used in training.
      6) Add a batch dimension so shape => (1,3,48,48).
    """
    # Convert ROI from BGR to grayscale
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Resize the face to 48x48 (FER standard resolution)
    face_resized = cv2.resize(face_gray, (48, 48))
    # Normalize pixel values to [0,1]
    face_normalized = face_resized / 255.0
    # Expand grayscale image to 3 channels
    face_normalized = np.repeat(face_normalized[:, :, np.newaxis], 3, axis=2)

    # Convert NumPy array -> PyTorch tensor
    face_tensor = T.ToTensor()(face_normalized).type(torch.float32)
    # Apply the same mean/std normalization used by ResNet
    face_tensor = T.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])(face_tensor)
    # Add a batch dimension => shape (1,3,48,48)
    face_tensor = face_tensor.unsqueeze(0)
    return face_tensor


###############################################################################
# Full-Screen Login Window
###############################################################################
def show_login_window():
    """
    Displays a full-screen login window that requires the user to enter
    a username and password. Closes upon successful authentication or ESC key.
    """
    login_window = tk.Toplevel()
    login_window.title("Login")
    login_window.attributes("-fullscreen", True)

    title_label = tk.Label(login_window, text="Login", font=("Helvetica", 24))
    title_label.pack(pady=50)

    # Username
    tk.Label(login_window, text="Username:", font=("Helvetica", 16)).pack(pady=5)
    username_entry = tk.Entry(login_window, font=("Helvetica", 16))
    username_entry.pack()

    # Password
    tk.Label(login_window, text="Password:", font=("Helvetica", 16)).pack(pady=5)
    password_entry = tk.Entry(login_window, show="*", font=("Helvetica", 16))
    password_entry.pack()

    # Attempt login callback
    def attempt_login():
        username = username_entry.get()
        password = password_entry.get()
        if authenticate_user(username, password):
            messagebox.showinfo("Login", "Login successful!")
            login_window.destroy()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    # Login button
    tk.Button(login_window, text="Login", font=("Helvetica", 16),
              command=attempt_login).pack(pady=30)

    # Allow exiting from full-screen via the ESC key
    def exit_fullscreen(event=None):
        login_window.attributes("-fullscreen", False)

    login_window.bind("<Escape>", exit_fullscreen)
    # Ensure the login window is modal (captures focus)
    login_window.grab_set()
    login_window.focus_set()


###############################################################################
# Main GUI Application Class
###############################################################################
class BorderControlFERGUI:
    """
    Main class for the Facial Expression Recognition (FER) GUI application.
    Handles:
      - Model loading
      - Webcam / video analysis
      - Real-time emotion detection
      - Chart displays (confidence bar chart + session emotion counts)
      - System resource usage
    """
    def __init__(self, root):
        """
        Initialize the GUI, including layout, styling, model loading,
        and the initial login window.
        """
        self.root = root
        self.root.title("Facial Expression Recognition System")
        self.root.geometry("1000x700")

        # Define the color palette and styling for the UI
        self.bg_color = "#1e1e2f"
        self.accent_color = "#9B1313"
        self.text_color = "#ffffff"
        self.button_text = "#ffffff"
        self.button_shadow = "#8e0000"
        self.root.configure(bg=self.bg_color)

        # Use ttk Style to customize the look
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Custom.TFrame", background=self.bg_color)
        self.style.configure(
            "Custom.TLabel",
            background=self.bg_color,
            foreground=self.text_color,
            font=("Helvetica", 12)
        )
        self.style.configure(
            "Header.TLabel",
            background=self.bg_color,
            foreground=self.text_color,
            font=("Helvetica", 14, "bold")
        )
        self.style.configure(
            "Accent.TButton",
            background=self.accent_color,
            foreground=self.button_text,
            borderwidth=2,
            focusthickness=3,
            focuscolor="none",
            padding=10,
            relief="raised",
            anchor="center"
        )
        # Map the button style to reflect pressed and active states
        self.style.map(
            "Accent.TButton",
            relief=[('pressed', 'sunken'), ('active', 'ridge')],
            background=[('active', self.button_shadow), ('!active', self.accent_color)]
        )

        # Load the PyTorch model (MaskedResNet)
        self.load_model()

        # Define emotion labels (7 typical FER classes)
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        # Emotions considered "high risk"
        self.high_risk_emotions = {"Angry", "Fear"}
        # Confidence threshold for "alert" in high-risk emotions
        self.danger_threshold = 0.80

        # Flags & Variables controlling webcam and video states
        self.is_webcam_running = False
        self.is_paused = False
        self.video_cap = None
        self.frame_count = 0
        self.last_prediction_time = time.time()

        # Create a folder for storing screenshots if it doesn't exist
        os.makedirs("snapshots", exist_ok=True)

        # Canvas dimensions
        self.canvas_width = 600
        self.canvas_height = 450

        # Track count of recognized emotions during the session
        self.emotion_counts = {label: 0 for label in self.emotion_labels}

        # Build the overall GUI layout
        self.create_widgets()

        # Show the login window (covers the main app until login)
        show_login_window()

    ############################################################################
    # Load PyTorch Model
    ############################################################################
    def load_model(self):
        """
        Loads a trained PyTorch model (MaskedResNet) from a .pth file.
        If the file doesn't exist, sets self.model to None.
        """
        default_model_path = "/Users/zeynep/PycharmProjects/FER/model/best_model.pth"
        if not os.path.exists(default_model_path):
            print("No model found at:", default_model_path)
            self.model = None
            return

        try:
            # Decide whether to use CPU or GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Instantiate the MaskedResNet
            #   arch="resnet18", pretrained=False (since we have a fine-tuned .pth),
            #   num_classes=7, dropout_p=0.3
            self.model = MaskedResNet(arch="resnet18", pretrained=False,
                                      num_classes=7, dropout_p=0.3)
            # Load the saved weights
            state_dict = torch.load(default_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

            # Set the model to evaluation mode
            self.model.eval()
            self.model.to(self.device)

            print(f"[INFO] PyTorch model loaded from {default_model_path}")
        except Exception as e:
            print(f"[ERROR] Error loading PyTorch model: {e}")
            self.model = None

    ############################################################################
    # Create Widgets and Layout
    ############################################################################
    def create_widgets(self):
        """
        Builds the user interface components: top info frame, controls,
        canvas for video, charts, and status labels.
        """
        # ---------- Top Info Frame ----------
        info_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=20)
        info_frame.pack(fill=tk.X)
        header_label = ttk.Label(
            info_frame,
            text="Facial Expression Recognition System\nBorder Security",
            style="Header.TLabel",
            wraplength=980,
            justify="left"
        )
        header_label.pack()

        # ---------- Button Frame (top row of buttons) ----------
        button_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=10)
        button_frame.pack(fill=tk.X, pady=5)

        # We'll arrange these in 3 columns
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        # Start/Stop webcam button
        self.webcam_btn = ttk.Button(
            button_frame,
            text="Start Webcam",
            style="Accent.TButton",
            command=self.toggle_webcam
        )
        self.webcam_btn.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # Pause/Resume detection button
        self.pause_btn = ttk.Button(
            button_frame,
            text="Pause/Resume",
            style="Accent.TButton",
            command=self.pause_resume
        )
        self.pause_btn.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

        # Upload video button
        upload_video_btn = ttk.Button(
            button_frame,
            text="Upload Video",
            style="Accent.TButton",
            command=self.upload_video
        )
        upload_video_btn.grid(row=0, column=2, padx=10, pady=5, sticky="nsew")

        # ---------- Secondary Controls (camera index, threshold, etc.) ----------
        controls_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=5)
        controls_frame.pack(fill=tk.X)

        # Camera index selection
        ttk.Label(
            controls_frame,
            text="Camera Index:",
            style="Custom.TLabel"
        ).pack(side=tk.LEFT, padx=(10, 5))

        self.camera_index_var = tk.IntVar(value=0)
        self.camera_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.camera_index_var,
            values=[0, 1, 2, 3],
            width=5
        )
        self.camera_combo.pack(side=tk.LEFT, padx=(0, 15))

        # Danger threshold for alert
        ttk.Label(
            controls_frame,
            text="Danger Threshold:",
            style="Custom.TLabel"
        ).pack(side=tk.LEFT, padx=(10, 5))

        self.threshold_var = tk.DoubleVar(value=self.danger_threshold)
        threshold_entry = ttk.Entry(
            controls_frame,
            textvariable=self.threshold_var,
            width=5
        )
        threshold_entry.pack(side=tk.LEFT, padx=(0, 15))

        # CPU usage label
        self.resource_label = ttk.Label(
            controls_frame,
            text="CPU Usage: 0%",
            style="Custom.TLabel"
        )
        self.resource_label.pack(side=tk.RIGHT, padx=10)

        # ---------- Main area: Canvas + Graphs ----------
        main_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas to display the video frames
        self.canvas = tk.Canvas(
            main_frame,
            bg='#2e2e3e',
            width=600,
            height=450,
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # 1) Bar Chart (top-right) - for the current model's output confidences
        self.fig = Figure(figsize=(3, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim([0, 1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.bar_canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.bar_canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # 2) Session Emotions (bottom-right) - a bar chart counting how often each emotion appears
        self.fig2 = Figure(figsize=(3, 2), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Session Emotions (Counts)")
        self.spectrum_canvas = FigureCanvasTkAgg(self.fig2, master=main_frame)
        self.spectrum_canvas.get_tk_widget().grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Make the main frame responsive
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # A label to display the most recent prediction text
        self.results_label = ttk.Label(
            self.root,
            text="Predicted Emotion: [None]",
            style="Header.TLabel",
            padding=15
        )
        self.results_label.pack()

        # Start periodic updates of CPU usage
        self.update_resource_usage()

    def on_canvas_configure(self, event):
        """
        Called whenever the canvas widget is resized.
        Updates stored width/height so we can correctly scale frames.
        """
        self.canvas_width = event.width
        self.canvas_height = event.height

    # ---------------------------------------------------------------------------
    # HELPER: Clear bar charts
    # ---------------------------------------------------------------------------
    def clear_charts(self):
        """
        Clears both the confidence bar chart and the session counts chart.
        Also can reset the emotion_counts if desired.
        """
        # Clear the top bar chart
        self.ax.clear()
        self.ax.set_ylim([0, 1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.bar_canvas.draw()

        # Clear the session counts chart
        self.ax2.clear()
        self.ax2.set_title("Session Emotions (Counts)")
        self.spectrum_canvas.draw()

        # Uncomment if you want to reset the session counts:
        # self.emotion_counts = {label: 0 for label in self.emotion_labels}

    # ---------------------------------------------------------------------------
    # Webcam / Video Logic
    # ---------------------------------------------------------------------------
    def toggle_webcam(self):
        """
        Toggles the webcam on/off. If currently running, it stops; otherwise starts.
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
        Stops the webcam stream, releases the capture, and resets UI states.
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.is_webcam_running = False
        self.webcam_btn.config(text="Start Webcam")
        self.results_label.config(text="Predicted Emotion: [None]")
        self.canvas.delete("all")

    def pause_resume(self):
        """
        Pauses or resumes detection, for both the webcam or an uploaded video.
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
        Captures frames from the webcam,
        runs face detection and classification on each frame.
        """
        if not self.is_webcam_running or self.is_paused:
            return
        ret, frame = self.cap.read()
        if ret:
            self.detect_and_classify(frame)
        self.root.after(10, self.capture_frames)

    # ---------------------------------------------------------------------------
    # Detection / Classification
    # ---------------------------------------------------------------------------
    def detect_and_classify(self, frame):
        """
        Runs the face detection on a single frame (via detect_face_opencv),
        then processes the detected face ROI with the model to classify the emotion.
        Draws results on the canvas.
        """
        try:
            # Attempt to detect a face ROI in the current frame
            detection_result = detect_face_opencv(frame)
            if detection_result is None:
                self.display_image(frame)
                self.results_label.config(text="No face detected.")
                return

            # detection_result is (face_roi, x, y, w, h, image_bgr)
            face_roi, x, y, w, h, image_bgr = detection_result

            # If we have a loaded model, run inference
            if self.model is not None:
                face_tensor = preprocess_image(face_roi).to(self.device)

                # Evaluate model
                self.model.eval()
                with torch.no_grad():
                    output = self.model(face_tensor)  # shape (1,7)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predicted_class = np.argmax(probs)
                    predicted_emotion = self.emotion_labels[predicted_class]
                    confidence = probs[predicted_class]

                # Update textual result
                self.results_label.config(
                    text=f"Predicted Emotion: {predicted_emotion} ({confidence * 100:.1f}%)"
                )
                # Update the bar chart with confidence values
                self.update_bar_chart(probs)
                # Update the session counts chart
                self.update_emotion_counts_chart(predicted_emotion)

                # Draw bounding box around the face
                color = (0, 255, 0)
                # If it's a high-risk emotion over threshold, turn box red & alert
                if predicted_emotion in self.high_risk_emotions and confidence >= self.danger_threshold:
                    color = (0, 0, 255)
                    self.trigger_alert(image_bgr, predicted_emotion, confidence)

                # Draw the face rectangle on the displayed image
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                self.display_image(image_bgr)
            else:
                # If no model is loaded, just show the image
                self.results_label.config(text="Model not loaded.")
                self.display_image(image_bgr)
        except Exception as e:
            self.results_label.config(text=f"Error: {str(e)}")

    def display_image(self, image_bgr):
        """
        Utility to draw a cv2 BGR image on the main Tkinter canvas.
        Resizes to fit the canvas, then converts to PIL for Tkinter compatibility.
        """
        frame_resized = cv2.resize(image_bgr, (self.canvas_width, self.canvas_height),
                                   interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(frame_resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        self.canvas.image = tk_img

    # ---------------------------------------------------------------------------
    # Video Playback
    # ---------------------------------------------------------------------------
    def upload_video(self):
        """
        Opens a file dialog to select a local video file, then starts playback
        and performs face detection+classification on each frame.
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
        Initiate video capture from a selected file, release any previous capture,
        and start reading frames.
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
        if not self.video_cap or not self.video_cap.isOpened():
            return
        if self.is_paused:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            # The video is finished
            self.video_cap.release()
            self.results_label.config(text="Video playback finished.")

            # Optionally clear both the canvas and the charts
            self.canvas.delete("all")
            self.clear_charts()

            return

        self.frame_count += 1
        frame_interval = 5
        if self.frame_count % frame_interval == 0:
            self.detect_and_classify(frame)
        else:
            self.display_image(frame)

        self.root.after(15, self.play_video_frames)

    """
    # Image Upload

    def upload_image(self):
        self.clear_charts()
        self.canvas.delete("all")
        self.results_label.config(text="Predicted Emotion: [None]")

        filetypes = [("Image files", "*.jpg *.png *.jpeg")]
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=filetypes)
        if file_path:
            if self.is_webcam_running:
                self.stop_webcam()
            self.process_image(file_path)
        else:
            print("No image file selected.")

    def process_image(self, file_path):
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self.results_label.config(text="Error: Unable to load image.")
            return

        detection_result = detect_face_opencv(image_bgr)
        if detection_result is None:
            self.results_label.config(text="No face detected in the image.")
            self.display_image(image_bgr)
            return

        face_roi, x, y, w, h, image_bgr = detection_result

        if self.model is not None:
            # same steps as detect_and_classify
            preprocessed_face = preprocess_image(face_roi)  # shape (1,48,48,1)
            face_tensor = torch.from_numpy(preprocessed_face).float() # => (1,48,48,1)
            face_tensor = face_tensor.permute(0, 3, 1, 2)              # => (1,1,48,48)
            face_tensor = face_tensor.expand(-1, 3, -1, -1)            # => (1,3,48,48)
            face_tensor = face_tensor.to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(face_tensor)  # shape (1,7)

            probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # shape (7,)
            predicted_class = np.argmax(probs)
            predicted_emotion = self.emotion_labels[predicted_class]
            confidence = probs[predicted_class]

            self.results_label.config(
                text=f"Predicted Emotion: {predicted_emotion} ({confidence*100:.1f}%)"
            )
            self.update_bar_chart(probs)
            self.update_emotion_counts_chart(predicted_emotion)

            self.danger_threshold = self.threshold_var.get()
            color = (0, 255, 0)
            if (predicted_emotion in self.high_risk_emotions) and (confidence >= self.danger_threshold):
                color = (0, 0, 255)
                self.trigger_alert(image_bgr, predicted_emotion, confidence)

            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
            self.display_image(image_bgr)
        else:
            self.results_label.config(text="Model not loaded.")
            self.display_image(image_bgr)
    """

    # ---------------------------------------------------------------------------
    # Chart Updates
    # ---------------------------------------------------------------------------
    def update_bar_chart(self, predictions):
        """
        Updates the top-right bar chart with the current model output confidences.
        predictions: a list or array of length 7 with values in [0,1].
        """
        self.ax.clear()
        self.ax.bar(self.emotion_labels, predictions)
        self.ax.set_ylim([0, 1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.ax.tick_params(axis='x', labelsize=8)
        self.ax.tick_params(axis='y', labelsize=8)
        self.ax.set_ylabel("Confidence", fontsize=9)
        self.ax.set_title("Emotion Spectrum (Bar Chart)", fontsize=10)
        self.bar_canvas.draw()

    def update_emotion_counts_chart(self, predicted_emotion):
        """
        Increments the count for the newly predicted emotion,
        and refreshes the session-counts bar chart.
        """
        self.emotion_counts[predicted_emotion] += 1
        self.ax2.clear()
        emotions = list(self.emotion_counts.keys())
        counts = list(self.emotion_counts.values())
        # Plot bar chart of how many times each emotion has been detected
        self.ax2.bar(emotions, counts, color="orange")
        self.ax2.set_title("Session Emotions (Counts)")
        self.ax2.tick_params(axis='x', labelsize=8)
        self.ax2.tick_params(axis='y', labelsize=8)
        self.ax2.set_ylabel("Confidence", fontsize=9)
        self.ax2.set_title("Emotion Spectrum (Bar Chart)", fontsize=10)
        self.spectrum_canvas.draw()

    def trigger_alert(self, frame_bgr, emotion, confidence):
        """
        Triggers an alert if a high-risk emotion is detected above the danger threshold.
        - Plays a system bell sound.
        - Saves a snapshot of the face with a timestamp.
        """
        self.root.bell()
        alert_msg = f"ALERT! High-risk emotion detected: {emotion} ({confidence*100:.1f}%)"
        print(alert_msg)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"snapshots/alert_{emotion}_{timestamp_str}.jpg"
        cv2.imwrite(snapshot_filename, frame_bgr)
        print(f"Snapshot saved: {snapshot_filename}")

    # ---------------------------------------------------------------------------
    # Resource Usage
    # ---------------------------------------------------------------------------
    def update_resource_usage(self):
        """
        Periodically updates and displays the system's CPU usage in the UI.
        Uses psutil.cpu_percent() to get CPU usage. Schedules itself again after 2s.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        self.resource_label.config(text=f"CPU Usage: {cpu_percent:.0f}%")
        # Re-run this function after 2000ms (2 seconds)
        self.root.after(2000, self.update_resource_usage)


###############################################################################
# Main Entry Point
###############################################################################
def launch_gui():
    """
    Creates the main Tkinter window, initializes the BorderControlFERGUI,
    and starts the Tk event loop.
    """
    root = tk.Tk()
    app = BorderControlFERGUI(root)
    root.mainloop()
