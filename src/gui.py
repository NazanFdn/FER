import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import os
import time
import psutil
from datetime import datetime

# Matplotlib imports
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import your masked ResNet from the separate Python file
# e.g. "masked_resnet.py" in the same directory or in a src folder
from model.masking import MaskedResNet
from src.opencv_detector import detect_face_opencv


################################################################################
# If you have a separate "auth_manager.py", import it:
# from auth_manager import authenticate_user
# We'll define a fallback here:
################################################################################
try:
    from auth_manager import authenticate_user
except ImportError:
    def authenticate_user(username, password):
        return (username == "admin" and password == "admin")


################################################################################
# Preprocessing: from face ROI -> (1,48,48,1) in NumPy
################################################################################
def preprocess_image(face_roi):
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)        # shape (H,W)
    face_resized = cv2.resize(face_gray, (48, 48))                # shape (48,48)
    face_normalized = face_resized / 255.0
    # Expand dims => (1,48,48,1)
    face_expanded = np.expand_dims(face_normalized, axis=-1)
    face_expanded = np.expand_dims(face_expanded, axis=0)
    return face_expanded


################################################################################
# Full-screen login window
################################################################################
def show_login_window():
    login_window = tk.Toplevel()
    login_window.title("Login")
    login_window.attributes("-fullscreen", True)

    title_label = tk.Label(login_window, text="Login", font=("Helvetica", 24))
    title_label.pack(pady=50)

    tk.Label(login_window, text="Username:", font=("Helvetica", 16)).pack(pady=5)
    username_entry = tk.Entry(login_window, font=("Helvetica", 16))
    username_entry.pack()

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

    tk.Button(login_window, text="Login", font=("Helvetica", 16), command=attempt_login).pack(pady=30)

    def exit_fullscreen(event=None):
        login_window.attributes("-fullscreen", False)

    login_window.bind("<Escape>", exit_fullscreen)
    login_window.grab_set()
    login_window.focus_set()


################################################################################
# Main Application Class
################################################################################
class BorderControlFERGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Recognition System")
        self.root.geometry("1000x700")

        # -- Color Palette and Styling --
        self.bg_color = "#1e1e2f"
        self.accent_color = "#9B1313"
        self.text_color = "#ffffff"
        self.button_text = "#ffffff"
        self.button_shadow = "#8e0000"
        self.root.configure(bg=self.bg_color)

        # Tkinter Style
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
        self.style.map(
            "Accent.TButton",
            relief=[('pressed', 'sunken'), ('active', 'ridge')],
            background=[('active', self.button_shadow), ('!active', self.accent_color)]
        )

        # Load model (PyTorch)
        self.load_model()

        # Emotion labels
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.high_risk_emotions = {"Angry", "Fear"}
        self.danger_threshold = 0.70

        # Flags & Variables
        self.is_webcam_running = False
        self.is_paused = False
        self.video_cap = None
        self.frame_count = 0
        self.last_prediction_time = time.time()

        # Create directory for snapshots if needed
        os.makedirs("snapshots", exist_ok=True)

        # Canvas display
        self.canvas_width = 600
        self.canvas_height = 450

        # Track session emotion counts
        self.emotion_counts = {label: 0 for label in self.emotion_labels}

        # Build GUI
        self.create_widgets()

        # Show login window
        show_login_window()

    ############################################################################
    # Load PyTorch Model Instead of tf.keras
    ############################################################################
    def load_model(self):
        default_model_path = "/Users/zeynep/PycharmProjects/FER/model/best_masked_resnet18_48x48.pth"
        if not os.path.exists(default_model_path):
            print("No model found at:", default_model_path)
            self.model = None
            return

        try:
            # Initialize a masked ResNet with the same structure as training
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # match your training setup: resnet18, pretrained=False (since we have the .pth)
            self.model = MaskedResNet(arch="resnet18", pretrained=False, num_classes=7, dropout_p=0.3)
            state_dict = torch.load(default_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)

            print(f"PyTorch model loaded from {default_model_path}")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            self.model = None

    ############################################################################
    # Create Widgets
    ############################################################################
    def create_widgets(self):
        # ---------- Top Info Frame ----------
        info_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=20)
        info_frame.pack(fill=tk.X)
        header_label = ttk.Label(
            info_frame,
            text="Facial Expression Recognition System\nBorder Security Prototype",
            style="Header.TLabel",
            wraplength=980,
            justify="left"
        )
        header_label.pack()

        # ---------- Button Frame ----------
        button_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=10)
        button_frame.pack(fill=tk.X, pady=5)

        for i in range(4):
            button_frame.columnconfigure(i, weight=1)

        self.webcam_btn = ttk.Button(
            button_frame,
            text="Start Webcam",
            style="Accent.TButton",
            command=self.toggle_webcam
        )
        self.webcam_btn.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        self.pause_btn = ttk.Button(
            button_frame,
            text="Pause/Resume",
            style="Accent.TButton",
            command=self.pause_resume
        )
        self.pause_btn.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

        upload_video_btn = ttk.Button(
            button_frame,
            text="Upload Video",
            style="Accent.TButton",
            command=self.upload_video
        )
        upload_video_btn.grid(row=0, column=2, padx=10, pady=5, sticky="nsew")

        upload_image_btn = ttk.Button(
            button_frame,
            text="Upload Image",
            style="Accent.TButton",
            command=self.upload_image
        )
        upload_image_btn.grid(row=0, column=3, padx=10, pady=5, sticky="nsew")

        # ---------- Secondary Controls ----------
        controls_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=5)
        controls_frame.pack(fill=tk.X)

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

        self.resource_label = ttk.Label(
            controls_frame,
            text="CPU Usage: 0%",
            style="Custom.TLabel"
        )
        self.resource_label.pack(side=tk.RIGHT, padx=10)

        # ---------- Main area: Canvas + Graphs ----------
        main_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for video feed
        self.canvas = tk.Canvas(
            main_frame,
            bg='#2e2e3e',
            width=600,
            height=450,
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # 1) Bar Chart (top-right)
        self.fig = Figure(figsize=(3, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim([0, 1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.bar_canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.bar_canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # 2) Session Emotions (bottom-right)
        self.fig2 = Figure(figsize=(3, 2), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Session Emotions (Counts)")
        self.spectrum_canvas = FigureCanvasTkAgg(self.fig2, master=main_frame)
        self.spectrum_canvas.get_tk_widget().grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Results label
        self.results_label = ttk.Label(
            self.root,
            text="Predicted Emotion: [None]",
            style="Header.TLabel",
            padding=15
        )
        self.results_label.pack()

        # Periodic CPU usage
        self.update_resource_usage()


    def on_canvas_configure(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height

    # ---------------------------------------------------------------------------
    # HELPER: Clear bar charts
    # ---------------------------------------------------------------------------
    def clear_charts(self):
        self.ax.clear()
        self.ax.set_ylim([0, 1])
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("Emotion Spectrum (Bar Chart)")
        self.bar_canvas.draw()

        self.ax2.clear()
        self.ax2.set_title("Session Emotions (Counts)")
        self.spectrum_canvas.draw()
        # If you want to reset the session counts:
        # self.emotion_counts = {label: 0 for label in self.emotion_labels}

    # ---------------------------------------------------------------------------
    # Webcam / Video Logic
    # ---------------------------------------------------------------------------
    def toggle_webcam(self):
        if self.is_webcam_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
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
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.is_webcam_running = False
        self.webcam_btn.config(text="Start Webcam")
        self.results_label.config(text="Predicted Emotion: [None]")
        self.canvas.delete("all")

    def pause_resume(self):
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
        if not self.is_webcam_running or self.is_paused:
            return
        ret, frame = self.cap.read()
        if ret:
            self.detect_and_classify(frame)
        self.root.after(10, self.capture_frames)

    # ---------------------------------------------------------------------------
    # Detection / Classification with PyTorch
    # ---------------------------------------------------------------------------
    def detect_and_classify(self, frame):
        try:
            detection_result = detect_face_opencv(frame)
            if detection_result is None:
                self.display_image(frame)
                self.results_label.config(text="No face detected.")
                return

            face_roi, x, y, w, h, image_bgr = detection_result

            if self.model is not None:
                # Preprocess to shape (1,48,48,1) in NumPy
                preprocessed_face = preprocess_image(face_roi)  # shape (1,48,48,1)

                # Convert to PyTorch tensor => shape (1,1,48,48)
                face_tensor = torch.from_numpy(preprocessed_face).float()  # shape (1,48,48,1)
                face_tensor = face_tensor.permute(0, 3, 1, 2)              # => (1,1,48,48)
                # If your masked model expects 3 channels, replicate
                face_tensor = face_tensor.expand(-1, 3, -1, -1)            # => (1,3,48,48)
                face_tensor = face_tensor.to(self.device)

                # Inference
                self.model.eval()
                with torch.no_grad():
                    output = self.model(face_tensor)  # shape (1,7)

                # Softmax -> predictions
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # shape (7,)
                predicted_class = np.argmax(probs)
                predicted_emotion = self.emotion_labels[predicted_class]
                confidence = probs[predicted_class]

                # 1) Update label
                self.results_label.config(
                    text=f"Predicted Emotion: {predicted_emotion} ({confidence*100:.1f}%)"
                )

                # 2) Update bar chart
                self.update_bar_chart(probs)

                # 3) Update counts
                self.update_emotion_counts_chart(predicted_emotion)

                # 4) High-risk => alert
                color = (0, 255, 0)
                self.danger_threshold = self.threshold_var.get()
                if (predicted_emotion in self.high_risk_emotions) and (confidence >= self.danger_threshold):
                    color = (0, 0, 255)
                    self.trigger_alert(frame, predicted_emotion, confidence)

                # 5) Draw bounding box
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                self.display_image(image_bgr)
            else:
                self.results_label.config(text="Model not loaded.")
                self.display_image(image_bgr)

        except Exception as e:
            self.results_label.config(text=f"Error: {str(e)}")

    def display_image(self, image_bgr):
        frame_resized = cv2.resize(image_bgr, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(frame_resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        self.canvas.image = tk_img

    # ---------------------------------------------------------------------------
    # Video Playback
    # ---------------------------------------------------------------------------
    def upload_video(self):
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
            self.video_cap.release()
            self.results_label.config(text="Video playback finished.")
            return

        self.frame_count += 1
        frame_interval = 5
        if self.frame_count % frame_interval == 0:
            self.detect_and_classify(frame)
        else:
            self.display_image(frame)

        self.root.after(15, self.play_video_frames)

    # ---------------------------------------------------------------------------
    # Image Upload
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Chart Updates
    # ---------------------------------------------------------------------------
    def update_bar_chart(self, predictions):
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
        self.emotion_counts[predicted_emotion] += 1
        self.ax2.clear()
        emotions = list(self.emotion_counts.keys())
        counts = list(self.emotion_counts.values())
        self.ax2.bar(emotions, counts, color="orange")
        self.ax2.set_title("Session Emotions (Counts)")
        self.ax2.tick_params(axis='x', labelsize=8)
        self.ax2.tick_params(axis='y', labelsize=8)
        self.ax2.set_ylabel("Confidence", fontsize=9)
        self.ax2.set_title("Emotion Spectrum (Bar Chart)", fontsize=10)
        self.spectrum_canvas.draw()

    def trigger_alert(self, frame_bgr, emotion, confidence):
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
        cpu_percent = psutil.cpu_percent(interval=None)
        self.resource_label.config(text=f"CPU Usage: {cpu_percent:.0f}%")
        self.root.after(2000, self.update_resource_usage)


################################################################################
# Main Entry Point
################################################################################
def launch_gui():
    root = tk.Tk()
    app = BorderControlFERGUI(root)
    root.mainloop()
