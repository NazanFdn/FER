import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from src.opencv_detector import detect_face_opencv
from src.face_detector_image import detect_face_dnn
from src.preprocess import preprocess_image


class BorderControlFERGUI:
    """
    A GUI application for demonstrating Facial Expression Recognition.
    It supports webcam, video file, and image file input.
    """

    def __init__(self, root):
        """
        Initialize the GUI, load model, and create widgets.

        Parameters
        ----------
        root : tk.Tk
            The main Tkinter root window.
        """
        self.root = root
        self.root.title("Facial Expression Recognition System")
        self.root.geometry("800x600")

        # -- Color Palette and Styling --
        self.bg_color = "#1e1e2f"       # Dark background for modern look
        self.accent_color = "#9B1313"   # Accent red color
        self.text_color = "#ffffff"     # White text for contrast
        self.button_text = "#ffffff"    # White button text
        self.button_shadow = "#8e0000"  # Shadow for 3D effect

        # Set root background
        self.root.configure(bg=self.bg_color)

        # -- Create a style --
        self.style = ttk.Style()
        self.style.theme_use("clam")  # "clam" for a more modern look

        # Frame style
        self.style.configure("Custom.TFrame", background=self.bg_color)

        # Label style
        self.style.configure("Custom.TLabel",
                             background=self.bg_color,
                             foreground=self.text_color,
                             font=("Helvetica", 12))

        # Header label style
        self.style.configure("Header.TLabel",
                             background=self.bg_color,
                             foreground=self.text_color,
                             font=("Helvetica", 14, "bold"))

        # 3D Button style: "raised" normally, "sunken" on active
        # Increase borderwidth for a more pronounced 3D effect
        self.style.configure("Accent.TButton",
                             background=self.accent_color,
                             foreground=self.button_text,
                             borderwidth=2,
                             focusthickness=3,
                             focuscolor="none",
                             padding=10,
                             relief="raised",
                             anchor="center")

        self.style.map("Accent.TButton",
                       relief=[('pressed', 'sunken'), ('active', 'ridge')],
                       background=[('active', self.button_shadow), ('!active', self.accent_color)])

        # -- Load pre-trained FER model --
        self.load_model()

        # Label mapping
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # -- Build GUI elements --
        self.create_widgets()

        # Webcam & Video states
        self.is_webcam_running = False
        self.video_cap = None  # Will hold the cv2.VideoCapture object
        self.frame_count = 0
        self.last_prediction_time = time.time()

    def load_model(self):
        """
        Loads the pre-trained Keras model from disk.
        """
        default_model_path = "model/mobilenetV2_fer_model2.keras"
        if os.path.exists(default_model_path):
            try:
                self.model = tf.keras.models.load_model(default_model_path)
                print(f"Model loaded successfully from {default_model_path}")
            except Exception as e:
                print(f"Error loading model from {default_model_path}: {e}")
                self.model = None
        else:
            print(f"No pre-trained model found at {default_model_path}")
            self.model = None

    def create_widgets(self):
        """
        Creates and places all GUI widgets.
        """
        # -- Top info frame --
        info_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=20)
        info_frame.pack(fill=tk.X)

        header_label = ttk.Label(info_frame,
                                 text="Facial Expression Recognition System\n",
                                 style="Header.TLabel",
                                 wraplength=780,
                                 justify="left")
        header_label.pack()

        # -- Button frame (centered) --
        button_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=10)
        button_frame.pack(fill=tk.X, pady=20)

        # Configure grid columns to expand equally
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        self.webcam_btn = ttk.Button(button_frame,
                                     text="Start Webcam",
                                     style="Accent.TButton",
                                     command=self.toggle_webcam)
        self.webcam_btn.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        upload_video_btn = ttk.Button(button_frame,
                                      text="Upload Video",
                                      style="Accent.TButton",
                                      command=self.upload_video)
        upload_video_btn.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

        upload_image_btn = ttk.Button(button_frame,
                                      text="Upload Image",
                                      style="Accent.TButton",
                                      command=self.upload_image)
        upload_image_btn.grid(row=0, column=2, padx=10, pady=5, sticky="nsew")

        # -- Canvas for video/image display --
        self.canvas = tk.Canvas(self.root, bg='#2e2e3e', width=400, height=400, highlightthickness=0)
        self.canvas.pack(pady=10)

        # -- Results label --
        self.results_label = ttk.Label(self.root,
                                       text="Predicted Emotion: [None]",
                                       style="Header.TLabel",
                                       padding=15)
        self.results_label.pack()

    def toggle_webcam(self):
        """
        Toggles between starting and stopping the webcam.
        """
        if self.is_webcam_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        """
        Starts the webcam and continuously captures frames.
        """
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.results_label.config(text="Error: Unable to access the camera.")
            return

        self.is_webcam_running = True
        self.webcam_btn.config(text="Stop Webcam")
        self.capture_frames()

    def stop_webcam(self):
        """
        Stops the webcam, resets the button, and clears the canvas.
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        self.is_webcam_running = False
        self.webcam_btn.config(text="Start Webcam")
        self.results_label.config(text="Predicted Emotion: [None]")
        self.canvas.delete("all")

    def capture_frames(self):
        """
        Capture frames from webcam, do detection & prediction, display.
        """
        if not self.is_webcam_running:
            return

        ret, frame = self.cap.read()
        if ret:
            try:
                face_roi, x, y, w, h, image_bgr = detect_face_opencv(frame)
                preprocessed_face = preprocess_image(face_roi)

                if self.model is not None:
                    preds = self.model.predict(preprocessed_face)
                    predicted_class = np.argmax(preds)
                    predicted_emotion = self.emotion_labels[predicted_class]
                    self.results_label.config(text=f"Predicted Emotion: {predicted_emotion}")
                else:
                    self.results_label.config(text="Model not loaded.")

                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.display_image(image_bgr)

            except Exception as e:
                self.results_label.config(text=f"Error: {str(e)}")

        # Schedule next frame read
        self.root.after(10, self.capture_frames)

    def display_image(self, image_bgr):
        """
        Display an image (BGR) on the Tkinter canvas.
        """
        display_size = (400, 400)
        frame_resized = cv2.resize(image_bgr, display_size, interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(frame_resized)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(display_size[0]//2,
                                 display_size[1]//2,
                                 image=tk_img,
                                 anchor=tk.CENTER)
        self.canvas.image = tk_img

    def upload_video(self):
        """
        Opens a file dialog to choose a video and then starts playing it asynchronously.
        """
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        file_path = filedialog.askopenfilename(title="Select a Video", filetypes=filetypes)

        if file_path:
            self.start_video_playback(file_path)
        else:
            print("No video file selected.")

    def start_video_playback(self, file_path):
        """
        Initialize video capture and start the asynchronous frame reading.
        """
        # If a video is already playing, release it first
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()

        self.video_cap = cv2.VideoCapture(file_path)
        if not self.video_cap.isOpened():
            self.results_label.config(text="Error: Unable to open video file.")
            return

        # Reset counters
        self.frame_count = 0
        self.last_prediction_time = time.time()
        self.results_label.config(text="Playing Video...")

        # Clear the canvas
        self.canvas.delete("all")

        # Start reading frames asynchronously
        self.play_video_frames()

    def play_video_frames(self):
        """
        Read the next frame from the video, run detection/prediction occasionally,
        and schedule the next frame read.
        """
        if not self.video_cap or not self.video_cap.isOpened():
            return  # Video ended or not initialized

        ret, frame = self.video_cap.read()
        if not ret:
            # No more frames - release the video
            self.video_cap.release()
            self.results_label.config(text="Video playback finished.")
            return

        self.frame_count += 1

        # Only run detection/prediction on certain frames, e.g. every 5th
        frame_interval = 5
        prediction_interval = 1.0  # seconds
        current_time = time.time()

        # If it's time to do face detection + prediction
        if self.frame_count % frame_interval == 0:
            try:
                # Even if we skip prediction, we can still do face detection
                face_roi, x, y, w, h, image_bgr = detect_face_opencv(frame)

                # If enough time has passed since last prediction
                if (current_time - self.last_prediction_time) >= prediction_interval:
                    if self.model is not None:
                        preprocessed_face = preprocess_image(face_roi)
                        preds = self.model.predict(preprocessed_face)
                        predicted_class = np.argmax(preds)
                        predicted_emotion = self.emotion_labels[predicted_class]
                        self.results_label.config(text=f"Predicted Emotion: {predicted_emotion}")
                    else:
                        self.results_label.config(text="Model not loaded.")

                    self.last_prediction_time = current_time
                else:
                    # No new prediction yet, we still have face_roi for bounding box
                    pass

                # Draw bounding box
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.display_image(image_bgr)

            except Exception as e:
                self.results_label.config(text=f"Error: {str(e)}")
        else:
            # No detection/prediction, just display the raw frame
            self.display_image(frame)

        # Schedule next frame read
        # Use a small delay (e.g., 10-30ms) to mimic video playback speed
        self.root.after(15, self.play_video_frames)

    def upload_image(self):
        """
        Clears the canvas, then opens a file dialog for user to select an image.
        """
        self.canvas.delete("all")
        self.results_label.config(text="Predicted Emotion: [None]")

        filetypes = [("Image files", "*.jpg *.png *.jpeg")]
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=filetypes)

        if file_path:
            self.process_image(file_path)
        else:
            print("No image file selected.")

    def process_image(self, file_path):
        """
        Process a single image, detect face, preprocess, and predict emotion.
        """
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self.results_label.config(text="Error: Unable to load image.")
            return

        # 1. Attempt to detect face
        detection_result = detect_face_opencv(image_bgr)

        # 2. If detect_face_opencv returns None, no face was found
        if detection_result is None:
            self.results_label.config(text="No face detected in the image.")
            self.display_image(image_bgr)  # Optionally show the original image with no bounding box
            return

        # 3. Otherwise, unpack the detection results
        face_roi, x, y, w, h, image_bgr = detection_result

        # 4. Preprocess and predict
        preprocessed_face = preprocess_image(face_roi)

        if self.model is not None:
            preds = self.model.predict(preprocessed_face)
            predicted_class = np.argmax(preds)
            predicted_emotion = self.emotion_labels[predicted_class]
            self.results_label.config(text=f"Predicted Emotion: {predicted_emotion}")
        else:
            self.results_label.config(text="Model not loaded.")

        # 5. Draw bounding box and display
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.display_image(image_bgr)


def launch_gui():
    root = tk.Tk()
    app = BorderControlFERGUI(root)
    root.mainloop()
