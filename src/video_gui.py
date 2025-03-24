import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
from src.preprocess import preprocess_image  # Importing preprocessing module
from src.opencv_detector import detect_face_opencv

class BorderControlFERGUI:
    """
    A GUI application for demonstrating
    Facial Expression Recognition in a border control context.
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
        self.root.title("Border Control Facial Expression Recognition")
        self.root.geometry("800x600")

        # ---------------------------------------------------------------------
        # 1) COLOR PALETTE & STYLING (Enhanced for Professional Look)
        # ---------------------------------------------------------------------
        self.bg_color = "#1e1e2f"  # Dark background for modern look
        self.accent_color = "#9B1313"  # Accent red color
        self.text_color = "#ffffff"  # White text for better contrast
        self.button_text = "#ffffff"  # White button text for visibility
        self.button_shadow = "#8e0000"  # Button shadow effect color for 3D look

        # Set root background
        self.root.configure(bg=self.bg_color)

        # Create a style
        self.style = ttk.Style()
        self.style.theme_use("clam")  # 'clam' for a more modern UI style

        # Configure style for frames
        self.style.configure("Custom.TFrame", background=self.bg_color)
        self.style.configure("Custom.TLabel", background=self.bg_color, foreground=self.text_color, font=("Helvetica", 12))
        self.style.configure("Header.TLabel", background=self.bg_color, foreground=self.text_color, font=("Helvetica", 14, "bold"))
        self.style.configure("Accent.TButton", background=self.accent_color, foreground=self.button_text, borderwidth=0, focusthickness=3, focuscolor="none", padding=10, relief="flat")

        self.style.map("Accent.TButton", relief=[('active', 'sunken'), ('!active', 'flat')], background=[('active', self.button_shadow), ('!active', self.accent_color)])

        # Load pre-trained FER model
        self.load_model()

        # Label mapping
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # Build all GUI elements
        self.create_widgets()

    def load_model(self):
        """
        Loads the pre-trained Keras model from disk.
        """
        default_model_path = "model.keras"
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
        Creates and places all GUI widgets, including disclaimers, buttons,
        canvas for image display, and a results label for predicted emotion.
        """
        info_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=20)
        info_frame.pack(fill=tk.X)

        header_label = ttk.Label(info_frame, text="Border Control Facial Expression Recognition System\n", style="Header.TLabel", wraplength=780, justify="left")
        header_label.pack()

        button_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=10)
        button_frame.pack(fill=tk.X, pady=30)

        upload_btn = ttk.Button(button_frame, text="Start Webcam", style="Accent.TButton", command=self.start_webcam)
        upload_btn.pack(side=tk.TOP, pady=10)

        self.canvas = tk.Canvas(self.root, bg='#2e2e3e', width=400, height=400)
        self.canvas.pack(pady=10)

        self.results_label = ttk.Label(self.root, text="Predicted Emotion: [None]", style="Header.TLabel", padding=15)
        self.results_label.pack()

    def start_webcam(self):
        """
        Starts the webcam and continuously captures frames for face detection
        and emotion recognition.
        """
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.results_label.config(text="Error: Unable to access the camera.")
            return

        self.capture_frames()

    def capture_frames(self):
        """
        Captures frames from the webcam and processes them for face detection
        and emotion prediction.
        """
        ret, frame = self.cap.read()
        if ret:
            try:
                # Detect face using the frame captured from the webcam
                face_roi, x, y, w, h, image_bgr = detect_face_opencv(frame)

                # Preprocess the image using the preprocessing module
                preprocessed_face = preprocess_image(face_roi)

                # Predict emotion
                if self.model is not None:
                    preds = self.model.predict(preprocessed_face)
                    idx = np.argmax(preds)
                    predicted_emotion = self.emotion_labels[idx]
                    self.results_label.config(text=f"Predicted Emotion: {predicted_emotion}")
                else:
                    self.results_label.config(text="Model not loaded. Please check your model file path.")

                # Draw bounding box on the main image
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the annotated image
                self.display_image(image_bgr)

            except Exception as e:
                self.results_label.config(text=f"Error: {str(e)}")

        self.root.after(10, self.capture_frames)

    def display_image(self, image_rgb):
        """
        Utility function to display an RGB image on the Tkinter canvas.
        """
        display_size = (400, 400)
        display_img = cv2.resize(image_rgb, display_size, interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(display_size[0]//2, display_size[1]//2, image=tk_img, anchor=tk.CENTER)
        self.canvas.image = tk_img

    def __del__(self):
        """Release the webcam when the program is closed"""
        if hasattr(self, 'cap'):
            self.cap.release()


def launch_gui():
    root = tk.Tk()
    app = BorderControlFERGUI(root)
    root.mainloop()
