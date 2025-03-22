"""
Facial Expression Recognition GUI for Border Control
-----------------------------------------------------
Author : Nazan Kafadaroglu
Date   : 2025-02-01
Course : Computer Science
Project: Facial Expression Recognition in Border Control for Augmented Security

Description:
This GUI demonstrates how a trained CNN-based Facial Expression Recognition (FER)
system can be used in a border control setting to detect and classify emotions.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

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

        # Attempt to load the pre-trained FER model
        self.load_model()

        # Label mapping (adjust according to your trained model's output layer)
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # Build all GUI elements
        self.create_widgets()

    def load_model(self):
        """
        Loads the pre-trained Keras model from disk.
        Update the path if your model is saved elsewhere.
        """
        default_model_path = "model\model.keras"
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
        # ------------------ TOP FRAME FOR HEADLINE/DISCLAIMER ------------------ #
        info_frame = ttk.Frame(self.root, padding=10)
        info_frame.pack(fill=tk.X)

        # Header / disclaimers referencing your project context
        header_label = ttk.Label(
            info_frame,
            text=(
                "Border Control Facial Expression Recognition System\n"
                "-------------------------------------------------\n"
                "This demonstration system classifies facial expressions to aid border control\n"
                "officers in detecting potential stress or suspicious behavior. Please note:\n"
                "1) Images are processed locally and not stored.\n"
                "2) Ethical, privacy, and bias considerations are paramount.\n"
            ),
            wraplength=780,
            justify="left"
        )
        header_label.pack()

        # ------------------ BUTTON FRAME ------------------ #
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)

        upload_btn = ttk.Button(button_frame, text="Upload Image", command=self.upload_image)
        upload_btn.pack(side=tk.LEFT, padx=5)

        # You can add more buttons if needed, e.g. to show logs, or load a different model

        # ------------------ IMAGE DISPLAY CANVAS ------------------ #
        # We'll use a 400x400 display area for the uploaded or processed image
        self.canvas = tk.Canvas(self.root, bg='gray', width=400, height=400)
        self.canvas.pack(pady=10)

        # ------------------ LABEL FOR RESULTS ------------------ #
        self.results_label = ttk.Label(
            self.root,
            text="Predicted Emotion: [None]",
            font=('Helvetica', 14, 'bold'),
            padding=10
        )
        self.results_label.pack()

    def upload_image(self):
        """
        Opens a file dialog for the user to select an image. On successful selection,
        calls process_image() for detection and classification.
        """
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=filetypes)

        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        """
        Reads the image, detects the first face, preprocesses it, and makes a prediction
        using the loaded model. Draws a bounding box on the face for demonstration.

        Parameters
        ----------
        file_path : str
            The path to the chosen image file.
        """
        # Load image with OpenCV
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self.results_label.config(text="Error: Unable to open image.")
            return

        # Convert from BGR -> RGB for display
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Use Haar cascade (can be replaced by a more advanced face detector)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            self.results_label.config(text="No face detected.")
            self.display_image(image_rgb)
            return

        # For simplicity, we only process the first face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Resize to match your model input. If your model expects 48x48, 64x64, or 224x224, change here
        target_size = (48, 48)  # Example: 48x48 if that’s your training config
        face_roi = cv2.resize(face_roi, target_size)

        # Normalize & shape for model
        face_roi = face_roi.astype("float32") / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)  # -> (48,48,1)
        face_roi = np.expand_dims(face_roi, axis=0)   # -> (1,48,48,1)

        # Check if model is loaded
        if self.model is not None:
            try:
                predictions = self.model.predict(face_roi)
                predicted_index = np.argmax(predictions)
                predicted_emotion = self.emotion_labels[predicted_index]
                self.results_label.config(text=f"Predicted Emotion: {predicted_emotion}")
            except Exception as e:
                print(f"Error during prediction: {e}")
                self.results_label.config(text="Error: Model prediction failed.")
        else:
            self.results_label.config(text="Model not loaded. Please check your model file path.")

        # Draw bounding box on the main image
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display annotated image
        self.display_image(image_rgb)

    def display_image(self, image_rgb):
        """
        Utility function to display an RGB image on the Tkinter canvas.

        Parameters
        ----------
        image_rgb : np.ndarray
            The image in RGB format (height x width x channels).
        """
        # Set display size
        display_size = (400, 400)
        display_img = cv2.resize(image_rgb, display_size, interpolation=cv2.INTER_AREA)

        # Convert to PIL & display on canvas
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(
            display_size[0] // 2,
            display_size[1] // 2,
            image=tk_img,
            anchor=tk.CENTER
        )
        # Keep a reference so it’s not garbage-collected
        self.canvas.image = tk_img


def launch_gui():
    root = tk.Tk()
    app = BorderControlFERGUI(root)
    root.mainloop()
