import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
#from src.face_detector import detect_face  # Importing face detection module
from src.preprocess import preprocess_image  # Importing preprocessing module
from src.dlib_detetctor import detect_face_dlib


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
        # Overall color scheme for professional look
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
        self.style.configure("Custom.TFrame",
                             background=self.bg_color)

        # Configure style for labels
        self.style.configure("Custom.TLabel",
                             background=self.bg_color,
                             foreground=self.text_color,
                             font=("Helvetica", 12))

        # Configure style for headers/bold text
        self.style.configure("Header.TLabel",
                             background=self.bg_color,
                             foreground=self.text_color,
                             font=("Helvetica", 14, "bold"))

        # Configure style for accent buttons
        self.style.configure("Accent.TButton",
                             background=self.accent_color,
                             foreground=self.button_text,
                             borderwidth=0,  # Remove the border
                             focusthickness=3,
                             focuscolor="none",
                             padding=10,
                             relief="flat")  # Flat effect (no 3D border)

        # Add shadow effect to buttons for 3D appearance
        self.style.map("Accent.TButton",
                       relief=[('active', 'sunken'), ('!active', 'flat')],
                       background=[('active', self.button_shadow), ('!active', self.accent_color)])

        # Load pre-trained FER model
        self.load_model()

        # Label mapping
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # Build all GUI elements
        self.create_widgets()

    def load_model(self):
        """
        Loads the pre-trained Keras model from disk.
        Update the path if your model is saved elsewhere.
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

        # ------------------ TOP FRAME / DISCLAIMER ------------------ #
        info_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=20)
        info_frame.pack(fill=tk.X)

        header_label = ttk.Label(
            info_frame,
            text=(
                "Border Control Facial Expression Recognition System\n"),
            style="Header.TLabel",
            wraplength=780,
            justify="left"
        )
        header_label.pack()

        # ------------------ BUTTON FRAME ------------------ #
        button_frame = ttk.Frame(self.root, style="Custom.TFrame", padding=10)
        button_frame.pack(fill=tk.X, pady=30)

        upload_btn = ttk.Button(
            button_frame,
            text="Upload Image",
            style="Accent.TButton",
            command=self.upload_image
        )
        upload_btn.pack(side=tk.TOP, pady=10)  # Center the button

        # ------------------ IMAGE DISPLAY CANVAS ------------------ #
        self.canvas = tk.Canvas(self.root, bg='#2e2e3e', width=400, height=400)
        self.canvas.pack(pady=10)

        # ------------------ LABEL FOR RESULTS ------------------ #
        self.results_label = ttk.Label(
            self.root,
            text="Predicted Emotion: [None]",
            style="Header.TLabel",
            padding=15
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
        """
        try:
            # Detect face using the face_detection module
            face_roi, x, y, w, h, image_bgr = detect_face_dlib(file_path)

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
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display annotated image
            self.display_image(image_bgr)

        except Exception as e:
            self.results_label.config(text=f"Error: {str(e)}")

    def display_image(self, image_rgb):
        """
        Utility function to display an RGB image on the Tkinter canvas.
        """
        # Set display size
        display_size = (400, 400)
        display_img = cv2.resize(image_rgb, display_size, interpolation=cv2.INTER_AREA)

        # Convert to PIL & display on canvas
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(
            display_size[0]//2,
            display_size[1]//2,
            image=tk_img,
            anchor=tk.CENTER
        )
        # Keep a reference to avoid garbage collection
        self.canvas.image = tk_img


def launch_gui():
    root = tk.Tk()
    app = BorderControlFERGUI(root)
    root.mainloop()