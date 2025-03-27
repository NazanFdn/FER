import cv2
import numpy as np
import time

def process_video(self, file_path):
    """
    Reads the video, processes each frame, detects the first face, preprocesses it,
    and makes a prediction using the loaded model. Draws a bounding box on the face for demonstration.
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        self.results_label.config(text="Error: Unable to open video file.")
        print(f"Error: Unable to open video file at {file_path}")
        return

    last_prediction_time = time.time()  # Initialize time for first prediction

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Detect face using the frame captured from the video
            face_roi, x, y, w, h, image_bgr = detect_face_opencv(frame)

            # Get the current time and check if it's time to make a prediction
            current_time = time.time()
            if current_time - last_prediction_time >= 1:  # Every second
                # Preprocess the image using the preprocessing module
                preprocessed_face = preprocess_image(face_roi)

                # Predict emotion
                if self.model is not None:
                    preds = self.model.predict(preprocessed_face)

                    # Print the raw output of the model for debugging
                    print("Model output (raw probabilities):", preds)

                    # Check the predicted class
                    predicted_class = np.argmax(preds)
                    print(f"Predicted class index: {predicted_class}")
                    print(f"Predicted emotion: {self.emotion_labels[predicted_class]}")

                    # Get the predicted emotion label
                    predicted_emotion = self.emotion_labels[predicted_class]

                    self.results_label.config(text=f"Predicted Emotion: {predicted_emotion}")
                else:
                    self.results_label.config(text="Model not loaded. Please check your model file path.")

                # Update last prediction time
                last_prediction_time = current_time

            # Draw bounding box on the main image
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the annotated image
            self.display_image(image_bgr)

        except Exception as e:
            self.results_label.config(text=f"Error: {str(e)}")

    cap.release()
