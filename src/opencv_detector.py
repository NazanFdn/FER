import cv2

def detect_face_opencv(image_bgr):
    """
    Detect faces using OpenCV's Haar Cascade Classifier.
    """
    # Load the pre-trained Haar Cascade Classifier for frontal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the captured image to grayscale (required for face detection)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        raise Exception("No face detected.")

    # Get the first face detected (you can modify this to handle multiple faces)
    (x, y, w, h) = faces[0]

    # Draw a rectangle around the detected face
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Return the region of interest (ROI) and face coordinates
    return image_bgr[y:y + h, x:x + w], x, y, w, h, image_bgr
