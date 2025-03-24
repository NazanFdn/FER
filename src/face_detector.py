import cv2


def detect_face(image_path):
    """
    Detects the first face in an image using OpenCV Haar Cascade Classifier.

    Parameters:
    - image_path: Path to the input image.

    Returns:
    - face_roi: The region of interest (ROI) containing the detected face.
    - x, y, w, h: Coordinates of the bounding box around the face.
    - image_bgr: The original image in BGR format.
    """
    # Load the image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise Exception("Unable to open image.")

    # Resize image for better detection (optional, adjust according to your needs)
    image_bgr = cv2.resize(image_bgr, (800, 600))  # Resizing to a manageable size

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Haar Cascade classifier could not be loaded.")

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    if len(faces) == 0:
        raise Exception("No face detected.")

    # Get the first face
    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]

    return face_roi, x, y, w, h, image_bgr
