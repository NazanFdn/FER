import dlib
import cv2

detector = dlib.get_frontal_face_detector()

def detect_face_dlib(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise Exception("Unable to open image.")

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    #for face in faces:
    #    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    #    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imshow("Detected Faces", image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if len(faces) == 0:
        raise Exception("No face detected."+ image_path)

    # Get the first face
    face = faces[0]
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return face, x, y, w, h, image_bgr