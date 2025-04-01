import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Function to calculate the EAR (Eye Aspect Ratio) to detect blinks
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Load dlib's pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Set thresholds and frame counter
EYE_AR_THRESH = 0.25  # Threshold for drowsiness (lower means more sensitive)
EYE_AR_CONSEC_FRAMES = 48  # Number of frames the eye aspect ratio should stay below threshold to count as a blink
COUNTER = 0
ALERT_COUNT = 0

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Get the coordinates for the left and right eyes
        left_eye = []
        right_eye = []

        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        # Convert the coordinates to numpy arrays
        left_eye = np.array(left_eye)
        right_eye = np.array(right_eye)

        # Calculate the EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)

        # Calculate the average EAR
        ear = (left_EAR + right_EAR) / 2.0

        # Check if the EAR is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                ALERT_COUNT += 1
            COUNTER = 0

        # Alert the driver if drowsiness is detected
        if ALERT_COUNT >= 1:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw the eyes on the face (for visualization)
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the frame with the eye detection
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
