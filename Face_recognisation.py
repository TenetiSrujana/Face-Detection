import cv2
import numpy as np

# Load the pre-trained Haar Cascade model for face detection
face_classifier = cv2.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces are detected, return the original image
    if len(faces) == 0:
        return img

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

print("Live face detection started. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Detect faces in the frame
    frame = detect_faces(frame)

    # Display the frame with detected faces
    cv2.imshow('Video Face Detection', frame)

    # Break the loop if 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close the video window
cap.release()
cv2.destroyAllWindows()
