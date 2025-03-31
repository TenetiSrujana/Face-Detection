import cv2 as cv
import numpy as np

# Load the pre-trained Haar Cascade models for face detection
face_cascade_frontal = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade_profile = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_profileface.xml")

# Start webcam capture
cap = cv.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

print("Face recognition started. Press 'Q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Convert the frame to grayscale for better detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame using both frontal and profile models
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Combine detected faces
    faces = list(faces_frontal) + list(faces_profile)

    # Count and draw rectangles around detected faces
    num_faces = len(faces)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the count of detected faces on the screen
    text = f"People detected: {num_faces}"
    cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)

    # Show the frame in a window
    cv.imshow("Face Recognition", frame)

    # Quit if 'Q' key is pressed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources and close the window
cap.release()
cv.destroyAllWindows()
