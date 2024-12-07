import cv2
import numpy as np
from joblib import load

# Load the pre-trained SVM model
svm_model = load("svm_model.joblib")  # Replace with your saved SVM model

# Define a dictionary to map predictions to letters
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O',
    14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define a region of interest (ROI) for the hand
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Preprocess the ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (28, 28))  # Resize to 28x28
    normalized = resized.astype('float32') / 255.0  # Normalize to [0, 1]
    flattened = normalized.flatten()  # Flatten to a 1D array for the SVM model

    # Predict the class of the ROI
    prediction = svm_model.predict([flattened])[0]  # Predict using the SVM model

    # Get the corresponding letter
    print(prediction)

    # Display the predicted letter on the screen
    cv2.putText(frame, f'Letter: {prediction}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Hand Sign Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
