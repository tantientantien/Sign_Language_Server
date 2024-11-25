import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Mediapipe setup
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
sequence_length = input_details[0]['shape'][1]  # Number of timesteps
expected_features = input_details[0]['shape'][2]  # Features per timestep

# Define actions and variables
actions = np.array(['Ăn uống', 'Cảm ơn', 'Lặp lại', 'Mẹ', 'Nguy hiểm', 
                    'Ngủ', 'Tắm', 'Tạm biệt', 'Thêm', 'Xin chào', 'Xin lỗi'])
sequence = []
sentence = []
threshold = 0.8

# Extract keypoints (ensure it matches model features)
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])  # Total: 126 features

# Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image.flags.writeable = False  # Optimize for inference
    results = model.process(image)  # Make predictions
    image.flags.writeable = True  # Make image writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results

# Drawing landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Left hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Right hand

# Open webcam feed
cap = cv2.VideoCapture(0)

# Use Mediapipe Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()  # Read frame
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_landmarks(image, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]  # Keep the last N timesteps

        # Perform prediction if sequence is ready
        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0).astype(input_details[0]['dtype'])  # Shape: [1, timesteps, features]
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            # Determine action
            if output_data[np.argmax(output_data)] > threshold:
                if len(sentence) == 0 or actions[np.argmax(output_data)] != sentence[-1]:
                    sentence.append(actions[np.argmax(output_data)])

            # Limit sentence to last action
            sentence = sentence[-1:]
        print(sentence)

        # Display results
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Language Recognition', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()