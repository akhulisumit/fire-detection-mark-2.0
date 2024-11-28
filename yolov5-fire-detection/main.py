import cv2
import torch
import numpy as np

# Load the custom trained YOLOv5 model
model_path = 'D:/fire/yolov5-fire-detection/model/yolov5s_best.pt'
model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=model_path)

# Initialize the webcam (camera index 0 by default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference on the captured frame
    results = model(frame)

    # Extract predictions: boxes, confidences, and class IDs
    pred = results.pred[0]  # Prediction results

    # Loop through the detections (boxes, confidence, class_id)
    for det in pred:
        x1, y1, x2, y2, conf, cls = det.tolist()  # Extract coordinates, confidence, and class_id

        # Only consider detections with high confidence
        if conf > 0.4:  # Confidence threshold
            label = results.names[int(cls)]  # Get class label
            if label == 'fire':  # Check if the detected class is 'fire'
                # Draw bounding box around fire detection
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Fire Detection - Live', frame)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
