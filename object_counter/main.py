import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection on the frame
    bbox, label, conf = cv.detect_common_objects(frame)

    # Draw bounding box and count objects
    out = draw_bbox(frame, bbox, label, conf)

    # Show object count on screen
    object_count = len(bbox)
    cv2.putText(out, f'Object count: {object_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Object Detection and Counting", out)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

