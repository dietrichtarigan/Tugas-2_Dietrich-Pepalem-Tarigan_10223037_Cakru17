import cv2
from ultralytics import YOLO

# Load the YOLO model
yolo = YOLO('yolov8s.pt')

# Start video capture
videoCap = cv2.VideoCapture(0)

# Function to get unique colors for each class
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i] + increments[color_index][i] * (cls_num // len(base_colors)) % 256 
        for i in range(3)
    ]
    return tuple(color)

# Main loop for real-time object tracking
while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    # Perform object tracking with YOLO
    results = yolo.track(frame, stream=True)

    # Hitung jumlah objek yang terdeteksi
    total_objects = sum(len(result.boxes) for result in results)

    for result in results:
        class_names = result.names

        # Iterate over each detected box
        for box in result.boxes:
            # Only proceed if confidence is above 40%
            if box.conf[0] > 0.4:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class ID and name
                cls = int(box.cls[0])
                class_name = class_names[cls]

                # Get color for the class
                color = getColours(cls)

                # Draw bounding box and add text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f'{class_name} {box.conf[0]:.2f}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # Tampilkan jumlah objek yang terdeteksi di pojok kiri atas
    cv2.putText(
        frame,
        f'Total Objects Detected: {total_objects}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Display the frame
    cv2.imshow('YOLO Object Tracking', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
videoCap.release()
cv2.destroyAllWindows()
