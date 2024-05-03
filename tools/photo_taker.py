import cv2
import time

def capture_photo(save_path='reference.jpg', max_attempts=30, delay_seconds=1):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Waiting for a non-black frame...")

    for _ in range(max_attempts):
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was read successfully and is not black
        if ret and cv2.mean(frame)[0] > 40:  # Adjust the threshold as needed
            cv2.imwrite(save_path, frame)

            

        # Wait for a short duration before the next attempt
        time.sleep(delay_seconds)

    # Release the webcam
    cap.release()

    if ret:
        # Save the captured frame as a JPEG image
        cv2.imwrite(save_path, frame)
        print(f"Photo captured and saved as {save_path}")
    else:
        print("Error: Could not capture a non-black frame.")

# Capture photo with waiting for a non-black frame and save it with the default filename 'captured_photo.jpg'
capture_photo()
