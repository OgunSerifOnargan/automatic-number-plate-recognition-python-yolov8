import cv2

def main():
    # RTSP URL of the stream
    rtsp_url = "rtsp://admin:endurans2024.@192.168.0.161:554/Streaming/channels/1" #rtsp://<username>:<password>@<IP address of device>:<RTSP port>/Streaming/channels/<channel number><stream number>

    # Create a VideoCapture object
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Loop to capture and display frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check for the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
