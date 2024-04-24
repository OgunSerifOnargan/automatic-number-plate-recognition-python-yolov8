import cv2
def collect_frames(stop_event, display_queue, face_rec_queue, recorder_option):
    cap = cv2.VideoCapture("rtsp://192.168.1.101")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if not face_rec_queue.full():
            face_rec_queue.put(frame, block=True, timeout=200)

    cap.release()
