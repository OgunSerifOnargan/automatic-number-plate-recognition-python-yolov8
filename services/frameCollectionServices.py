import cv2
def collect_frames(stop_event, face_rec_queue, recorder_option):
    cap = cv2.VideoCapture(recorder_option)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if not face_rec_queue.full():
            face_rec_queue.put(frame, block=True, timeout=200)

    cap.release()
