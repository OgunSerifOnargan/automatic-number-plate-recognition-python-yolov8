import cv2
def collect_frames(stop_event, rawFrame_queue, recorder_option):
    cap = cv2.VideoCapture(recorder_option)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if not rawFrame_queue.full():
            rawFrame_queue.put(frame, block=True, timeout=200)

    cap.release()
