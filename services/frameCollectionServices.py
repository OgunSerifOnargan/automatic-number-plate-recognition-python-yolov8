import cv2
def collect_frames(stop_event, display_queue, face_rec_queue, recorder_option):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video = 'output_video.mp4'
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        face_rec_queue.put(frame, block=True, timeout=200)

    cap.release()
