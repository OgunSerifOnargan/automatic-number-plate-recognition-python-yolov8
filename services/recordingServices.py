import cv2

def record_frames(stop_event, recording_queue, recorder_option):
    cap = cv2.VideoCapture(recorder_option)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video = 'output_video.mp4'
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    while not stop_event.is_set():
        frame = recording_queue.get()

    out.release()



    cap.release()
