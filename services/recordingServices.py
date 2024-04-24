import cv2

def record_frames(stop_event, recording_queue, recorder_option):
    frame_count = 0
    i = 0
    cap = cv2.VideoCapture(recorder_option)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video = f'output_video{i}.mp4'
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    while not stop_event.is_set():
        # Get frame from the recording queue
        frame = recording_queue.get()
        
        if frame is not None:
            # Write frame to the output video
            out.write(frame)
            frame_count+=1
        if frame_count == 1000:
            out.release()
        
    # Release video capture and video writer objects
    cap.release()
    out.release()
