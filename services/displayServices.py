import cv2
# def display_frames(stop_event, display_queue):
#     while not stop_event.is_set():
#         if not display_queue.empty():
#             display_frame = display_queue.get()
#             cv2.imshow('Frame', display_frame)
#             cv2.waitKey(1)

def display_frames(stop_event, display_queue, output_filename, fps, frame_size):
    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    
    try:
        while not stop_event.is_set():
            if not display_queue.empty():
                display_frame = display_queue.get()
                
                # Display the frame
                cv2.imshow('Frame', display_frame)
                cv2.waitKey(1)
                
                # Write the frame to the video file
                out.write(display_frame)
    finally:
        # Ensure the VideoWriter object is released properly
        out.release()
        cv2.destroyAllWindows()
