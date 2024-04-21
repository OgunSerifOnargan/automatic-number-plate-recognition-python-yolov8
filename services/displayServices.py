import cv2
def display_frames(stop_event, display_queue):
    while not stop_event.is_set():
        if not display_queue.empty():
            display_frame = display_queue.get()
            cv2.imshow('Frame', display_frame)
            cv2.waitKey(1)