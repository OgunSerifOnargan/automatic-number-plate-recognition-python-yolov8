import imutils
import cv2

class motion_detection():
    def __init__(self):
        self.min_area = 100
        self.motion_detected = True
        self.previous_frame = None
        self.current_frame = None

    def set_motion_detected(self, state):
        self.motion_detected = state
		
    def motion_checker(self, firstFrame, secondFrame):
        diff = cv2.absdiff(firstFrame, secondFrame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours( dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 2000:
                pass
            else:
                print("MOTION IS DETECTED")
                self.motion_detected = True