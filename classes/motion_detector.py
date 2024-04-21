import imutils
import cv2

class motion_detection():
    def __init__(self):
        self.min_area = 100
        self.motion_detected = False
        self.firstFrame = None

    def set_motion_detected(self, state):
        self.motion_detected = state
		
    def first_frame_preparer(self, frame):
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the first frame is None, initialize it
        if self.firstFrame is None:
            self.firstFrame = gray
            return self.firstFrame, gray
        else:
            return self.firstFrame, gray
        
    def motion_checker(self, firstFrame, gray):
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.min_area:
                continue
            else:
                break
        print("Motion is detected")
        self.motion_detected = True