from classes.face import face
from datetime import datetime
import supervision as sv
import numpy as np
import cv2
import copy

class person:
    def __init__(self, img, bbox, LINE_START, LINE_END):
        self.detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.img = img
        self.bbox = bbox
        self.lineCounter = sv.LineZone(start=LINE_START, end=LINE_END)
        self.line_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.solo_detection = None
        self.modified_trackingResults_for_lineCounter = None
        self.current_in_count = 0
        self.current_out_count = 0

        self.entranceTime = None
        self.exitTime = None
        self.face = face()
        self.name = "Unknown"

    def set_solo_detection(self, detection):
        self.solo_detection = detection
        
    def set_findings(self):
        self.face.isFaceIdentifiedProperly = True
        self.face.identification_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.name = self.face.faceProposal.name
        self.face.name = self.face.faceProposal.name
        self.face.img = self.face.faceProposal.img
        self.face.encodedVector = self.face.faceProposal.encodedVector
        self.face.bbox_defaultFrame = self.face.faceProposal.bbox_defaultFrame

    def check_where_person_is(self):
        self.placementState = self.lineCounter.in_count - self.lineCounter.out_count

        if self.lineCounter.in_count - self.current_in_count > 0:
            self.entranceTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"The person  -- {self.name} -- is !!!IN!!! now. Entrance Time: {self.entranceTime}") 
            self.current_in_count = self.lineCounter.in_count

        if self.lineCounter.out_count - self.current_out_count > 0:
            self.exitTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"The person -- {self.name} -- is !!!OUT!!! now. Entrance Time: {self.exitTime}") 
            self.current_out_count = self.lineCounter.out_count

    def update_lineCounter(self, frame):
        self.lineCounter.trigger(self.modified_solo_detection_for_lineCounter)
        frame = self.line_annotator.annotate(frame=frame, line_counter=self.lineCounter)
        return frame
    
    def modify_solo_detection_for_lineCounter(self, frame, placement="body"):
        x1 = self.solo_detection.xyxy[:,0][0]
        y1 = self.solo_detection.xyxy[:, 1][0]
        x2 = self.solo_detection.xyxy[:, 2][0]
        y2 = self.solo_detection.xyxy[:, 3][0]
        x_mean = (x1 + x2) / 2
        y_mean = (y1 + y2) / 2 
        w = x2 - x1
        h = y2 - y1
        if placement == "body":
            x1_new = np.array(x_mean - (w/4))
            x2_new = np.array(x_mean + (w/4))
            y1_new = np.array(y_mean - (h/8))
            y2_new = np.array(y_mean + (h/8))
        if placement == "foot":
            x1_new = np.array(x_mean - (w/5))
            x2_new = np.array(x_mean + (w/5))
            y1_new = np.array(y_mean + (h/2.2))
            y2_new = np.array(y_mean + (h/2))
        frame = cv2.rectangle(frame, (int(x1_new), int(y1_new)), (int(x2_new), int(y2_new)), (0, 0, 255), 2)  # (0, 0, 255) is BGR color (red), 2 is thickness
        self.modified_solo_detection_for_lineCounter = copy.deepcopy(self.solo_detection)
        self.modified_solo_detection_for_lineCounter.xyxy[0][0] = x1_new
        self.modified_solo_detection_for_lineCounter.xyxy[0][1] = y1_new
        self.modified_solo_detection_for_lineCounter.xyxy[0][2] = x2_new
        self.modified_solo_detection_for_lineCounter.xyxy[0][3] = y2_new
        return frame
    
    def update_person_img_bbox_info(self, trackerId, img_person, bbox_person):
        self.img = img_person
        self.bbox = bbox_person
#        append_string_to_csv(f"person {trackerId}'s image and bbox are updated.", 'log.csv')
    #    return people_dict
        

    

