from classes.licensePlate import licensePlate
from datetime import datetime
import supervision as sv
import numpy as np
import cv2
import copy

class vehicle:
    def __init__(self, trackerId, img_body, bbox_body, lines_sv):
        #@initializing 
        self.trackerId = trackerId
        self.img_body = img_body
        self.bbox_body = bbox_body
        self.img_skewed_plate = None
        self.img_vehicle = None
        self.detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.lineCounter = sv.LineZone(start=lines_sv[0][0], end=lines_sv[0][1])
        self.line_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
        #@processing
        self.solo_detection = None
        self.modified_trackingResults_for_lineCounter = None
        self.gone_counter = 0
        self.current_in_count = 0
        self.current_out_count = 0
        self.location_state = 1        
        #@Finalizing
        self.identificationTime = None
        self.img_licensePlate = None

        #vehicle height calculator
        # self.heightLine0 = sv.LineZone(start=line_sv[1][0], end=line_sv[1][1])
        # self.heightLine1 = sv.LineZone(start=line_sv[2][0], end=line_sv[2][1])
        self.entranceTime = None
        self.exitTime = None
        self.licensePlate = licensePlate()
        self.licenseCode = None
        
    def set_solo_detection(self, detection):
        self.solo_detection = detection
        
    def set_findings(self):
        self.identification_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.licensePlate.encodedVector = self.licensePlate.proposal.encodedVector

    def set_unidentified_findings(self, id):
        self.licensePlate.islicensePlateIdentifiedProperly = True
        self.licensePlate.identification_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.licenseCode = id
        self.img_licensePlate = self.licensePlate.img
        self.licensePlate.encodedVector = self.licensePlate.proposal.encodedVector

    def check_where_vehicle_is(self):
        self.placementState = self.lineCounter.in_count - self.lineCounter.out_count

        if self.lineCounter.in_count - self.current_in_count > 0:
            self.entranceTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"The vehicle  -- {self.licenseCode} -- is !!!IN!!! now. Entrance Time: {self.entranceTime}") 
            self.current_in_count = self.lineCounter.in_count
            self.location_state = 0

        if self.lineCounter.out_count - self.current_out_count > 0:
            self.exitTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"The vehicle -- {self.licenseCode} -- is !!!OUT!!! now. Entrance Time: {self.exitTime}") 
            self.current_out_count = self.lineCounter.out_count
            self.location_state = 1

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
    
    def update_vehicle_img_bbox_info(self, img_vehicle, bbox_vehicle):
        self.img_body = img_vehicle
        self.bbox_body = bbox_vehicle
#        append_string_to_csv(f"vehicle {trackerId}'s image and bbox are updated.", 'log.csv')
    #    return people_dict
        
    # def check_height_calculator_activated(self):
    #     if self.heightLine0.in_count != self.heightLine1.in_count or self.heightLine0.out_count != self.heightLine1.out_count:
    #         height = (self.bbox[3] - self.bbox[1]) * Mperpix
