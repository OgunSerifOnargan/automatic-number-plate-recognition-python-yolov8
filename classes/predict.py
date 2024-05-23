from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2
from services.utils import get_objects_within_time_interval
from supervision.detection.core import Detections
import csv
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class licensePlate_predictor:
    def __init__(self):
        #YOLO
        self.licensePlate_detector = YOLO("models/license_plate_detector.pt")
        self.yoloResult = None
        self.refresh_needed = False

        #Predictor: licensePlate
        self.OCR_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.OCR_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        self.licenseCode = None

    def predict_licensePlate_yolo(self, vehicle):
        if not vehicle.img_body.shape[1] == 0:
            result = self.licensePlate_detector(vehicle.img_body, verbose=False, device="mps")
            licensePlate_locations = result[0].boxes.xyxy.tolist()
            return licensePlate_locations
        else:
            return None
    
    def predict_license_number(self, vehicle):
        image_cv2_rgb = cv2.cvtColor(vehicle.img_skewed_plate, cv2.COLOR_BGR2RGB)
        # Convert the cv2 image array to PIL Image
        image = Image.fromarray(image_cv2_rgb)
        pixel_values = self.OCR_processor(image, return_tensors="pt").pixel_values
        generated_ids = self.OCR_model.generate(pixel_values, max_length=12)
        generated_text = self.OCR_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        vehicle.licensePlate.proposal.licenseCode = generated_text
        return vehicle

class vehicle_predictor:
    def __init__(self):
        #vehicle Detector
        self.vehicle_detector = YOLO('models/yolov8n.pt')
        self.vehicle_list = [2, 3, 5, 7]
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        #Predictor: Tracking
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=30)
        #Results: Tracking
        self.trackingResult = None
        self.modified_solo_detection_for_lineCounter = None
        self.trackingResult_labels = None
        self.solo_detections_dict = {}
        self.vehicle_annotated_frame = None

    def predict_vehicle(self, frame):
            self.yoloResult = self.vehicle_detector.predict(frame, verbose=False, device="mps", conf=0.50)[0]
            detections_tracking = sv.Detections.from_ultralytics(self.yoloResult)
            self.trackingResult = detections_tracking[np.isin(detections_tracking.class_id, self.vehicle_list)]
            self.trackingResult = self.tracker.update_with_detections(self.trackingResult)

    def assign_final_licenseCode_for_display(self, vehicles):
            self.trackingResult_labels = []
            for confidence, class_id, tracker_id in zip(self.trackingResult.confidence, self.trackingResult.class_id, self.trackingResult.tracker_id):
                if tracker_id in vehicles:
                    if vehicles[tracker_id].licenseCode != None:
                        self.trackingResult_labels.append(f"#{tracker_id} {vehicles[tracker_id].licenseCode} {confidence:0.2f}")
                    else:
                        self.trackingResult_labels.append(f"#{tracker_id} {self.vehicle_detector.model.names[class_id]} {confidence:0.2f}")
                else:
                    self.trackingResult_labels.append(f"#{tracker_id} {self.vehicle_detector.model.names[class_id]} {confidence:0.2f}")
            
    def annotate_vehicles(self, frame):
        self.vehicle_annotated_frame = self.box_annotator.annotate(scene=frame, detections=self.trackingResult, labels=self.trackingResult_labels)

    def display_results(self, display_queue, frame, vehicles):
        #Log Display
        vehicles_in_time_interval = get_objects_within_time_interval(vehicles, 30)
        # Display rows on top right corner of the frame
        display_frame = _display_rows_on_frame(vehicles_in_time_interval, frame)
        # vehicle Annotator
        self.annotate_vehicles(display_frame)
        #licensePlate Image Displayer

        display_queue.put(self.vehicle_annotated_frame)

    def crop_objects(self, image):
        cropped_images_info = {}
        for box, tracker_id in zip(self.trackingResult.xyxy, self.trackingResult.tracker_id):
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]

            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_bbox = [x_min, y_min, x_max, y_max]

            cropped_images_info[tracker_id] = [cropped_image, cropped_bbox]
        return cropped_images_info
    
    def separate_detections(self, vehicle, trackerId):
        for i in range(len(self.trackingResult.xyxy)):
            if self.trackingResult.tracker_id[i:i+1][0] == trackerId:
                solo_detection = Detections(
                    xyxy=self.trackingResult.xyxy[i:i+1],
                    confidence=self.trackingResult.confidence[i:i+1],
                    class_id=self.trackingResult.class_id[i:i+1],
                    tracker_id=self.trackingResult.tracker_id[i:i+1]
                )
                vehicle.set_solo_detection(solo_detection)
        return vehicle

class predictors:
    def __init__(self):
        self.licensePlate_pred = licensePlate_predictor()
        self.vehicle_pred = vehicle_predictor()
    
def _display_rows_on_frame(vehicles_in_interval, frame):
    y_offset = 120  # Starting y-coordinate for displaying text
    for i, vehicle in enumerate(vehicles_in_interval):
        if vehicle.licenseCode != None:
            text = f"{vehicle.detection_time}: {vehicle.licenseCode}"
            frame = cv2.putText(frame, text, (15, y_offset + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def read_known_licensePlates_from_csv_file(file_path): #TODO: Güncelle
    known_licensePlate_indexes = []
    known_licensePlate_licenseCodes = []
    known_licensePlate_encodings = []

    with open(file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            known_licensePlate_indexes.append(int(row['Index']))
            known_licensePlate_licenseCodes.append(row['licenseCode'])
            known_licensePlate_encodings.append([float(val) for val in row['Encoding'][1:-1].split()])

    return known_licensePlate_indexes, known_licensePlate_licenseCodes, known_licensePlate_encodings