from ultralytics import YOLO
import supervision as sv
import numpy as np
import face_recognition
import os
import cv2
from datetime import datetime
import csv
from ultralight_face_detector.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from ultralight_face_detector.vision.ssd.config.fd_config import define_img_size
class predictors:
    def __init__(self):
    #Predictor: Body
    #Person Detector
        self.person_detector = YOLO('models/yolov8n.pt')
        self.person_list = [0]
    #Face Detector
        #YOLO
        self.face_detector = YOLO("/Users/onarganogun/Downloads/best.pt")
        #Ultralight
        self.lightweight_predictor = load_lightweight_model()

        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        #Results: Body
        self.yoloResult = None

        #Predictor: Tracking
        #load tracker tools
        self.tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
        #Results: Tracking
        self.trackingResult = None
        self.trackingResult_labels = None
        self.person_annotated_frame = None
        #Predictor: Face
        self.name = None

    def predict_person(self, frame, people):
        self.yoloResult = self.person_detector.predict(frame, verbose=False, device="mps")[0]
        detections_tracking = sv.Detections.from_ultralytics(self.yoloResult)
        self.trackingResult = detections_tracking[np.isin(detections_tracking.class_id, self.person_list)]
        self.trackingResult = self.tracker.update_with_detections(self.trackingResult)
        self.trackingResult_labels = []
        for confidence, class_id, tracker_id in zip(self.trackingResult.confidence, self.trackingResult.class_id, self.trackingResult.tracker_id):
            if tracker_id in people:
                if people[tracker_id].face.name != None:
                    self.trackingResult_labels.append(f"#{tracker_id} {people[tracker_id].face.name} {confidence:0.2f}")
                else:
                    self.trackingResult_labels.append(f"#{tracker_id} {self.person_detector.model.names[class_id]} {confidence:0.2f}")
            else:
                self.trackingResult_labels.append(f"#{tracker_id} {self.person_detector.model.names[class_id]} {confidence:0.2f}")
    
    def predict_face(self, img_person_body):
        if not img_person_body.shape[1] == 0:
            face_locations = face_recognition.face_locations(img_person_body, model="hog")
            return face_locations
        else:
            return None
        
    def predict_face_ultralight(self, img_person_body):
        if not img_person_body.shape[1] == 0:
            boxes, labels, probs = self.lightweight_predictor.predict(img_person_body, 1, 0.70)
        return boxes
    
    def predict_face_yolo(self, img_person_body):
        if not img_person_body.shape[1] == 0:
            result = self.face_detector(img_person_body, verbose=False, device="mps")
            face_locations = result[0].boxes.xyxy
            return face_locations
        else:
            return None
        
    def annotate_people(self, frame):
        self.person_annotated_frame = self.box_annotator.annotate(scene=frame, detections=self.trackingResult, labels=self.trackingResult_labels)

    def person_photo_registration(self, folder_path):
        known_face_encodings = []
        known_face_names = []

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):
                # Load the image
                image_path = os.path.join(folder_path, filename)
                person_image = face_recognition.load_image_file(image_path)

                # Extract face encoding
                face_encoding = face_recognition.face_encodings(person_image)
                
                # Ensure that the image contains exactly one face
                if len(face_encoding) == 1:
                    known_face_encodings.append(face_encoding[0])
                    
                    # Extract the name from the filename (excluding the extension)
                    known_face_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"Skipping {filename} as it doesn't contain exactly one face.")
        return known_face_names, known_face_encodings
    
    def display_results(self, display_queue, frame):
        #Log Display
        rows_in_interval = _get_rows_in_interval("Face_records.csv")
        # Display rows on top right corner of the frame
        display_frame = _display_rows_on_frame(rows_in_interval, frame)
        # person Annotator
        self.annotate_people(display_frame)
        #Face Image Displayer
        display_queue.put(self.person_annotated_frame)
    
    def crop_objects(self, image):
        cropped_images_info = {}
        for box, tracker_id in zip(self.trackingResult.xyxy, self.trackingResult.tracker_id):
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]

            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_bbox = [x_min, y_min, x_max, y_max]

            cropped_images_info[tracker_id] = [cropped_image, cropped_bbox]
        return cropped_images_info
    
    def person_photo_registration_yolo(self, folder_path):
        known_face_encodings = []
        known_face_names = []

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):
                # Load the image
                image_path = os.path.join(folder_path, filename)
                person_image = face_recognition.load_image_file(image_path)

                # Extract face encoding
                face_encoding = face_recognition.face_encodings(person_image)
                
                # Ensure that the image contains exactly one face
                if len(face_encoding) == 1:
                    known_face_encodings.append(face_encoding[0])
                    
                    # Extract the name from the filename (excluding the extension)
                    known_face_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"Skipping {filename} as it doesn't contain exactly one face.")
        return known_face_names, known_face_encodings
    
    






def load_lightweight_model():
    define_img_size(640)
    label_path = "ultralight_face_detector/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True)
    lightweight_predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1500)
    net.load("ultralight_face_detector/models/pretrained/version-RFB-320.pth")
    return lightweight_predictor

def _display_rows_on_frame(rows, frame):
    y_offset = 120  # Starting y-coordinate for displaying text
    for i, row in enumerate(rows):
        text = f"{row['Time']}: {row['Data']}"
        frame = cv2.putText(frame, text, (15, y_offset + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def _add_image_to_top_right(image1, image2, desired_bbox=None, trackId_foundFace=None):
    # Read the images
    if desired_bbox is None or len(desired_bbox) > 1:
        # Get the dimensions of the images
        height1, width1, _ = image1.shape
        height2, width2, _ = image2.shape

        # Define the position for image2 on image1 (top right corner)
        x_offset = width1 - width2
        y_offset = 0

        # Paste image2 onto image1
        image1[y_offset:y_offset+height2, x_offset:x_offset+width2] = image2
        return image1
    elif trackId_foundFace in desired_bbox:
        desired_bbox = desired_bbox[trackId_foundFace]
                # Get the dimensions of the images
        height1, width1 = desired_bbox[3], desired_bbox[2]
        height2, width2, _ = image2.shape

        # Define the position for image2 on image1 (top right corner)
        x_offset = width1 - width2
        y_offset = height1 - height2

        # Paste image2 onto image1
        image1[y_offset-5:y_offset+height2-5, x_offset-500:x_offset+width2-500] = image2
        return image1
    else:
        return image1 

def _get_rows_in_interval(csv_file):
    if not os.path.isfile(csv_file):
        print(f"File '{csv_file}' does not exist. Creating...")
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Data'])  # Write header to the CSV file
            print(f"File '{csv_file}' created.")
    time1 = datetime.now()
    rows_in_interval = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            time_str = row['Time']
            time2 = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

            difference = (time1 - time2).total_seconds()
            if difference < 60:
                rows_in_interval.append(row)
    return rows_in_interval







