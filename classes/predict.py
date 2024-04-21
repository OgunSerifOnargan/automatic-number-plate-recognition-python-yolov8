from ultralytics import YOLO
import supervision as sv
import numpy as np
import face_recognition
import os
import cv2
from datetime import datetime
import csv

class predictors:
    def __init__(self):
        #Predictor: Body
        self.person_detector = YOLO('models/yolov8n.pt')
        self.person_list = [0]
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

    def predict_person(self, frame):
        self.yoloResult = self.person_detector.predict(frame, verbose=False)[0]
        detections_tracking = sv.Detections.from_ultralytics(self.yoloResult)
        detections_filtered = detections_tracking[np.isin(detections_tracking.class_id, self.person_list)]
        if detections_filtered.tracker_id != None:
            labels = [
                f"#{tracker_id} {self.person_detector.model.names[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id
                in zip(detections_filtered.confidence, detections_filtered.class_id, detections_filtered.tracker_id)
            ]
        else:
            labels = []
        self.trackingResult = detections_filtered
        self.trackingResult_labels = labels
        return detections_filtered
    
    def track_person(self):
        # tracking algorithm
        self.trackingResult = self.tracker.update_with_detections(self.trackingResult)
    
    def predict_face(self, img_person_body):
        if not img_person_body.shape[1] == 0:
            face_locations = face_recognition.face_locations(img_person_body, model="hog")
            print(face_locations)
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
    

    def display_results(self, display_queue, frame, img_foundFace, trackerId):
        #Log Display
        rows_in_interval = _get_rows_in_interval("Face_records.csv")
        # Display rows on top right corner of the frame
        display_frame = _display_rows_on_frame(rows_in_interval, frame)
        # person Annotator
        self.annotate_people(display_frame)
        #Face Image Displayer
        if img_foundFace is not None:
            self.person_annotated_frame = _add_image_to_top_right(self.person_annotated_frame, img_foundFace, self.trackingResult.xyxy, trackerId)

        display_queue.put(self.person_annotated_frame)
    
    def crop_objects(self, image):
        cropped_images_info = {}
        for box, tracker_id in zip(self.trackingResult.xyxy, self.trackingResult.tracker_id):
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]

            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_bbox = [x_min, y_min, x_max, y_max]

            cropped_images_info[tracker_id] = [cropped_image, cropped_bbox]
        return cropped_images_info



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
