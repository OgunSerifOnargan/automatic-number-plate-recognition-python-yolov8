from ultralytics import YOLO
import supervision as sv
import numpy as np
import face_recognition
import os
import cv2
from services.utils import get_objects_within_time_interval
from ultralight_face_detector.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from ultralight_face_detector.vision.ssd.config.fd_config import define_img_size
from supervision.detection.core import Detections
from deepface import DeepFace
class face_predictor:
    def __init__(self):
        #YOLO
        self.face_detector = YOLO("/Users/onarganogun/Downloads/best.pt")
        self.yoloResult = None
        #Ultralight
        self.lightweight_predictor = load_lightweight_model()

        #Predictor: Face
        self.name = None
        self.known_face_names, self.known_face_encodings = person_photo_registration("known_faces")

    def predict_face_dlib(self, img_person_body):
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
            result = self.face_detector(img_person_body, verbose=False)
            face_locations = result[0].boxes.xyxy
            return face_locations
        else:
            return None
        
    def predict_face_deepface_SSD(self, img_person_body):
        try:
            if not img_person_body.shape[1] == 0:
                obj = DeepFace.extract_faces(img_path = img_person_body, detector_backend = "ssd", align=True)
        except:
            obj = []
        return obj
    
    def predict_face(self, model_name, person):
        if model_name == "yolo":
            bbox_face_proposals = self.predict_face_yolo(person.img)
        if model_name == "dlib":
            bbox_face_proposals = self.predict_face_dlib(person.img)
        if model_name == "ultralight":
            bbox_face_proposals = self.predict_face_ultralight(person.img)
        if model_name == "deepface_ssd":
            bbox_face_proposals = self.predict_face_deepface_SSD(person.img)
        return bbox_face_proposals
    
    def identify_face(self, person, model_name):
        #encode the face
        if model_name in ["yolo", "ultralight", "deepface_ssd"]:
            person.face.faceProposal.encodedVector = np.array(face_recognition.face_encodings(np.ascontiguousarray(person.img[:, :, ::-1]), 
                                                                                            [person.face.faceProposal.dlib_bbox]))
        if model_name == "dlib":
            person.face.faceProposal.encodedVector = face_recognition.face_encodings(np.ascontiguousarray(person.img[:, :, ::-1]), person.face.faceProposal.bbox_dlib)
        #get binary list of matches according to the constraints
        matches = face_recognition.compare_faces(self.known_face_encodings, np.array(person.face.faceProposal.encodedVector))
        person.face.faceProposal.name = "Unknown"
        #calculate face distances between known_faces and our img
        face_distances = face_recognition.face_distance(self.known_face_encodings, np.array(person.face.faceProposal.encodedVector))
        #get the name of best match
        best_match_index = np.argmin(face_distances)
#        print(face_distances[best_match_index])
        if matches[best_match_index] and face_distances[best_match_index]<=0.60:
            person.face.faceProposal.name = self.known_face_names[best_match_index]
        return person
    
class person_predictor:
    def __init__(self):
        #Person Detector
        self.person_detector = YOLO('models/yolov8n.pt')
        self.person_list = [0]
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        #Predictor: Tracking
        self.tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
        #Results: Tracking
        self.trackingResult = None
        self.modified_solo_detection_for_lineCounter = None
        self.trackingResult_labels = None
        self.solo_detections_dict = {}
        self.person_annotated_frame = None

    def predict_person(self, frame):
            self.yoloResult = self.person_detector.predict(frame, verbose=False, device="mps")[0]
            detections_tracking = sv.Detections.from_ultralytics(self.yoloResult)
            self.trackingResult = detections_tracking[np.isin(detections_tracking.class_id, self.person_list)]
            self.trackingResult = self.tracker.update_with_detections(self.trackingResult)

    def assign_final_name_for_display(self, people):
            self.trackingResult_labels = []
            for confidence, class_id, tracker_id in zip(self.trackingResult.confidence, self.trackingResult.class_id, self.trackingResult.tracker_id):
                if tracker_id in people:
                    if people[tracker_id].face.name != None:
                        self.trackingResult_labels.append(f"#{tracker_id} {people[tracker_id].face.name} {confidence:0.2f}")
                    else:
                        self.trackingResult_labels.append(f"#{tracker_id} {self.person_detector.model.names[class_id]} {confidence:0.2f}")
                else:
                    self.trackingResult_labels.append(f"#{tracker_id} {self.person_detector.model.names[class_id]} {confidence:0.2f}")
            
    def annotate_people(self, frame):
        self.person_annotated_frame = self.box_annotator.annotate(scene=frame, detections=self.trackingResult, labels=self.trackingResult_labels)

    def display_results(self, display_queue, frame, people):
        #Log Display
        people_in_time_interval = get_objects_within_time_interval(people, 30)
        # Display rows on top right corner of the frame
        display_frame = _display_rows_on_frame(people_in_time_interval, frame)
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
    
    def separate_detections(self, person, trackerId):
        for i in range(len(self.trackingResult.xyxy)):
            if self.trackingResult.tracker_id[i:i+1][0] == trackerId:
                solo_detection = Detections(
                    xyxy=self.trackingResult.xyxy[i:i+1],
                    confidence=self.trackingResult.confidence[i:i+1],
                    class_id=self.trackingResult.class_id[i:i+1],
                    tracker_id=self.trackingResult.tracker_id[i:i+1]
                )
                person.set_solo_detection(solo_detection)
        return person
class predictors:
    def __init__(self):
        self.face_pred = face_predictor()
        self.person_pred = person_predictor()


def load_lightweight_model():
    define_img_size(640)
    label_path = "ultralight_face_detector/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True)
    lightweight_predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1500)
    net.load("ultralight_face_detector/models/pretrained/version-RFB-320.pth")
    return lightweight_predictor

def person_photo_registration(folder_path):
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
    
def _display_rows_on_frame(people_in_interval, frame):
    y_offset = 120  # Starting y-coordinate for displaying text
    for i, person in enumerate(people_in_interval):
        if person.name != None:
            text = f"{person.detection_time}: {person.name}"
            frame = cv2.putText(frame, text, (15, y_offset + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    return frame