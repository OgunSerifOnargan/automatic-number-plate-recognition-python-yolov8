from classes.face_proposals import faceProposal
from classes.motion_detector import motion_detection
from classes.result_person_info import result_person_info
from classes.predict import person_predictor
from services.db_utils import append_item_to_json, read_json_as_dict
from services.utils import initialize_people, rect_to_xyxy, update_person_img_bbox_info, append_string_to_csv
import cv2

append_string_to_csv("Face Recognizer has been started...", 'log.csv')
def faceDet(stop_event, people, faceRec_queue, faceDet_to_faceId_queue, faceId_to_faceDet_queue, display_queue, lines_sv):
    # load predictors and motion_detector
    person_pred = person_predictor()
    motion_detector = motion_detection()
    #initialize dummy variables
    frame = None
    trackerId = None
    frame_count = 0

    while not stop_event.is_set():
        if not faceRec_queue.empty():
            motion_detector.previous_frame = frame
            frame = faceRec_queue.get()
            frame_count +=1
            #raw frame copy
            defaultFrame = frame.copy()
            #motion detection algorithm
            if not motion_detector.motion_detected and motion_detector.previous_frame is not None:
                motion_detector.motion_checker(motion_detector.previous_frame, defaultFrame)
            if motion_detector.motion_detected: #and frame_count%10 == 0:
            # body detection and tracking
                person_pred.predict_person(frame)
                #display frame
                #display function
                #crop body from frame
                cropped_images_info = person_pred.crop_objects(defaultFrame)
                #if body cant be found by model, then return to motion detection algorithm
                if len(cropped_images_info) != 0:
                    for trackerId, [img_person, bbox_person] in cropped_images_info.items():
                        if img_person.size>0:
                            #update new frame's incomings
                            if not trackerId in people:
                                initialize_people(people, trackerId, img_person, bbox_person, lines_sv)
                                person = people[trackerId]
                            else:
                                person = people[trackerId]
                                update_person_img_bbox_info(person, trackerId, img_person, bbox_person)
      
                            person = person_pred.separate_detections(person, trackerId)
                            #get updated person
                            frame = person.modify_solo_detection_for_lineCounter(frame)
                            frame = person.update_lineCounter(frame)
                            person.check_where_person_is()
                            person_pred.assign_final_name_for_display(people)
                            person_pred.display_results(display_queue, frame, people)
                            people[trackerId] = person
                            faceDet_to_faceId_queue.put([defaultFrame, trackerId, person])
                else:
                    motion_detector.set_motion_detected(False)
                    print("Camera turn off")
            while not faceId_to_faceDet_queue.empty():
                update_elements = faceId_to_faceDet_queue.get()
                update_faceId_results(update_elements, people)
cv2.destroyAllWindows()
# write results
append_string_to_csv("license Code Recognition system has been shut down.", 'log.csv')



def update_faceId_results(update_elements, people):
    trackerId, person_face_faceProposal_encodedVector, person_face_faceProposal_name, person_face_face_finalizer, person_face_isFaceIdentifiedProperly, person_face_identification_time, person_name, person_face_name, person_face_img, person_face_encodedVector, person_face_bbox_defaultFrame = update_elements
    person = people[trackerId]
    if person_face_identification_time is not None:
        person.face.identification_time = person_face_identification_time
    if person_name is not None:
        person.name = person_name
    if person_face_name is not None:
        person.face.name = person_face_name
    if person_face_img is not None:
        person.face.img = person_face_img
    if person_face_encodedVector is not None:
        person.face.encodedVector = person_face_encodedVector
    person.face.face_finalizer = person_face_face_finalizer
    person.face.isFaceIdentifiedProperly = person_face_isFaceIdentifiedProperly
    
    people[trackerId] = person
