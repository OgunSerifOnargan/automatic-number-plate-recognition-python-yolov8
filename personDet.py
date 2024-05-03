from classes.motion_detector import motion_detection
from classes.predict import person_predictor
from services.utils import initialize_people, update_faceId_results, append_string_to_csv
import cv2, time

append_string_to_csv("Face Recognizer has been started...", 'log.csv')
def personDet(stop_event, people, faceRec_queue, faceDet_to_faceId_queue, faceId_to_faceDet_queue, display_queue, lines_sv):
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
                st = time.time()
            # body detection and tracking
                person_pred.predict_person(frame)
                #crop body from frame
                cropped_images_info = person_pred.crop_objects(defaultFrame)
                #if body cant be found by model, then return to motion detection algorithm
                if len(cropped_images_info) != 0:
                    motion_detector.last_motion_detection_time = time.time()
                    for trackerId, [img_person, bbox_person] in cropped_images_info.items():
                        if img_person.size>0:
                            #update new frame's incomings
                            if not trackerId in people:
                                initialize_people(people, trackerId, img_person, bbox_person, lines_sv)
                                person = people[trackerId]
                            else:
                                person = people[trackerId]
                                person.update_person_img_bbox_info(trackerId, img_person, bbox_person)
      
                            person = person_pred.separate_detections(person, trackerId)
                            #get updated person
                            frame = person.modify_solo_detection_for_lineCounter(frame)
                            frame = person.update_lineCounter(frame)
                            person.check_where_person_is()
                            person_pred.assign_final_name_for_display(people)
                            person_pred.display_results(display_queue, frame, people)
                            people[trackerId] = person
                            faceDet_to_faceId_queue.put([defaultFrame, trackerId, person])

                            et = time.time()
                            print(f'personDet: {et-st}')
                elif len(cropped_images_info) == 0 and (time.time() - motion_detector.last_motion_detection_time > 5):
                    motion_detector.set_motion_detected(False)
                    print("Camera turn off")
            while not faceId_to_faceDet_queue.empty():
                update_elements = faceId_to_faceDet_queue.get()
                update_faceId_results(update_elements, people)
cv2.destroyAllWindows()
# write results
append_string_to_csv("license Code Recognition system has been shut down.", 'log.csv')

