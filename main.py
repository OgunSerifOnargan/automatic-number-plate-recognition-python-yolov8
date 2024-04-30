from classes.face_proposals import faceProposal
from classes.motion_detector import motion_detection
from classes.result_person_info import result_person_info
from classes.predict import predictors
from services.db_utils import append_item_to_json, read_json_as_dict
from services.utils import initialize_people, rect_to_xyxy, update_people_img_bbox_info, append_string_to_csv
import cv2
import time

append_string_to_csv("licenseCode Recognizer has been started...", 'log.csv')
def main(stop_event, model_name, lines_sv, display_queue, faceRec_queue, post_queue):
    # load predictors and motion_detector
    predictor = predictors()
    motion_detector = motion_detection()
    print(predictor.known_face_names)
    #initialize dummy variables
    frame = None
    trackerId = None
    people = {}
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
                predictor.predict_person(frame, people)
                #display frame
                #display function
                #crop body from frame
                cropped_images_info = predictor.crop_objects(defaultFrame)
                if len(cropped_images_info) != 0:
                    for trackerId, [img_person, bbox_person] in cropped_images_info.items():
                        if img_person.size>0:
                            #update new frame's incomings
                            if not trackerId in people:
                                people = initialize_people(people, trackerId, img_person, bbox_person, lines_sv)
                            else:
                                people = update_people_img_bbox_info(people, trackerId, img_person, bbox_person)
                            people = predictor.separate_detections(people, trackerId)
                            #get updated person
                            person = people[trackerId]
                            frame = person.modify_solo_detection_for_lineCounter(frame, placement="foot")
                            frame = person.update_lineCounter(frame)
                            person.check_where_person_is()
                            predictor.display_results(display_queue, frame, people)
                            if not person.face.isFaceIdentifiedProperly:
                                #predict face
                                if model_name == "yolo":
                                    bbox_face_proposals = predictor.predict_face_yolo(person.img)
                                if model_name == "dlib":
                                    bbox_face_proposals = predictor.predict_face(person.img)
                                if model_name == "ultralight":
                                    bbox_face_proposals = predictor.predict_face_ultralight(person.img)
                                if model_name == "deepface_ssd":
                                    st = time.time()
                                    bbox_face_proposals = predictor.predict_face_deepface_SSD(person.img)
                                    et = time.time()
                                    print(f"deepface_ssd running time is: {et-st}")
                                if len(bbox_face_proposals) == 1: #For deepface_SSD
                                # if len(bbox_face_proposals.size()):    #FOR YOLO and ultralight
                                    for bbox_face_proposal in bbox_face_proposals:
                                        #initialize faceProposal at person.face
                                        person.face.faceProposal = faceProposal()
                                        #calculate and set bbox for face, defaultFrame and dlib  (dlib format is top, right, bottom, left. It's unusual format.)
                                        if model_name == "dlib":
                                            #body_xyxy
                                            person.face.faceProposal.bbox = rect_to_xyxy(bbox_face_proposal)
                                            #defaultFrame_xyxy
                                            person.face.faceProposal.set_bbox_defaultFrame(person.bbox)
                                            #dlib top, right, bottom, left
                                            person.face.faceProposal.bbox_dlib = [bbox_face_proposal]
                                            #calculate and set img
                                            person.face.faceProposal.crop_and_set_img_faceProposal(defaultFrame)

                                        if model_name in ["yolo", "ultralight", "deepface_ssd"]:
                                            if model_name == "deepface_ssd":
                                                bbox_face_proposal['facial_area']
                                                x1 = bbox_face_proposal['facial_area']['x']
                                                y1 = bbox_face_proposal['facial_area']['y']
                                                x2 = x1 + bbox_face_proposal['facial_area']['w']
                                                y2 = y1 + bbox_face_proposal['facial_area']['h']
                                                person.face.faceProposal.yolo_bbox = [x1, y1, x2, y2]
                                            else:
                                                person.face.faceProposal.yolo_bbox = bbox_face_proposal.tolist()

                                            person.face.faceProposal.yolo_to_top_right_bottom_left()
                                            person.face.faceProposal.set_bbox_defaultFrame_yolo(person.bbox)
                                            person.face.faceProposal.crop_and_set_img_faceProposal_yolo(defaultFrame) #içerde .img e ekliyor.
                                        if person.face.faceProposal.img.size>10000:  #!!!IMPORTANT PARAMETER: adjust min & max face size from here!!!
                                            cv2.imshow("face_cropped", person.face.faceProposal.img)
                                            cv2.waitKey(1)
                                            #log write
                                            append_string_to_csv(f"Face is detected. tracker_id: {trackerId}", 'log.csv')
                                            #convert img to dlib.face_descriptor() format
                                            person = predictor.identify_face(person)
                                            #Stop point of face prediction 
                                            #log write
                                            append_string_to_csv(f'Face has been read: \n{trackerId} : {person.face.faceProposal.name} ', 'log.csv')
                                            #search 3 consecutive name return
                                            person.face.face_finalizer.pop(0)
                                            person.face.face_finalizer.append(person.face.faceProposal.name)
                                            if person.face.face_finalizer[0] == person.face.face_finalizer[1] == person.face.face_finalizer[2] != "Unknown":
                                                #log write and print new written name
                                                print(f'{person.face.faceProposal.name}    NEW FACE IS FOUND!!!')
                                                #set final variables into objects' attributes
                                                person.set_findings()
                                            elif person.face.face_finalizer[0] == person.face.face_finalizer[1] == person.face.face_finalizer[2] == "Unknown":
                                                person.face.face_finalizer = ["", "", ""]
                                                person.face.unknown_count+=1
                                                if person.face.unknown_count >= 10:
                                                    print(f'{person.face.faceProposal.name}    NEW FACE IS FOUND!!!')
                                                    #set final variables into objects' attributes
                                                    person.set_findings()

                                            # API set requests
                                            info = result_person_info(person)
                                            post_queue.put(info)
                                            append_item_to_json(trackerId, person, "db_json")
                #if body cant be found by model, then return to motion detection algorithm
                else:
                    motion_detector.set_motion_detected(False)
                    print("Camera turn off")
cv2.destroyAllWindows()
# write results
append_string_to_csv("license Code Recognition system has been shut down.", 'log.csv')
