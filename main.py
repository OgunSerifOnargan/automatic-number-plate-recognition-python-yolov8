from classes.face_proposals import faceProposal
from classes.motion_detector import motion_detection
from classes.result_person_info import result_person_info
from services.utils import get_last_row_info, initialize_people, is_time_outside_interval, rect_to_xyxy, update_people_img_bbox_info, append_string_to_csv
import face_recognition
import numpy as np
from classes.predict import predictors
import cv2

append_string_to_csv("licenseCode Recognizer has been started...", 'log.csv')

def main(stop_event, display_queue, faceRec_queue):
    # load predictors
    predictor = predictors()
    #get known_faces and their encodings
    known_face_names, known_face_encodings = predictor.person_photo_registration("known_faces")
    print(known_face_names)
    #load motion_detector
    motion_detector = motion_detection()

    img_foundFace = None
    trackerId = None
    people = {}

    while not stop_event.is_set():

        if not faceRec_queue.empty():
            frame = faceRec_queue.get()
            #motion detection algorithm
            if motion_detector.motion_detected == False:
                firstFrame, gray = motion_detector.first_frame_preparer(frame)
                motion_detector.motion_checker(firstFrame, gray)
            if motion_detector.motion_detected:
                # body detection
                predictor.predict_person(frame=frame)
                # tracking algorithm
                predictor.track_person()
                #display frame
                predictor.display_results(display_queue, frame, img_foundFace, trackerId)
                #raw frame copy
                defaultFrame = frame.copy()
                #display function
                #check if tracked obj is identified
                #crop body from frame
                cropped_images_info = predictor.crop_objects(defaultFrame)
                if len(cropped_images_info) != 0:
                    for trackerId, [img_person, bbox_person] in cropped_images_info.items():
                        if img_person.size>0:
                            #update new frame's incomings
                            if not trackerId in people:
                                people = initialize_people(people, trackerId, img_person, bbox_person)
                            else:
                                people = update_people_img_bbox_info(people, trackerId, img_person, bbox_person)

                            # cv2.imshow("img_person", img_person)
                            # cv2.waitKey(1)
                            
                            #get updated person
                            person = people[trackerId]
                            if not person.face.isFaceIdentifiedProperly:
                                person.img = img_person
                                person.bbox = bbox_person
                                #predict face
                                bbox_face_proposals = predictor.predict_face(img_person)
                                if bbox_face_proposals:
                                    for bbox_face_proposal in bbox_face_proposals:
                                        #initialize faceProposal at person.face
                                        person.face.faceProposal = faceProposal()
                                        #calculate and set bbox for face, defaultFrame and dlib  (dlib format is top, right, bottom, left. It's unusual format.)
                                        #body_xyxy
                                        person.face.faceProposal.bbox = rect_to_xyxy(bbox_face_proposal)
                                        #defaultFrame_xyxy
                                        person.face.faceProposal.set_bbox_defaultFrame(person.bbox)
                                        #dlib top, right, bottom, left
                                        person.face.faceProposal.bbox_dlib = [bbox_face_proposal]
                                        #calculate and set img
                                        person.face.faceProposal.crop_and_set_img_faceProposal(frame) #içerde .img e ekliyor.

                                        if person.face.faceProposal.img.size>13000:  #!!!IMPORTANT PARAMETER: adjust min & max face size from here!!!
                                            cv2.imshow("face_cropped", person.face.faceProposal.img)
                                            cv2.waitKey(1)
                                            #log write
                                            append_string_to_csv(f"license Plate is detected. tracker_id: {trackerId}", 'log.csv')
                                            #convert img to dlib.face_descriptor() format
                                            img_person = np.ascontiguousarray(img_person[:, :, ::-1])
                                            #encode the face
                                            person.face.faceProposal.encodedVector = np.array(face_recognition.face_encodings(img_person, [bbox_face_proposal])) #TODO: xyxy formatını düzelt. 2kişi buldugunda sorun yaşamayalım.
                                            #get binary list of matches according to the constraints
                                            matches = face_recognition.compare_faces(known_face_encodings, person.face.faceProposal.encodedVector)
                                            person.face.faceProposal.name = "Unknown"
                                            #NOTE: Commented part which is below causes errors, due to multiple max (constraint: 0.6 face distance)
                                            #NOTE: It can be manipulated to increase both accuracy and FP alarms.

                                            # # If a match was found in known_face_encodings, just use the first one.
                                            # if True in matches:
                                            #     first_match_index = matches.index(True)
                                            #     name = known_face_names[first_match_index]
                                            # Or instead, use the known face with the smallest distance to the new face

                                            #calculate face distances between known_faces and our img
                                            face_distances = face_recognition.face_distance(known_face_encodings, person.face.faceProposal.encodedVector)
                                            #get the name of best match
                                            best_match_index = np.argmin(face_distances)
                                            if matches[best_match_index]:
                                                person.face.faceProposal.name = known_face_names[best_match_index]
                                                print(person.face.faceProposal.name)
                                            #Stop point of face prediction 
                                            if (person.face.isFaceIdentifiedProperly == False) and (person.face.faceProposal.name != "Unknown"):
                                                #log write
                                                append_string_to_csv(f'license Code has been read: \n{trackerId} : {person.face.faceProposal.name} ', 'log.csv')
                                                #search 3 consecutive name return
                                                person.face.face_finalizer.pop(0)
                                                person.face.face_finalizer.append(person.face.faceProposal.name)
                                                if person.face.face_finalizer[0] == person.face.face_finalizer[1] == person.face.face_finalizer[2]:
                                                    
                                                    last_time_recorded, lastFace = get_last_row_info('face_records.csv')
                                                    #stop if the person found in 6 minutes #TODO:This adjustable parameter should be replaced according to inside-outside rules
                                                    if lastFace != person.face.faceProposal.name or is_time_outside_interval(last_time_recorded):
                                                        #log write and print new written name
                                                        print(f'{person.face.faceProposal.name}    NEW FACE IS FOUND!!!')
                                                        append_string_to_csv(f'{person.face.faceProposal.name}', 'face_records.csv')
                                                        #set final variables into objects' attributes
                                                        person.face.isFaceIdentifiedProperly = True
                                                        person.set_findings()
                                                        # API set requests
                                                        info = result_person_info(person)
                                                        info.set_img_base64()
                                                        info.construct_body_info()
                                                        info.construct_body_img()
                                                        info.send_post_request('PostCRFormAnswersPublic')
                                                        info.send_post_request('PostCRFormAnswersPublicFileUpload')
                #if body cant be found by model, then return to motion detection algorithm
                else:
                    motion_detector.set_motion_detected(False)
                                                    

cv2.destroyAllWindows()
# write results
append_string_to_csv("license Code Recognition system has been shut down.", 'log.csv')

