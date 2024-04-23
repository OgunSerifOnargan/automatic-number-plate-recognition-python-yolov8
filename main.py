from classes.face_proposals import faceProposal
from classes.motion_detector import motion_detection
from classes.result_person_info import result_person_info
from services.utils import get_last_row_info, initialize_people, is_time_outside_interval, rect_to_xyxy, update_people_img_bbox_info, append_string_to_csv
import face_recognition
import numpy as np
from classes.predict import predictors
import cv2, time

append_string_to_csv("licenseCode Recognizer has been started...", 'log.csv')

def main(stop_event, display_queue, faceRec_queue):
    st_ins_fecRec = time.time()
    # load predictors
    predictor = predictors()
    #get known_faces and their encodings
    known_face_names, known_face_encodings = predictor.person_photo_registration("known_faces")
    print(known_face_names)
    #load motion_detector
    motion_detector = motion_detection()

    trackerId = None
    people = {}
    frame_count = 0
    et_ins_faceRec = time.time() 
    print(f'faceRec installation is done in {et_ins_faceRec-st_ins_fecRec}')
    while not stop_event.is_set():

        if not faceRec_queue.empty():
            frame = faceRec_queue.get()
            frame_count +=1
            #motion detection algorithm
            if motion_detector.motion_detected == False:
                st_MD = time.time()
                firstFrame, gray = motion_detector.first_frame_preparer(frame)
                motion_detector.motion_checker(firstFrame, gray)
                et_MD = time.time()
                print(f'MD works in {et_MD-st_MD}')
            if motion_detector.motion_detected and frame_count%10 == 0:
                st_pPerson = time.time()
                # body detection and tracking
                predictor.predict_person(frame, people)
                et_pPerson = time.time()
                print(f'Person detection in {et_pPerson-st_pPerson}')
                #display frame
                st_display = time.time()
                predictor.display_results(display_queue, frame)
                et_display = time.time()
                print(f'Display function works in {et_display-st_display}')
                #raw frame copy
                defaultFrame = frame.copy()
                #display function
                #check if tracked obj is identified
                #crop body from frame
                st_crop_objects = time.time()
                cropped_images_info = predictor.crop_objects(defaultFrame)
                et_crop_objects = time.time()
                print(f'predictor.crop_objects works in {et_crop_objects-st_crop_objects}')
                if len(cropped_images_info) != 0:
                    for trackerId, [img_person, bbox_person] in cropped_images_info.items():
                        if img_person.size>0:
                            #update new frame's incomings
                            st_register_objs = time.time()
                            if not trackerId in people:
                                people = initialize_people(people, trackerId, img_person, bbox_person)
                            else:
                                people = update_people_img_bbox_info(people, trackerId, img_person, bbox_person)

                            # cv2.imshow("img_person", img_person)
                            # cv2.waitKey(1)
                            
                            #get updated person
                            person = people[trackerId]
                            et_register_objs = time.time()
                            print(f'Person-trackingId Registerations works in {et_register_objs-st_register_objs}')
                            if not person.face.isFaceIdentifiedProperly:
                                st_face_detection = time.time()
                                person.img = img_person
                                person.bbox = bbox_person
                                #predict face
                                bbox_face_proposals = predictor.predict_face(img_person)
                                et_face_detection = time.time()
                                print(f'face_detection works in {et_face_detection-st_face_detection}')
                                if bbox_face_proposals:
                                    for bbox_face_proposal in bbox_face_proposals:
                                        st_register_faceProposal = time.time()
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
                                        et_register_faceProposal = time.time()
                                        print(f'faceProposal installation works in {et_register_faceProposal-st_register_faceProposal}')

                                        if person.face.faceProposal.img.size>13000:  #!!!IMPORTANT PARAMETER: adjust min & max face size from here!!!
                                            # cv2.imshow("face_cropped", person.face.faceProposal.img)
                                            # cv2.waitKey(1)
                                            #log write
                                            append_string_to_csv(f"Face is detected. tracker_id: {trackerId}", 'log.csv')
                                            #convert img to dlib.face_descriptor() format
                                            st_face_identification = time.time()
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
                                            et_face_identification = time.time()
                                            print(f'face_identification works in {et_face_identification-st_face_identification}')
                                            #Stop point of face prediction 
                                            if not person.face.isFaceIdentifiedProperly:
                                                #log write
                                                st_post_processing = time.time()
                                                append_string_to_csv(f'Face has been read: \n{trackerId} : {person.face.faceProposal.name} ', 'log.csv')
                                                #search 3 consecutive name return
                                                person.face.face_finalizer.pop(0)
                                                person.face.face_finalizer.append(person.face.faceProposal.name)

                                                if person.face.face_finalizer[0] == person.face.face_finalizer[1] == person.face.face_finalizer[2] != "Unknown":
                                                    
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
                                                elif person.face.face_finalizer[0] == person.face.face_finalizer[1] == person.face.face_finalizer[2] == "Unknown":
                                                    person.face.face_finalizer = ["", "", ""]
                                                    person.face.unknown_count+=1
                                                    if person.face.unknown_count >= 3:
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
                                                        # info.send_post_request('PostCRFormAnswersPublic')
                                                        # info.send_post_request('PostCRFormAnswersPublicFileUpload')
                                                et_post_processing = time.time()
                                                print(f'Post-Processing and post request work in {et_post_processing-st_post_processing}')

                                                    
                #if body cant be found by model, then return to motion detection algorithm
                else:
                    motion_detector.set_motion_detected(False)
                                                    

cv2.destroyAllWindows()
# write results
append_string_to_csv("license Code Recognition system has been shut down.", 'log.csv')

