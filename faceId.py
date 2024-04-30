import cv2
from classes.face_proposals import faceProposal
from classes.predict import face_predictor
from classes.result_person_info import result_person_info
from services.db_utils import append_item_to_json
from services.utils import append_string_to_csv, initialize_people, rect_to_xyxy, update_people_img_bbox_info
import time

def faceId(stop_event, people, faceDet_to_faceId_queue, post_queue, model_name): 
    face_pred = face_predictor()
    print(face_pred.known_face_names)
    while not stop_event.is_set():
        if not faceDet_to_faceId_queue.empty():
            defaultFrame, trackerId, person = faceDet_to_faceId_queue.get()
            #person = people[trackerId]
            if not person.face.isFaceIdentifiedProperly:
                #predict face
                if model_name == "yolo":
                    bbox_face_proposals = face_pred.predict_face_yolo(person.img)
                if model_name == "dlib":
                    bbox_face_proposals = face_pred.predict_face(person.img)
                if model_name == "ultralight":
                    bbox_face_proposals = face_pred.predict_face_ultralight(person.img)
                if model_name == "deepface_ssd":
                    st = time.time()
                    bbox_face_proposals = face_pred.predict_face_deepface_SSD(person.img)
                    et = time.time()
                    print(f"deepface_ssd running time is: {et-st}")
                if len(bbox_face_proposals) == 1: #For DLIB
                #if len(bbox_face_proposals.size()):    #FOR YOLO and ultralight
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
                            person = face_pred.identify_face(person)
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
                                    info = result_person_info(person)
                                    post_queue.put(info)
                                    append_item_to_json(trackerId, person, "db_json")
                            people[trackerId] = person
