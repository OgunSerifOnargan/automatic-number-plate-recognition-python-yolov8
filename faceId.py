import cv2
from classes.face_proposals import faceProposal
from classes.predict import face_predictor, read_known_faces_from_csv_file
from classes.result_person_info import result_person_info
from services.db_utils import append_item_to_json
from services.utils import append_string_to_csv, convert_xywh_to_xyxy, rect_to_xyxy
import time

def faceId(stop_event, faceDet_to_faceId_queue, faceId_to_faceDet_queue, post_queue, model_name, camId): 
    face_pred = face_predictor()
    recognized_trackerIds = []
    print(face_pred.known_face_names)
    while not stop_event.is_set():
        if not faceDet_to_faceId_queue.empty():
            defaultFrame, trackerId, person, refresh_needed = faceDet_to_faceId_queue.get()
            if refresh_needed:
                face_pred.known_face_indexes, face_pred.known_face_names, face_pred.known_face_encodings = read_known_faces_from_csv_file("known_faces.csv")
            #person = people[trackerId]
            if not person.face.isFaceIdentifiedProperly and trackerId not in recognized_trackerIds:
                st = time.time()
                #predict face
                bbox_face_proposals = face_pred.predict_face(model_name, person)
                if len(bbox_face_proposals) == 1: #For DLIB and deepface_ssd
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
                            person.face.bbox_dlib = [bbox_face_proposal]
                            #calculate and set img
                            person.face.faceProposal.crop_and_set_img_faceProposal(defaultFrame)
                        if model_name in ["yolo", "ultralight", "deepface_ssd"]:
                            if model_name == "deepface_ssd":
                                person = convert_xywh_to_xyxy(bbox_face_proposal, person)
                            else:
                                person.face.faceProposal.yolo_bbox = bbox_face_proposal.tolist()

                            person.face.faceProposal.yolo_to_top_right_bottom_left()
                            person.face.faceProposal.set_bbox_defaultFrame_yolo(person.bbox)
                            person.face.faceProposal.crop_and_set_img_faceProposal_yolo(defaultFrame) #içerde .img e ekliyor.
                            person.face.img = person.face.faceProposal.img
                        if person.face.faceProposal.img.size>5000:  #!!!IMPORTANT PARAMETER: adjust min & max face size from here!!!
                            cv2.imshow("face_cropped", person.face.faceProposal.img)
                            cv2.waitKey(1)
                            #log write
                            append_string_to_csv(f"Face is detected. tracker_id: {trackerId}", 'log.csv')
                            #convert img to dlib.face_descriptor() format
                            person = face_pred.identify_face(person, model_name, 0.50)
                            #Stop point of face prediction 
                            #log write
                            append_string_to_csv(f'Face has been read: \n{trackerId} : {person.face.faceProposal.name} ', 'log.csv')
                            #search 3 consecutive name return
                            person.face.face_finalizer.pop(0)
                            person.face.face_finalizer.append(person.face.faceProposal.name)
                            if person.face.face_finalizer[0] == person.face.face_finalizer[1] != "":
                                #log write and print new written name
                                print(f'{person.face.faceProposal.name}    NEW FACE IS FOUND!!!')
                                recognized_trackerIds.append(trackerId)
                                #set final variables into objects' attributes
                                person.set_findings()
                                info = result_person_info(person, camId)
                                post_queue.put(info)
                            faceId_to_faceDet_queue.put([trackerId, 
                                                        person.face.face_finalizer,         
                                                        person.face.isFaceIdentifiedProperly,
                                                        person.face.identification_time,
                                                        person.name,
                                                        person.face.name,
                                                        person.face.img,
                                                        person.face.encodedVector])
                et = time.time()
                #print(f'faceId: {et-st}')


