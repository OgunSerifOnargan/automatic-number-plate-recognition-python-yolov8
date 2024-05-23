import cv2
from classes.proposal import proposal
from classes.predict import licensePlate_predictor, read_known_licensePlates_from_csv_file
from classes.result_vehicle_info import result_vehicle_info
from services.skewness_correction import correct_skew
from services.utils import convert_xywh_to_xyxy
import time
from datetime import datetime

def plateId(stop_event, vehicleDet_to_plateId_queue, plateId_to_vehicleDet_queue, post_queue, post_transfer_queue, model_name, camId): 
    info = []
    licensePlate_pred = licensePlate_predictor()
    list_of_detected_Codes = []
    while not stop_event.is_set():
        if not vehicleDet_to_plateId_queue.empty():
            defaultFrame, trackerId, vehicle = vehicleDet_to_plateId_queue.get()
            #if a licenseCode is assigned on vehicleDet, then do not check the vehicle's licensePlate. Update known licensePlates
            if not vehicle.licensePlate.islicensePlateIdentifiedProperly:
                st = time.time()
                vehicle.licensePlate.proposal = proposal()
                #predict licensePlate
                bbox_licensePlate_proposals = licensePlate_pred.predict_licensePlate_yolo(vehicle)
                if len(bbox_licensePlate_proposals) == 1:
                    for bbox_licensePlate_proposal in bbox_licensePlate_proposals:
                        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox_licensePlate_proposal]
                        vehicle.licensePlate.proposal.bbox_xyxy = [x_min, y_min, x_max, y_max]
                        if model_name in ["yolo"]:
                            vehicle.licensePlate.proposal.crop_and_set_img_proposal_xyxy(vehicle.img_body) #içerde .img e ekliyor.
                            vehicle.licensePlate.img = vehicle.licensePlate.proposal.img
                            print(vehicle.licensePlate.proposal.img.size)
                        if vehicle.licensePlate.proposal.img.size>23000:
                            angle, vehicle.img_skewed_plate = correct_skew(vehicle)
                            vehicle.img_skewed_plate_default = vehicle.img_skewed_plate.copy()
                            cv2.imshow("plate", vehicle.img_skewed_plate)
                            cv2.waitKey(1)
                            vehicle = licensePlate_pred.predict_license_number(vehicle)
                            if vehicle.licensePlate.proposal.licenseCode is not None:
                                vehicle.licensePlate.proposal.code_parts_initilizer()
                                vehicle.licensePlate.proposal.turkishlicenseCorrection()
                                #Stop point of licensePlate prediction 
                                #search 3 consecutive licenseCode return
                                if vehicle.licensePlate.proposal.licenseCode != None and vehicle.licensePlate.proposal.licenseCode not in list_of_detected_Codes:
                                    vehicle.licensePlate.licensePlate_finalizer.pop(0)
                                    vehicle.licensePlate.licensePlate_finalizer.append(vehicle.licensePlate.proposal.licenseCode)
                                    if vehicle.licensePlate.licensePlate_finalizer[0] == vehicle.licensePlate.licensePlate_finalizer[1] != "" and len(vehicle.licensePlate.licensePlate_finalizer[0]) >= 7 and len(vehicle.licensePlate.licensePlate_finalizer[0]) <= 8:
                                        print(vehicle.licensePlate.licensePlate_finalizer)
                                        #log write and print new written licenseCode
                                        print(f'{vehicle.licensePlate.proposal.licenseCode}    NEW licensePlate IS FOUND!!!')
                                        #set final variables into objects' attributes

                                        vehicle.identificationTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        vehicle.licenseCode = vehicle.licensePlate.proposal.licenseCode
                                        vehicle.img_vehicle = vehicle.img_body
                                        vehicle.licensePlate.islicensePlateIdentifiedProperly = True
                                        list_of_detected_Codes.append(vehicle.licensePlate.proposal.licenseCode)

                                        info = result_vehicle_info(vehicle, camId)
                                        post_transfer_queue.put(info)

                                    plateId_to_vehicleDet_queue.put([trackerId, 
                                                            vehicle.licensePlate.licensePlate_finalizer,         
                                                            vehicle.licensePlate.islicensePlateIdentifiedProperly,
                                                            vehicle.identificationTime,
                                                            vehicle.licenseCode,
                                                            vehicle.img_skewed_plate_default,
                                                            vehicle.licensePlate.proposal.licenseCode])
                                

                et = time.time()
                #print(f'plateId: {et-st}')


