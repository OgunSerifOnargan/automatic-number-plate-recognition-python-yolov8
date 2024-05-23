from classes.motion_detector import motion_detection
from classes.predict import vehicle_predictor
from services.utils import vehicle_cleaner_accordingtoTime, initialize_vehicles, update_licensePlateId_results, append_string_to_csv
import cv2, time
from datetime import datetime

append_string_to_csv("licensePlate Recognizer has been started...", 'log.csv')
def vehicleDet(stop_event, rawFrame_queue, vehicleDet_to_licensePlateId_queue, licensePlateId_to_vehicleDet_queue, display_queue, post_queue, post_transfer_queue, lines_sv, camId):
    # load predictors and motion_detector
    vehicles = {}
    vehicle_pred = vehicle_predictor()
    motion_detector = motion_detection()
    #initialize dummy variables
    frame = None
    trackerId = None
    refresh_needed = False
    frame_count = 0

    while not stop_event.is_set():
        if not rawFrame_queue.empty():
            recognized_trackerIds_from_unidentified = []
            motion_detector.previous_frame = frame
            frame = rawFrame_queue.get()
            frame_count +=1
            if frame_count % 100 == 0:
                vehicle_cleaner_accordingtoTime(vehicles, 5)
            #raw frame copy
            defaultFrame = frame.copy()
            #motion detection algorithm
            if not motion_detector.motion_detected and motion_detector.previous_frame is not None:
                motion_detector.motion_checker(motion_detector.previous_frame, defaultFrame)
            if motion_detector.motion_detected: #and frame_count%10 == 0:
                st = time.time()
            # body detection and tracking
                vehicle_pred.predict_vehicle(frame)
                #crop body from frame
                cropped_images_info = vehicle_pred.crop_objects(defaultFrame)
                #if body cant be found by model, then return to motion detection algorithm
                if len(cropped_images_info) != 0:
                    motion_detector.last_motion_detection_time = time.time()

                    for trackerId, [img_vehicle, bbox_vehicle] in cropped_images_info.items():
                        if img_vehicle.size>0:
                            #update new frame's incomings
                            if not trackerId in vehicles:
                                initialize_vehicles(vehicles, trackerId, img_vehicle, bbox_vehicle, lines_sv)
                                vehicle = vehicles[trackerId]
                            else:
                                vehicle = vehicles[trackerId]
                                vehicle.update_vehicle_img_bbox_info(img_vehicle, bbox_vehicle)

                            vehicle = vehicle_pred.separate_detections(vehicle, trackerId)
                            #get updated vehicle
                            frame = vehicle.modify_solo_detection_for_lineCounter(frame)
                            frame = vehicle.update_lineCounter(frame)
                            vehicle.check_where_vehicle_is()
                            if vehicle.location_state == 0:
                                while not post_transfer_queue.empty():
                                    info = post_transfer_queue.get()
                                    print(info.trackerId)
                                    cv2.imshow("Sent vehicle", info.body_img)
                                    cv2.waitKey(1)
                                    if info.licenseCode is not None:
                                        cv2.imshow("Sent Plate", info.licensePlate_img)
                                        cv2.waitKey(1)
                                        print(info.licenseCode)
                                    else:
                                        print("Unidentified Vehicle is just entered...")
                                    #post_queue.put(info)
                            vehicle_pred.assign_final_licenseCode_for_display(vehicles)
                            vehicle_pred.display_results(display_queue, frame, vehicles)
                            if vehicle.licensePlate.islicensePlateIdentifiedProperly == False:
                                vehicleDet_to_licensePlateId_queue.put([defaultFrame, trackerId, vehicle])

                            et = time.time()
                            #print(f'vehicleDet: {et-st}')
                elif len(cropped_images_info) == 0 and (time.time() - motion_detector.last_motion_detection_time > 50):
                    motion_detector.set_motion_detected(False)
                    print("Camera turn off")
            while not licensePlateId_to_vehicleDet_queue.empty():
                update_elements = licensePlateId_to_vehicleDet_queue.get()
                vehicles = update_licensePlateId_results(update_elements, vehicles)

cv2.destroyAllWindows()
# write results
append_string_to_csv("license Code Recognition system has been shut down.", 'log.csv')
