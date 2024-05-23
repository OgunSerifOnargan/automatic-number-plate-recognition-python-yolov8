import multiprocessing.managers
import multiprocessing, time
from classes.result_vehicle_info import post_results_MP
from services.db_utils import create_json, reassign_tracker_ids
from services.displayServices import display_frames
from services.frameCollectionServices import collect_frames
from vehicleDet import vehicleDet, vehicleDet
from services.utils import append_string_to_csv, get_video_properties
import supervision as sv
from tools.coords_getter_v3 import get_coords
from plateId import plateId
import cv2

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    rawFrame_queue = multiprocessing.Queue()
    licensePlateDet_to_plateId_queue = multiprocessing.Queue(maxsize=1)
    plateId_to_licensePlateDet_queue = multiprocessing.Queue()
    info_transfer_queue = multiprocessing.Queue()
    display_queue = multiprocessing.Queue(maxsize=1)
    post_queue = multiprocessing.Queue()
#    recording_queue = multiprocessing.Queue(maxsize=1000)
    stop_event = multiprocessing.Event()
    recorder_option ="/Users/onarganogun/Downloads/y2mate.com - Güvenlik Kamerasıyla Plaka Okunur mu  HaikonHikvision_1080p.mp4"
    fps, width, height = get_video_properties(recorder_option)
    frame_size = (width, height)
    output_filename = "/Users/onarganogun/Downloads/plateRec.mp4"
    CAM_ID = 1
    display_option = 1
    mode_option = 2
    time_test = 0
    model_name = "yolo"
    append_string_to_csv("vehicle Location Checker has been started...", 'log.csv')
    create_json("db_json")
    reassign_tracker_ids("db_json")

    lines = get_coords(recorder_option, lineCounter=True)
    cv2.destroyAllWindows()
    lines_sv = []
    for i, line in enumerate(lines):
        LINE_START = sv.Point(line[0][0], line[0][1])
        LINE_END = sv.Point(line[1][0], line[1][1])
        lines_sv.append([LINE_START, LINE_END])

    frame_collector_process = multiprocessing.Process(target=collect_frames, 
                                                        args=(stop_event, rawFrame_queue, recorder_option))
    vehicleDet_process = multiprocessing.Process(target=vehicleDet,
                                                args=(stop_event, rawFrame_queue, licensePlateDet_to_plateId_queue, plateId_to_licensePlateDet_queue, display_queue, post_queue, info_transfer_queue, lines_sv, CAM_ID))
    plateId_process = multiprocessing.Process(target=plateId,
                                                args=(stop_event, licensePlateDet_to_plateId_queue, plateId_to_licensePlateDet_queue, post_queue, info_transfer_queue, model_name, CAM_ID))    
    
    display_process = multiprocessing.Process(target= display_frames,
                                                args=(stop_event, display_queue, output_filename, fps, frame_size))
    post_process = multiprocessing.Process(target=post_results_MP,
                                           args=(stop_event, post_queue))
    frame_collector_process.start()
    vehicleDet_process.start()
    plateId_process.start()
    post_process.start()
    if display_option == 1:
        display_process.start()

    #region Stop Condition
    try:
        while True:
            time.sleep(0.0000000001)
        
    except KeyboardInterrupt:
        stop_event.set()
        frame_collector_process.join()
        vehicleDet_process.join()
        plateId_process.join()
        post_process.join()
        if display_option == 1: 
            display_process.join()

    #endregion
