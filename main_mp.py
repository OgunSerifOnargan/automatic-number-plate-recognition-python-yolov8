#TODO: Fonksiyon süre testlerini yap ve MP'ye alabileceklerini al
#TODO: Piyasadaki optimize person modellerini araştır Şuan yolov8n kullanıyoruz.
#TODO: DB'ye hangi bilgileri göndereceğiz
#TODO: DB'den veri gelmeye başlayınca durma koşulunu manipüle et
#TODO: DB'ye SSMS üzrinden nasıl erişilebileceğini ve tablo oluşturulacağını öğren.
#TODO: DB'de girdi çıktı yapısını kur

import multiprocessing.managers
import multiprocessing, time
from classes.result_person_info import post_results_MP
from services.db_utils import create_json, reassign_tracker_ids
from services.displayServices import display_frames
from services.frameCollectionServices import collect_frames
from faceDet import faceDet
from services.utils import append_string_to_csv
import supervision as sv
from tools.coords_getter_v3 import get_coords
from faceId import faceId

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    people = manager.dict()
    faceRec_queue = multiprocessing.Queue(maxsize=1)
    faceDet_to_faceId_queue = multiprocessing.Queue()
    faceId_to_faceDet_queue = multiprocessing.Queue()
    display_queue = multiprocessing.Queue(maxsize=1)
    post_queue = multiprocessing.Queue()
#    recording_queue = multiprocessing.Queue(maxsize=1000)
    stop_event = multiprocessing.Event()
    recorder_option = 0 #"rtsp://192.168.1.105"
    display_option = 1
    mode_option = 2
    time_test = 0
    model_name = "deepface_ssd"
    append_string_to_csv("person Location Checker has been started...", 'log.csv')
    create_json("db_json")
    reassign_tracker_ids("db_json")


    lines = get_coords(recorder_option, lineCounter=True)
    lines_sv = []
    for i, line in enumerate(lines):
        LINE_START = sv.Point(line[0][0], line[0][1])
        LINE_END = sv.Point(line[1][0], line[1][1])
        lines_sv.append([LINE_START, LINE_END])

    frame_collector_process = multiprocessing.Process(target=collect_frames, 
                                                        args=(stop_event, faceRec_queue, recorder_option))
    faceDet_process = multiprocessing.Process(target=faceDet,
                                                args=(stop_event, people, faceRec_queue, faceDet_to_faceId_queue, faceId_to_faceDet_queue, display_queue, lines_sv))
    faceId_process = multiprocessing.Process(target=faceId,
                                                args=(stop_event, faceDet_to_faceId_queue, faceId_to_faceDet_queue, post_queue, model_name))    
    
    display_process = multiprocessing.Process(target= display_frames,
                                                args=(stop_event, display_queue))
    post_process = multiprocessing.Process(target=post_results_MP,
                                           args=(stop_event, post_queue))
    frame_collector_process.start()
    faceDet_process.start()
    faceId_process.start()
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
        faceDet_process.join()
        faceId_process.join()
        post_process.join()
        if display_option == 1: 
            display_process.join()

    #endregion

