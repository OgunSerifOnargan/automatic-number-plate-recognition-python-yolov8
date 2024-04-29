#TODO: Fonksiyon süre testlerini yap ve MP'ye alabileceklerini al
#TODO: Piyasadaki optimize person modellerini araştır Şuan yolov8n kullanıyoruz.
#TODO: DB'ye hangi bilgileri göndereceğiz
#TODO: DB'den veri gelmeye başlayınca durma koşulunu manipüle et
#TODO: DB'ye SSMS üzrinden nasıl erişilebileceğini ve tablo oluşturulacağını öğren.
#TODO: DB'de girdi çıktı yapısını kur

import multiprocessing, time
from classes.result_person_info import post_results_MP
from services.db_utils import create_json, reassign_tracker_ids
from services.displayServices import display_frames
from services.frameCollectionServices import collect_frames
from main import main
from services.recordingServices import record_frames
from services.utils import append_string_to_csv
import supervision as sv
from tools.coords_getter_v3 import get_coords

if __name__ == '__main__':
    faceRec_queue = multiprocessing.Queue(maxsize=1)
    display_queue = multiprocessing.Queue(maxsize=1)
    post_queue = multiprocessing.Queue()
#    recording_queue = multiprocessing.Queue(maxsize=1000)
    stop_event = multiprocessing.Event()
    recorder_option = 0 #"rtsp://192.168.1.103"
    display_option = 1
    mode_option = 2
    time_test = 0
    model_name = "yolo"
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
    face_rec_process = multiprocessing.Process(target=main,
                                                args=(stop_event, model_name, lines_sv, display_queue, faceRec_queue, post_queue))
    display_process = multiprocessing.Process(target= display_frames,
                                                args=(stop_event, display_queue))
    post_process = multiprocessing.Process(target=post_results_MP,
                                           args=(stop_event, post_queue))
    frame_collector_process.start()
    face_rec_process.start()
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
        face_rec_process.join()
        post_process.join()
        if display_option == 1: 
            display_process.join()

    #endregion

