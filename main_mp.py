#TODO: Fonksiyon süre testlerini yap ve MP'ye alabileceklerini al
#TODO: Piyasadaki optimize person modellerini araştır Şuan yolov8n kullanıyoruz.
#TODO: DB'ye hangi bilgileri göndereceğiz
#TODO: DB'den veri gelmeye başlayınca durma koşulunu manipüle et
#TODO: DB'ye SSMS üzrinden nasıl erişilebileceğini ve tablo oluşturulacağını öğren.
#TODO: DB'de girdi çıktı yapısını kur

import multiprocessing, time
from services.displayServices import display_frames
from services.frameCollectionServices import collect_frames
from main import main
from services.recordingServices import record_frames
from services.utils import append_string_to_csv

if __name__ == '__main__':
    st_ins_main = time.time() 
    faceRec_queue = multiprocessing.Queue(maxsize=1)
    display_queue = multiprocessing.Queue(maxsize=1)
#    recording_queue = multiprocessing.Queue(maxsize=1000)
    stop_event = multiprocessing.Event()

    recorder_option = "rtsp://192.168.1.101"
    display_option = 1
    mode_option = 2
    time_test = 0
    append_string_to_csv("person Location Checker has been started...", 'log.csv')

    frame_collector_process = multiprocessing.Process(target=collect_frames, 
                                                        args=(stop_event, display_queue, faceRec_queue, recorder_option))

    face_rec_process = multiprocessing.Process(target=main,
                                                args=(stop_event, time_test, display_queue, faceRec_queue))
    # recording_process = multiprocessing.Process(target=record_frames,
    #                                             args =(stop_event, recording_queue))
    display_process = multiprocessing.Process(target= display_frames,
                                                args=(stop_event, display_queue))
    

    frame_collector_process.start()
    face_rec_process.start()
    if display_option == 1:
        display_process.start()

    #region Stop Condition
    et_ins_main = time.time() 
    print(f'Main installation is done in {et_ins_main-st_ins_main}')
    try:
        while True:
            time.sleep(0.0000000001)
        
    except KeyboardInterrupt:
        stop_event.set()
        frame_collector_process.join()
        face_rec_process.join()
        if display_option == 1: 
            display_process.join()

    #endregion

