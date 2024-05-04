import uuid, requests, cv2, base64, cv2, sys, json
from global_functions.app_constants import PUBLIC_TOKEN, CRF_FIELD_FILE_ID, CRF_FIELD_TEXT_ID, CRF_FORMS_ID
from global_functions.app_constants import base_url
from services.utils import generate_uid

API_URL = base_url
class result_person_info:

    def __init__(self, person_obj, camId):
        self.camId = camId
        self.face_img = person_obj.face.img
        self.face_img_base64 = _set_img_base64(person_obj.face.img)
        self.name = person_obj.name
        self.trackerId = person_obj.trackerId
        self.location_state = person_obj.location_state

        self.body_img = person_obj.img
        self.body_img_base64 = _set_img_base64(person_obj.img)
        self.uid_for_img_face = generate_uid()
        self.uid_for_img_body = generate_uid()

    def construct_body_info(self):
        print('3862:name, 3863:empty, 3866:trackerId, 3867:empty, 3868:location_state, 3863:img_face, 3867:img_body')
        body = {
                "PostCrFormAnswers": [
                    {
                    "ID": -1, #// her zaman -1
                    "CRF_FIELDS_ID": 3862,
                    "CRF_FORMS_ID": 414,
                    "ROWDATARAW": self.name,  #//// formun ilk fieldı, buraya label bilgisini yazacağız
                    "ROWDATARAW2": "",
                    "rowState": 0
                    },
                    {
                    "ID": -1, #// her zaman -1
                    "CRF_FIELDS_ID": 3863, #///Formun 2.fieldı buraya dosya göndereceğimin bilgisini yazacağım. 
                    "CRF_FORMS_ID": 414,
                    "ROWDATARAW": "", #//her zaman boş
                    "ROWDATARAW2": self.uid_for_img_face, #// göndermeden önce ürettiğimiz UID
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": 3866,
                    "CRF_FORMS_ID": 414,
                    "ROWDATARAW": str(self.trackerId), #// trackinID değeri neyse o, string alır
                    "ROWDATARAW2": "",
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": 3867,
                    "CRF_FORMS_ID": 414,
                    "ROWDATARAW": "",
                    "ROWDATARAW2": self.uid_for_img_body, #// göndermeden önce ürettiğimiz UID2 
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": 3868,
                    "CRF_FORMS_ID": 414,
                    "ROWDATARAW": self.location_state, #//in - out değeri, 0 out olsun 1 in olsun..
                    "ROWDATARAW2": "",
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": 3874,
                    "CRF_FORMS_ID": 414,
                    "ROWDATARAW": self.camId, #//cam ID
                    "ROWDATARAW2": "",
                    "rowState": 0
                    }

                ],
                "PUBLICTOKEN": "B4C58CBC-5385-4E61-B5C2-46CE5AB1A686"
                }
        self.body_info = body

    def construct_body_img(self):
        body = {
                "UploadList": [
                    {
                    "BatchID": self.uid_for_img_face, #// bu dosya gönderimi için yukarda oluşturduğun file uid
                    "FileDetails": [
                        {
                        "FILENAME": f"{self.name}_face.png",
                        "RESOURCEID": 3863, #// bu önemli, formun 2.fieldının ID’si
                        "IMGBASE64": self.face_img_base64,
                        "ARCHID": -1, #// her zaman -1
                        "ARCHIVECONTEXTID": -1, #// her zaman -1
                        
                        }
                    ]
                    },
                    {
                    "BatchID": self.uid_for_img_body, #// bu dosya gönderimi için yukarda oluşturduğun file uid
                    "FileDetails": [
                        {
                        "FILENAME": f"{self.name}_body.png",
                        "RESOURCEID": 3867, #// bu önemli, formun 4.fieldının ID’si
                        "IMGBASE64": self.body_img_base64,
                        "ARCHID": -1, #// her zaman -1
                        "ARCHIVECONTEXTID": -1, #// her zaman -1
                        
                        }
                    ]
                    }
                ]
        }
        self.body_img = body

    def send_post_request(self, url):
        final_url = API_URL + url
        # Request headers
        headers = {
            "Content-Type": "application/json"}
        try:
            # send the POST request with the provided URL and data
            if url == "PostCRFormAnswersPublic":
                self.body_info_json = json.dumps(self.body_info, cls=self.CustomEncoder)
                response = requests.post(final_url, data=self.body_info_json, headers=headers, verify=False)
            elif url == "PostCRFormAnswersPublicFileUpload":
                self.body_img_json = json.dumps(self.body_img, cls=self.CustomEncoder)
                response = requests.post(final_url, data=self.body_img_json, headers=headers, verify=False)
        except requests.ConnectionError as e:
            print("OOPS!! Connection Error. Make sure you are connected to Internet. Technical Details given below.\n")
            print(str(e))
            sys.exit()
        except requests.Timeout as e:
            print("OOPS!! Timeout Error")
            print(str(e))
            sys.exit()
        except requests.RequestException as e:
            print("OOPS!! General Error")
            print(str(e))
            sys.exit()

        if response.ok:  # check if the response was successful
            return response.text
        else:
            print("POST request failed\n")
            print(response.text)
            sys.exit()


    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, result_person_info):
                return obj

            return json.JSONEncoder.default(self, obj)
            
def generate_uid():
    uid = str(uuid.uuid4())
    return uid


def post_results_MP(stop_event, post_queue):
    while not stop_event.is_set():
        if not post_queue.empty():
            info = post_queue.get()
            info.construct_body_info()
            info.construct_body_img()
            info.send_post_request('PostCRFormAnswersPublic')
            info.send_post_request('PostCRFormAnswersPublicFileUpload')



def display_frames(stop_event, display_queue):
    while not stop_event.is_set():
        if not display_queue.empty():
            display_frame = display_queue.get()
            cv2.imshow('Frame', display_frame)
            cv2.waitKey(1)

def _set_img_base64(img):
    # Encode the image array to bytes
    _, buffer = cv2.imencode('.jpg', img)
    # Convert the bytes to base64 string
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str