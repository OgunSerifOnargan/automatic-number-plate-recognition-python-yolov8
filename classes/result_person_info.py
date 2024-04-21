import uuid, requests, cv2, base64, cv2, sys, json
from global_functions.app_constants import PUBLIC_TOKEN, CRF_FIELD_FILE_ID, CRF_FIELD_TEXT_ID, CRF_FORMS_ID
from global_functions.app_constants import base_url

API_URL = base_url
class result_person_info:

    def __init__(self, person_obj):
        self.face_img = person_obj.face.faceProposal.img
        self.face_bbox = person_obj.face.faceProposal.bbox_defaultFrame
        self.encodedVector = person_obj.face.faceProposal.encodedVector
        self.faceName = person_obj.name
        self.name = person_obj.name

        self.body_img = None
        self.body_info = None
        self.uid = generate_uid()
        self.img_base64 = None

    def construct_body_info(self):
        body = {
            "PostCrFormAnswers": [
                {
                    "ID": -1,
                    "CRF_FIELDS_ID": CRF_FIELD_TEXT_ID,
                    "CRF_FORMS_ID": CRF_FORMS_ID,
                    "ROWDATARAW": self.name,
                    "ROWDATARAW2": "",
                    "rowState": 0
                },
                {
                    "ID": -1,
                    "CRF_FIELDS_ID": CRF_FIELD_FILE_ID,
                    "CRF_FORMS_ID": CRF_FORMS_ID,
                    "ROWDATARAW": "",
                    "ROWDATARAW2": self.uid,
                    "rowState": 0
                }
            ],
            "PUBLICTOKEN": PUBLIC_TOKEN
        }
        self.body_info = body
    
    def set_img_base64(self):
        # Encode the image array to bytes
        _, buffer = cv2.imencode('.jpg', self.face_img)
        # Convert the bytes to base64 string
        base64_str = base64.b64encode(buffer).decode('utf-8')
        self.img_base64 = base64_str

    def construct_body_img(self):
        body = {
                "UploadList": [
                    {
                    "BatchID": self.uid,
                    "FileDetails": [
                        {
                        "FILENAME": f'{self.name}.png',
                        "RESOURCEID": CRF_FIELD_FILE_ID,
                        "IMGBASE64": self.img_base64,
                        "ARCHID": -1,
                        "ARCHIVECONTEXTID": -1
                        
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
            # print("POST request successful\n")
            # return the content of the response if the request was successful
            # use response.text instead of response.content to return a str object
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


