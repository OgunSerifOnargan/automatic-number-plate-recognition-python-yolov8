import face_recognition

class face:
    def __init__(self):

        self.faceProposal = None
        self.img = None
        self.bbox_coords = None
        self.encodedVector = None
        self.name = None
        
        self.face_finalizer = ["", "", ""]
        self.unknown_count = 0 
        self.isFaceIdentifiedProperly = False
        self.identification_time = None
    
    def set_encodedVector(self, encodedVector):
        self.encodedVector = encodedVector

    def set_bbox_coords(self, bbox_coords):
        self.bbox_coords = bbox_coords


def crop_face_proposals(frame, base_face_locations, trackerId):
    img_faces_proposal = {}
    for i, face_location in enumerate(base_face_locations):
        if len([int(coord) for coord in face_location]):
            x_max, y_max, x_min, y_min = [int(coord) for coord in face_location]

            img_face = frame[y_min:y_max, x_min:x_max]
            bboxes = [x_min, y_min, x_max, y_max]
            if trackerId in img_faces_proposal:
                img_faces_proposal[trackerId].append([bboxes, img_face])
            else:
                img_faces_proposal[trackerId] = [[bboxes, img_face]]

    return img_faces_proposal

def predict_face(self, img_vulnerable_body):
    if not img_vulnerable_body.shape[1] == 0:
        face_locations = face_recognition.face_locations(img_vulnerable_body, model="hog")
        return face_locations
    else:
        return None