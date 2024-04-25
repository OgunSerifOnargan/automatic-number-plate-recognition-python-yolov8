from classes.face import face
from datetime import datetime
import supervision as sv
class person:
    def __init__(self, img, bbox):
        self.detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.img = img
        self.bbox = bbox
        # self.line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

        self.entranceTime = None
        self.exitTime = None
        self.face = face()
        self.name = None

    def set_findings(self):
        self.face.isFaceIdentifiedProperly = True
        self.name = self.face.faceProposal.name
        self.face.name = self.face.faceProposal.name
        self.face.img = self.face.faceProposal.img
        self.face.encodedVector = self.face.faceProposal.encodedVector
        self.face.bbox_defaultFrame = self.face.faceProposal.bbox_defaultFrame


    

