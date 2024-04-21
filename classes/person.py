from classes.face import face


class person:
    def __init__(self, img, bbox):
        
        self.img_person = img
        self.bbox_person = bbox
        self.entranceTime = None
        self.exitTime = None
        self.face = face()
        self.name = None
    def set_findings(self):
        self.name = self.face.faceProposal.name
        self.face.name = self.face.faceProposal.name
        self.face.img = self.face.faceProposal.img
        self.face.encodedVector = self.face.faceProposal.encodedVector
        self.face.bbox_defaultFrame = self.face.faceProposal.bbox_defaultFrame
        
    

