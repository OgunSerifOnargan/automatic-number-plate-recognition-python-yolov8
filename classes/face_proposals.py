class faceProposal:
    def __init__(self):
        self.bbox = None
        self.bbox_defaultFrame = None
        self.bbox_dlib = None
        self.img = None
        self.encodedVector = None
        self.name = None
    
    def set_bbox_defaultFrame(self, bbox_person):
        x1_base_face = bbox_person[0] + self.bbox[0]  
        y1_base_face = bbox_person[1] + self.bbox[1]
        x2_base_face = bbox_person[0] + self.bbox[2]
        y2_base_face = bbox_person[1] + self.bbox[3]
        self.bbox_defaultFrame = (x1_base_face, y1_base_face, x2_base_face, y2_base_face)

    def crop_and_set_img_faceProposal(self, frame):
        x_min, y_min, x_max, y_max = [int(coord) for coord in self.bbox_defaultFrame]
        img_face = frame[y_min:y_max, x_min:x_max]
        self.img = img_face