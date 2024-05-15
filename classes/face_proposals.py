class faceProposal:
    def __init__(self):
        #DLIB
        self.bbox_xyxy = None
        self.bbox_defaultFrame = None
        self.bbox_dlib = None
        self.img = None
        self.encodedVector = None
        self.name = None
        #YOLO
        self.yolo_bbox = None
        self.dlib_bbox = None
        self.bbox_defaultFrame_yolo = None
        self.dlib_usage_bbox = None
        #ULTRALıghtWeight

    def set_bbox_defaultFrame(self, bbox_person):
        x1_base_face = bbox_person[0] + self.bbox_xyxy[0]  
        y1_base_face = bbox_person[1] + self.bbox_xyxy[1]
        x2_base_face = bbox_person[0] + self.bbox_xyxy[2]
        y2_base_face = bbox_person[1] + self.bbox_xyxy[3]
        self.bbox_defaultFrame = (x1_base_face, y1_base_face, x2_base_face, y2_base_face)

    def crop_and_set_img_faceProposal(self, frame):
        x_min, y_min, x_max, y_max = [int(coord) for coord in self.bbox_defaultFrame]
        img_face = frame[y_min:y_max, x_min:x_max]
        self.img = img_face

    def yolo_to_top_right_bottom_left(self):
        x_min, y_min, x_max, y_max = self.yolo_bbox
        top = int(y_min)
        right = int(x_max)
        bottom = int(y_max)
        left = int(x_min)
        self.bbox_dlib = (top, right, bottom, left)

    def crop_and_set_img_faceProposal_yolo(self, frame):
        x_min, y_min, x_max, y_max = [int(coord) for coord in self.bbox_defaultFrame_yolo]
        img_face = frame[y_min:y_max, x_min:x_max]
        self.img = img_face

    def set_bbox_defaultFrame_yolo(self, bbox_person):
        x1_base_face = bbox_person[0] + self.yolo_bbox[0]  
        y1_base_face = bbox_person[1] + self.yolo_bbox[1]
        x2_base_face = bbox_person[0] + self.yolo_bbox[2]
        y2_base_face = bbox_person[1] + self.yolo_bbox[3]
        self.bbox_defaultFrame_yolo = (x1_base_face, y1_base_face, x2_base_face, y2_base_face)
