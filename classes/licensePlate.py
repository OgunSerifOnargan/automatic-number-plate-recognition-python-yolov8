class licensePlate:
    def __init__(self):

        self.proposal = None
        self.img = None
        self.bbox_coords = None
        self.licenseCode = None
        self.bbox_xyxy = None
        self.licensePlate_finalizer = ["", ""]
        self.unknown_count = 0 
        self.islicensePlateIdentifiedProperly = False
        self.identification_time = None

    def set_bbox_coords(self, bbox_coords):
        self.bbox_coords = bbox_coords

    def correct_letter_to_int(self, char):
        if char == "A":
            new_char = "4"

        if char == "B":
            new_char = "8"

        if char == "C":
            new_char = "0"

        if char == "D":
            new_char = "0"

        if char == "E":
            new_char = "8"

        if char == "F":
            new_char = "F"

        if char == "G":
            new_char = "6"

        if char == "H":
            new_char = "4"

        if char == "I":
            new_char = "1"
        if char == "J":
            new_char = "9"
        if char == "K":
            new_char = "K"
        if char == "L":
            new_char = "4"

        if char == "M":
            new_char = "M"
        
        if char == "N":
            new_char = "N"
        if char == "O":
            new_char = "0"
        if char == "P": 
            new_char = "P"
        if char == "R": 
            new_char = "R"
        if char == "S": 
            new_char = "5"
        if char == "T": 
            new_char = "T"
        if char == "U": 
            new_char = "0"
        
        if char == "V": 
            new_char = "0"
        if char == "Y": 
            new_char = "Y"
        if char == "Z": 
            new_char = "2"
        else:
            new_char = char
        return new_char
     
    def correct_int_to_letter(self, char):
        if char == "0":
            new_char = "D"

        if char == "1":
            new_char = "I"

        if char == "2":
            new_char = "Z"

        if char == "3":
            new_char = "B"

        if char == "4":
            new_char = "A"

        if char == "5":
            new_char = "S"

        if char == "6":
            new_char = "G"

        if char == "7":
            new_char = "T"

        if char == "8":
            new_char = "B"

        if char == "9":
            new_char = "J"

        return new_char
    




def crop_objects(image, boxes, tracker_ids):
    cropped_images = {}
    cropped_bbox = {}
    for box, tracker_id in zip(boxes, tracker_ids):
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_images[tracker_id] = cropped_image
        cropped_bbox[tracker_id] = [x_min, y_min, x_max, y_max]
    return cropped_images, cropped_bbox

def crop_licensePlate_proposals(frame, base_licensePlate_locations, trackerId):
    img_licensePlates_proposal = {}
    for i, licensePlate_location in enumerate(base_licensePlate_locations):
        if len([int(coord) for coord in licensePlate_location]):
            x_max, y_max, x_min, y_min = [int(coord) for coord in licensePlate_location]

            img_licensePlate = frame[y_min:y_max, x_min:x_max]
            bboxes = [x_min, y_min, x_max, y_max]
            if trackerId in img_licensePlates_proposal:
                img_licensePlates_proposal[trackerId].append([bboxes, img_licensePlate])
            else:
                img_licensePlates_proposal[trackerId] = [[bboxes, img_licensePlate]]

    return img_licensePlates_proposal

