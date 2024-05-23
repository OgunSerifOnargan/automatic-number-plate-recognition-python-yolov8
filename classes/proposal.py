import string
import re
class proposal:
    def __init__(self):
        #@processing
        self.bbox_xyxy = None
        self.bbox_defaultFrame_xyxy = None
        self.img = None
        self.img_skewed_plate = None

        self.licenseCode = None
        self.cityCode = None
        self.letterCode = None
        self.numCode = None
        self.isPlateReadProperly = False

        #ULTRALıghtWeight


    def crop_and_set_img_proposal(self, frame):
        x_min, y_min, x_max, y_max = [int(coord) for coord in self.bbox_defaultFrame]
        img_licensePlate = frame[y_min:y_max, x_min:x_max]
        self.img = img_licensePlate


    def crop_and_set_img_proposal_xyxy(self, frame):
        x_min, y_min, x_max, y_max = [int(coord) for coord in self.bbox_xyxy]
        img_licensePlate = frame[y_min:y_max, x_min:x_max]
        self.img = img_licensePlate

    
    def code_parts_initilizer(self):
        if "." in self.licenseCode:
            replaceddot = self.licenseCode.replace(".", " ")
            replaceddot = replaceddot.replace("  ", " ")
            self.licenseCode = replaceddot
        pattern = r'[^\w\s]'
        if bool(re.search(pattern, self.licenseCode)):
            self.licenseCode = None
            self.cityCode = None
            self.letterCode = None
            self.numCode = None
        # if self.licenseCode[-1].isalpha():
        #     self.licenseCode = self.licenseCode[:-1]
        if self.licenseCode != None:
            if len(self.licenseCode.split(" ")) == 3:
                self.cityCode = self.licenseCode.split(" ")[0]
                self.letterCode = self.licenseCode.split(" ")[1]
                self.numCode = self.licenseCode.split(" ")[2]
            else:
                self.icenseCode = None
                self.cityCode = None
                self.letterCode = None
                self.numCode = None

    def turkishlicenseCorrection(self):
        if self.cityCode != None and self.letterCode != None and self.numCode != None:
            if not self.cityCode.isdigit():
                for i, char in enumerate(self.cityCode):
                    if char.isalpha():
                        newNum = self.correct_letter_to_int(char)
                        new_cityCode = self.cityCode[:i] + newNum + self.cityCode[i + 1:]
                        self.cityCode = new_cityCode
            if self.cityCode is not None:
                if len(self.cityCode) == 3:
                    self.cityCode = self.cityCode[1:]

            if self.cityCode.isdigit() and len(self.cityCode) == 2:
                pass
            else:
                self.licenseCode = None
                self.cityCode = None
                self.letterCode = None
                self.numCode = None

            if self.letterCode != None:
                if not self.letterCode.isalpha():    
                    for i, char in enumerate(self.letterCode):
                        if char.isdigit():
                            newChar = self.correct_int_to_letter(char)
                            new_letterCode = self.letterCode[:i] + newChar + self.letterCode[i + 1:]
                            print(f"{char} in letterCode is replaced with {newChar}")
                            if len(self.letterCode) <= 3 or len(self.letterCode) >= 1:
                                self.letterCode = new_letterCode
                            else:
                                self.licenseCode = None
                                self.cityCode = None
                                self.letterCode = None
                                self.numCode = None

            if self.numCode is not None and self.letterCode is not None:
                if not self.numCode.isdigit():
                    for i, char in enumerate(self.numCode):
                        if char.isalpha():
                            newNum = self.correct_letter_to_int(char)
                            self.numCode = self.numCode[:i] + newNum + self.numCode[i+1:]
                    if len(self.letterCode) == 1:
                        if len(self.numCode == 6):
                            self.numCode = self.numCode[:-1]
                        if len(self.numCode) != 5 or len(self.numCode) != 4:
                            self.licenseCode = None
                            self.cityCode = None
                            self.letterCode = None
                            self.numCode = None
                    if len(self.letterCode) == 2:
                        if len(self.numCode) != 3 or len(self.numCode) != 4:
                            self.licenseCode = None
                            self.cityCode = None
                            self.letterCode = None
                            self.numCode = None
                    if len(self.letterCode) == 3:
                        if len(self.numCode) != 2 or len(self.numCode) != 3:
                            if (len(self.numCode) != 2 and len(self.numCode) == 3) or (len(self.numCode) == 2 and len(self.numCode) != 3):
                                pass
                            else:
                                self.licenseCode = None
                                self.cityCode = None
                                self.letterCode = None
                                self.numCode = None
            if self.cityCode != None and self.letterCode != None and self.numCode != None:
                self.licenseCode = self.cityCode + self.letterCode + self.numCode
            else: 
                self.licenseCode = None
                self.cityCode = None
                self.letterCode = None
                self.numCode = None

    def correct_letter_to_int(self, char):
        if char == "A":
            new_char = "4"
        #TODO:rhese 2 are dummy. Will change
        elif char == "N":
            new_char = "M"
        
        elif char == "L":
            new_char = ""

        elif char == "B":
            new_char = "8"

        elif char == "C":
            new_char = "0"

        elif char == "D":
            new_char = "0"

        elif char == "E":
            new_char = "8"

        elif char == "F":
            new_char = "F"

        elif char == "G":
            new_char = "6"

        elif char == "H":
            new_char = "4"

        elif char == "I":
            new_char = "1"
        elif char == "J":
            new_char = "9"
        elif char == "K":
            new_char = "K"
        elif char == "L":
            new_char = "4"

        elif char == "M":
            new_char = "M"
        
        # if char == "N":
        #     new_char = "N"
        elif char == "O":
            new_char = "0"
        elif char == "P": 
            new_char = "P"
        elif char == "R": 
            new_char = "R"
        elif char == "S": 
            new_char = "5"
        elif char == "T": 
            new_char = "T"
        elif char == "U": 
            new_char = "0"
        
        elif char == "V": 
            new_char = "0"
        elif char == "Y": 
            new_char = "Y"
        elif char == "Z": 
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
    

