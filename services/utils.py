import time
import uuid
from classes.person import person
import csv
from datetime import datetime
import os
from datetime import datetime, timedelta

def initialize_people(people_dict, trackerId, img_person, bbox_person, lines_cv):
    people_dict[trackerId] = person(trackerId, img_person, bbox_person, lines_cv)
    append_string_to_csv(f"A person is detected. tracker_id: {trackerId}.", 'log.csv')
#    return people_dict
        
def rect_to_xyxy(bbox_face_proposal):
    top, right, bottom, left = bbox_face_proposal[0], bbox_face_proposal[1], bbox_face_proposal[2], bbox_face_proposal[3]
    x1 = left
    y1 = top
    x2 = right
    y2 = bottom
    xyxy_format = (x1, y1, x2, y2)
    return xyxy_format

def get_rows_in_interval(csv_file):
    if not os.path.isfile(csv_file):
        print(f"File '{csv_file}' does not exist. Creating...")
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Data'])  # Write header to the CSV file
            print(f"File '{csv_file}' created.")
    time1 = datetime.now()
    rows_in_interval = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            time_str = row['Time']
            time2 = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

            difference = (time1 - time2).total_seconds()
            if difference < 60:
                rows_in_interval.append(row)
    return rows_in_interval

def get_last_row_info(csv_file):
    last_time, last_face = "",""
    try:
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                last_time, last_face = row[0], row[1]  # Extract time and string info from the last row
    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty string
    return last_time, last_face

def is_time_outside_interval(last_time):   
    current_time = time.strftime('%H:%M:%S', time.localtime()) 
    if last_time:
        last_time = datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
        current_time = datetime.strptime(current_time, "%H:%M:%S")
        difference = (current_time - last_time).total_seconds()
        return difference > 360
    # else:
    #     return True  # If there's no last time recorded, return True

def append_string_to_csv(input_string, csv_file):
    # Get current time in HH:MM:SS format
    current_datetime = datetime.now()
    current_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # Check if the CSV file already exists
    file_exists = True
    try:
        with open(csv_file, 'r') as f:
            pass
    except FileNotFoundError:
        file_exists = False

    # Write string with time to CSV file
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If file doesn't exist, write the header
        if not file_exists:
            writer.writerow(["Time", "Data"])  # Header
            
        # Write time and string as a new row
        writer.writerow([current_time, input_string])

def generate_uid():
    """
    Generate a UUID (Universally Unique Identifier).
    """
    return uuid.uuid4()

def get_objects_within_time_interval(people, interval_seconds):
    current_time = datetime.now()
    objects_within_interval = []
    for trackerId, person_obj in people.items():
        try:
            if person_obj.face['identification_time'] is not None:
                detection_time = datetime.strptime(person_obj.face['identification_time'], "%Y-%m-%d %H:%M:%S")
                time_difference = current_time - detection_time
                if time_difference.total_seconds() <= interval_seconds:
                    objects_within_interval.append(person_obj)
        except:
            if person_obj.face.identification_time is not None:
                detection_time = datetime.strptime(person_obj.face.identification_time, "%Y-%m-%d %H:%M:%S")
                time_difference = current_time - detection_time
                if time_difference.total_seconds() <= interval_seconds:
                    objects_within_interval.append(person_obj)
    return objects_within_interval

def update_faceId_results(update_elements, people):
    trackerId, person_face_face_finalizer, person_face_isFaceIdentifiedProperly, person_face_identification_time, person_name, person_face_name, person_face_img, person_face_encodedVector = update_elements
    person = people[trackerId]
    if person_face_identification_time is not None:
        person.face.identification_time = person_face_identification_time
    if person_name is not None:
        person.name = person_name
    if person_face_name is not None:
        person.face.name = person_face_name
    if person_face_img is not None:
        person.face.img = person_face_img
    if person_face_encodedVector is not None:
        person.face.encodedVector = person_face_encodedVector
    person.face.face_finalizer = person_face_face_finalizer
    person.face.isFaceIdentifiedProperly = person_face_isFaceIdentifiedProperly
    people[trackerId] = person

def convert_xywh_to_xyxy(bbox_face_proposal, person):
    x1 = bbox_face_proposal['facial_area']['x']
    y1 = bbox_face_proposal['facial_area']['y']
    x2 = x1 + bbox_face_proposal['facial_area']['w']
    y2 = y1 + bbox_face_proposal['facial_area']['h']
    person.face.faceProposal.bbox_xyxy = [x1, y1, x2, y2]
    return person

def people_cleaner_accordingtoTime(people, threshold_time):
    # Calculate the datetime threshold for the last 5 minutes
    threshold_time = datetime.now() - timedelta(minutes=threshold_time)

    # List to store keys of objects to remove
    keys_to_remove = []

    # Iterate over the original dictionary
    for trackerId, person_obj in people.items():
        # Convert detection time string to datetime object
        detection_time = datetime.strptime(person_obj.detection_time, '%Y-%m-%d %H:%M:%S')
        
        # Check if the detection time is older than the threshold
        if detection_time < threshold_time:
            # Add the key to the list of keys to remove
            keys_to_remove.append(trackerId)

    # Remove objects from the dictionary based on keys in keys_to_remove
    for trackerId in keys_to_remove:
        del people[trackerId]