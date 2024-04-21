import time
import uuid
from classes.person import person
import csv
from datetime import datetime
import os

def initialize_people(people_dict, trackerId, img_person, bbox_person):
    people_dict[trackerId] = person(img_person, bbox_person)
    append_string_to_csv(f"A person is detected. tracker_id: {trackerId}.", 'log.csv')
    return people_dict
    
def update_people_img_bbox_info(people_dict, trackerId, img_person, bbox_person):
    people_dict[trackerId].img = img_person
    people_dict[trackerId].bbox = bbox_person
    append_string_to_csv(f"person {trackerId}'s image and bbox are updated.", 'log.csv')
    return people_dict
        
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
