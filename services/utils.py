import time
import uuid
from classes.vehicle import vehicle
import csv
import os
from datetime import datetime, timedelta
import cv2

def initialize_vehicles(vehicles_dict, trackerId, img_vehicle, bbox_vehicle, lines_cv):
    vehicles_dict[trackerId] = vehicle(trackerId, img_vehicle, bbox_vehicle, lines_cv)
    append_string_to_csv(f"A vehicle is detected. tracker_id: {trackerId}.", 'log.csv')
#    return vehicles_dict
        
def rect_to_xyxy(bbox_licensePlate_proposal):
    top, right, bottom, left = bbox_licensePlate_proposal[0], bbox_licensePlate_proposal[1], bbox_licensePlate_proposal[2], bbox_licensePlate_proposal[3]
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
    last_time, last_licensePlate = "",""
    try:
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                last_time, last_licensePlate = row[0], row[1]  # Extract time and string info from the last row
    except FileNotFoundError:
        pass  # If the file doesn't exist, return empty string
    return last_time, last_licensePlate

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

def get_objects_within_time_interval(vehicles, interval_seconds):
    current_time = datetime.now()
    objects_within_interval = []
    for trackerId, vehicle_obj in vehicles.items():
        try:
            if vehicle_obj['identificationTime'] is not None:
                detection_time = datetime.strptime(vehicle_obj.licensePlate['identificationTime'], "%Y-%m-%d %H:%M:%S")
                time_difference = current_time - detection_time
                if time_difference.total_seconds() <= interval_seconds:
                    objects_within_interval.append(vehicle_obj)
        except:
            if vehicle_obj.identificationTime is not None:
                detection_time = datetime.strptime(vehicle_obj.identificationTime, "%Y-%m-%d %H:%M:%S")
                time_difference = current_time - detection_time
                if time_difference.total_seconds() <= interval_seconds:
                    objects_within_interval.append(vehicle_obj)
    return objects_within_interval

def update_licensePlateId_results(update_elements, vehicles):
    trackerId, vehicle_licensePlate_licensePlate_finalizer, vehicle_islicensePlateIdentifiedProperly, vehicle_identificationTime, vehicle_licenseCode, vehicle_img_skewed_plate, vehicle_licensePlate_licenseCode = update_elements
    vehicle = vehicles[trackerId]
    if vehicle_identificationTime is not None:
        vehicle.identificationTime = vehicle_identificationTime
    if vehicle_licenseCode is not None:
        vehicle.licenseCode = vehicle_licenseCode
    if vehicle_licensePlate_licenseCode is not None:
        vehicle.licensePlate.licenseCode = vehicle_licensePlate_licenseCode
    if vehicle_img_skewed_plate is not None:
        vehicle.img_skewed_plate = vehicle_img_skewed_plate
    vehicle.licensePlate.licensePlate_finalizer = vehicle_licensePlate_licensePlate_finalizer
    vehicle.licensePlate.islicensePlateIdentifiedProperly = vehicle_islicensePlateIdentifiedProperly
    vehicles[trackerId] = vehicle
    return vehicles

def convert_xywh_to_xyxy(bbox_licensePlate_proposal, vehicle):
    x1 = bbox_licensePlate_proposal['facial_area']['x']
    y1 = bbox_licensePlate_proposal['facial_area']['y']
    x2 = x1 + bbox_licensePlate_proposal['facial_area']['w']
    y2 = y1 + bbox_licensePlate_proposal['facial_area']['h']
    vehicle.licensePlate.licensePlateProposal.bbox_xyxy = [x1, y1, x2, y2]
    return vehicle

def vehicle_cleaner_accordingtoTime(vehicles, threshold_time):
    # Calculate the datetime threshold for the last 5 minutes
    threshold_time = datetime.now() - timedelta(minutes=threshold_time)

    # List to store keys of objects to remove
    keys_to_remove = []

    # Iterate over the original dictionary
    for trackerId, vehicle_obj in vehicles.items():
        # Convert detection time string to datetime object
        detection_time = datetime.strptime(vehicle_obj.detection_time, '%Y-%m-%d %H:%M:%S')
        
        # Check if the detection time is older than the threshold
        if detection_time < threshold_time:
            # Add the key to the list of keys to remove
            keys_to_remove.append(trackerId)

    # Remove objects from the dictionary based on keys in keys_to_remove
    for trackerId in keys_to_remove:
        del vehicles[trackerId]

def get_video_properties(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None
    
    # Get the FPS (frames per second)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Get the frame width
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Get the frame height
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Release the video capture object
    video.release()
    
    return fps, width, height