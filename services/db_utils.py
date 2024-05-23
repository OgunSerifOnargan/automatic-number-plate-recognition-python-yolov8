import json
import os


def create_json(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        # Write an empty JSON object to the file
        with open(file_path, 'w') as json_file:
            json.dump({}, json_file)
        print("Empty JSON file created:", file_path)
    else:
        print("db_json already exists:", file_path)

def append_object_to_json(file_path, obj):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing JSON data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Find the maximum negative key value
        max_key = max(data.keys(), default=0)
        if max_key < 0:
            new_key = max_key - 1
        else:
            new_key = -1
        
        # Assign the new object to the JSON data
        data[f'{new_key}'] = obj
        
        # Write the updated JSON data back to the file
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        
        print("vehicle appended to JSON file:", file_path)
    else:
        print("Error: JSON file not found.")

def serialize_vehicle(vehicle_obj):
    return {
        "detection_time": vehicle_obj.detection_time,
#        "img_vehicle": vehicle_obj.img.tolist(),
        "bbox_vehicle": vehicle_obj.bbox,
        "entranceTime": vehicle_obj.entranceTime,
        "exitTime": vehicle_obj.exitTime,
        "licensePlate": serialize_licensePlate(vehicle_obj.licensePlate),  # Assuming serialize_licensePlate() is defined
        "licenseCode": vehicle_obj.licenseCode
    }
def serialize_licensePlate(licensePlate_obj):
    return {
        "licensePlateProposal": serialize_licensePlate_proposal(licensePlate_obj.licensePlateProposal),
#        "img": licensePlate_obj.img.tolist(),
        "bbox_coords": licensePlate_obj.bbox_coords,
        #"encodedVector": licensePlate_obj.encodedVector.tolist(),
        "licenseCode": licensePlate_obj.licenseCode,
        "licensePlate_finalizer": licensePlate_obj.licensePlate_finalizer,
        "unknown_count": licensePlate_obj.unknown_count,
        "islicensePlateIdentifiedProperly": licensePlate_obj.islicensePlateIdentifiedProperly
    }

def serialize_licensePlate_proposal(licensePlate_proposal_obj):
    return {
        "bbox": licensePlate_proposal_obj.bbox,
        "bbox_defaultFrame": licensePlate_proposal_obj.bbox_defaultFrame,
        "bbox_dlib": licensePlate_proposal_obj.bbox_dlib,
#        "img": licensePlate_proposal_obj.img.tolist(),
        #"encodedVector": licensePlate_proposal_obj.encodedVector.tolist(),
        "licenseCode": licensePlate_proposal_obj.licenseCode,
        "yolo_bbox": licensePlate_proposal_obj.yolo_bbox,
        "dlib_bbox": licensePlate_proposal_obj.dlib_bbox,
        "bbox_defaultFrame_yolo": licensePlate_proposal_obj.bbox_defaultFrame_yolo
    }
def read_json_as_dict(file_path):
    # Read the JSON file and load its contents into a dictionary
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def append_item_to_json(tracker_id, vehicle_obj, file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing JSON data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Serialize the vehicle object
        serialized_vehicle = serialize_vehicle(vehicle_obj)
        
        # Assign the new item to the JSON data
        data[str(tracker_id)] = serialized_vehicle
        
        # Write the updated JSON data back to the file
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        
        print("Item appended to JSON file:", file_path)
    else:
        print("Error: JSON file not found.")

def reassign_tracker_ids(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing JSON data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Find the smallest key value (trackerId)
        start_id = min(data.keys(), default='-1')  # Use string '-1' instead of integer -1
        
        # Convert start_id to integer before subtracting 1
        start_id = int(start_id) - 1
        
        # Reassign trackerIds starting from the smallest key value
        new_data = {}
        id_counter = start_id
        for old_id in sorted(data.keys(), reverse=True):
            new_data[str(id_counter)] = data[old_id]  # Convert id_counter to string
            id_counter -= 1
        
        # Write the updated JSON data back to the file
        with open(file_path, 'w') as json_file:
            json.dump(new_data, json_file)
        
        print("TrackerIds reassigned in JSON file:", file_path)
    else:
        print("Error: JSON file not found.")
