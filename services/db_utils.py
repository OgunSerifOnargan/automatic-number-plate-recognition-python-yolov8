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
        
        print("Person appended to JSON file:", file_path)
    else:
        print("Error: JSON file not found.")

def serialize_person(person_obj):
    return {
        "detection_time": person_obj.detection_time,
#        "img_person": person_obj.img.tolist(),
        "bbox_person": person_obj.bbox,
        "entranceTime": person_obj.entranceTime,
        "exitTime": person_obj.exitTime,
        "face": serialize_face(person_obj.face),  # Assuming serialize_face() is defined
        "name": person_obj.name
    }
def serialize_face(face_obj):
    return {
        "faceProposal": serialize_face_proposal(face_obj.faceProposal),
#        "img": face_obj.img.tolist(),
        "bbox_coords": face_obj.bbox_coords,
        #"encodedVector": face_obj.encodedVector.tolist(),
        "name": face_obj.name,
        "face_finalizer": face_obj.face_finalizer,
        "unknown_count": face_obj.unknown_count,
        "isFaceIdentifiedProperly": face_obj.isFaceIdentifiedProperly
    }

def serialize_face_proposal(face_proposal_obj):
    return {
        "bbox": face_proposal_obj.bbox,
        "bbox_defaultFrame": face_proposal_obj.bbox_defaultFrame,
        "bbox_dlib": face_proposal_obj.bbox_dlib,
#        "img": face_proposal_obj.img.tolist(),
        #"encodedVector": face_proposal_obj.encodedVector.tolist(),
        "name": face_proposal_obj.name,
        "yolo_bbox": face_proposal_obj.yolo_bbox,
        "dlib_bbox": face_proposal_obj.dlib_bbox,
        "bbox_defaultFrame_yolo": face_proposal_obj.bbox_defaultFrame_yolo
    }
def read_json_as_dict(file_path):
    # Read the JSON file and load its contents into a dictionary
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def append_item_to_json(tracker_id, person_obj, file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing JSON data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Serialize the person object
        serialized_person = serialize_person(person_obj)
        
        # Assign the new item to the JSON data
        data[str(tracker_id)] = serialized_person
        
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
