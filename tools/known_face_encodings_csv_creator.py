import csv
import os
import face_recognition

def person_photo_registration(folder_path, output_csv):
    known_face_encodings = []
    known_face_names = []

    # Loop through all files in the folder
    for index, filename in enumerate(os.listdir(folder_path), start=1):
        if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
            # Load the image
            image_path = os.path.join(folder_path, filename)
            person_image = face_recognition.load_image_file(image_path)

            # Extract face encoding
            face_encodings = face_recognition.face_encodings(person_image)
            
            # Ensure that the image contains exactly one face
            if len(face_encodings) == 1:
                known_face_encodings.append(face_encodings[0])
                
                # Extract the name from the filename (excluding the extension)
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"Skipping {filename} as it doesn't contain exactly one face.")

    # Write the data to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Name', 'Encoding'])
        for index, (name, encoding) in enumerate(zip(known_face_names, known_face_encodings), start=1):
            writer.writerow([index, name, encoding])

# Example usage:
folder_path = 'known_faces'
output_csv = 'known_faces.csv'
person_photo_registration(folder_path, output_csv)
