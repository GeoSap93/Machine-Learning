# Import ImageAI
from imageai.Detection import ObjectDetection
import os
import tensorflow_datasets as tfds

# Load EMNIST dataset
emnist_dataset = tfds.load('emnist', split='balanced', as_supervised=True)

# Initialize the object detection model
detector = ObjectDetection()
model_path = 'Machine_Learning_Dataset'
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3.pt")
detector.loadModel()

# List of objects to detect
objects_to_detect = [
    "laptop", "monitor", "keyboard", "mouse", "football", "bee", "train", "letter m", "letter t"
]


# Function to detect objects in an image
def detect_objects(image_path):
    detections = detector.detectObjectsFromImage(input_image=image_path, output_image_path="image-new.jpg",
                                                 minimum_percentage_probability=30)

    detected_objects = []
    object_count = {obj: 0 for obj in objects_to_detect}

    #Objects we detect that are not in our list of objects
    other_objects= []

    for eachObject in detections:
        object_name = eachObject["name"].lower()
        # Store objects from the specified list by using .append() method to add the items in the list detected_objects
        detected_objects.append(object_name)

        # Count objects from the specified list
        if object_name in objects_to_detect:
            object_count[object_name] += 1
        else:
        # Store objects that are not from the specified list by using .append() method in the list other_objects
            other_objects.append(object_name)

    # The len() funtion counts the number of items in the list "other_objects"
    object_count["Other"] = len(other_objects)

    return detected_objects, object_count, other_objects

# Example usage
image_path = "Machine_Learning_Dataset"
detected_objects, object_count, other_objects = detect_objects(image_path)

print("Detected Objects:")
print(detected_objects)

print("\nObject Count:")
for obj, count in object_count.items():
    print(f"{obj.capitalize()}: {count}")

print ("\nOther Objects Detected not present in the list:")
print (other_objects)
